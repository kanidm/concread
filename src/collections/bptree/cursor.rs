// The cursor is what actually knits a tree together from the parts
// we have, and has an important role to keep the system consistent.
//
// Additionally, the cursor also is responsible for general movement
// throughout the structure and how to handle that effectively

use super::node::{ABNode, Node};
use std::collections::LinkedList;
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;

use super::leaf::Leaf;
// use super::branch::Branch;
use super::states::{BLInsertState, BNClone, BRInsertState, CRInsertState};

#[derive(Debug)]
pub(crate) struct CursorRead<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    root: ABNode<K, V>,
}

#[derive(Debug)]
pub(crate) struct CursorWrite<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    // Need to build a stack as we go - of what, I'm not sure ...
    txid: usize,
    root: ABNode<K, V>,
}

impl<K: Clone + Ord + Debug, V: Clone> CursorWrite<K, V> {
    pub(crate) fn new(root: ABNode<K, V>) -> Self {
        let txid = root.txid + 1;
        // TODO: Check that txid < usize max.
        assert!(txid < usize::max_value());

        CursorWrite {
            txid: txid,
            root: root,
        }
    }

    pub(crate) fn finalise(self) -> ABNode<K, V> {
        // Return the new root for replacement into the txn manager.
        self.root
    }

    // Functions as insert_or_update
    pub(crate) fn insert(&mut self, k: K, v: V) -> Option<V> {
        match clone_and_insert(&mut self.root, self.txid, k, v) {
            CRInsertState::NoClone(res) => res,
            CRInsertState::Clone(res, mut nnode) => {
                // We have a new root node, swap it in.
                mem::swap(&mut self.root, &mut nnode);
                // Return the insert result
                res
            }
            CRInsertState::CloneSplit(lnode, rnode) => {
                // The previous root had to split - make a new
                // root now and put it inplace.
                let mut nroot = Node::new_branch(self.txid, lnode, rnode);
                mem::swap(&mut self.root, &mut nroot);
                // As we split, there must NOT have been an existing
                // key to overwrite.
                None
            }
            CRInsertState::Split(rnode) => {
                // The previous root was already part of this txn, but has now
                // split. We need to construct a new root and swap them.
                //
                // Note, that we have to briefly take an extra RC on the root so
                // that we can get it into the branch.
                let mut nroot = Node::new_branch(self.txid, self.root.clone(), rnode);
                mem::swap(&mut self.root, &mut nroot);
                // As we split, there must NOT have been an existing
                // key to overwrite.
                None
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn root_txid(&self) -> usize {
        self.root.txid
    }

    // Should probably be in the read trait.
    #[cfg(test)]
    pub(crate) fn verify(&self) -> bool {
        self.root.verify()
    }
}

impl<K: Clone + Ord + Debug, V: Clone> CursorRead<K, V> {
    pub(crate) fn new(root: ABNode<K, V>) -> Self {
        CursorRead { root: root }
    }
}

fn clone_and_insert<K: Clone + Ord + Debug, V: Clone>(
    node: &mut ABNode<K, V>,
    txid: usize,
    k: K,
    v: V,
) -> CRInsertState<K, V> {
    /*
     * Let's talk about the magic of this function. Come, join
     * me around the [🔥🔥🔥]
     *
     * This function is the heart and soul of a copy on write
     * structure - as we progress to the leaf location where we
     * wish to perform an alteration, we clone (if required) all
     * nodes on the path. This way an abort (rollback) of the
     * commit simply is to drop the cursor, where the "new"
     * cloned values are only referenced. To commit, we only need
     * to replace the tree root in the parent structures as
     * the cloned path must by definition include the root, and
     * will contain references to nodes that did not need cloning,
     * thus keeping them alive.
     *
     *
     */

    if node.is_leaf() {
        // Leaf path
        if node.txid == txid {
            // No clone required.
            // simply do the insert.
            let mref = Arc::get_mut(node).unwrap();
            match mref.as_mut_leaf().insert_or_update(k, v) {
                BLInsertState::Ok(res) => CRInsertState::NoClone(res),
                BLInsertState::Split(sk, sv) => {
                    // We split, but left is already part of the txn group, so lets
                    // just return what's new.
                    let rnode = Node::new_leaf_ins(txid, sk, sv);
                    CRInsertState::Split(rnode)
                }
            }
        } else {
            // Clone required.
            let mut cnode = node.req_clone(txid);
            let mref = Arc::get_mut(&mut cnode).unwrap();
            // insert to the new node.
            match mref.as_mut_leaf().insert_or_update(k, v) {
                BLInsertState::Ok(res) => CRInsertState::Clone(res, cnode),
                BLInsertState::Split(sk, sv) => {
                    let rnode = Node::new_leaf_ins(txid, sk, sv);
                    CRInsertState::CloneSplit(cnode, rnode)
                }
            }
        }
    } else {
        // Branch path
        // The branch path is a bit more reactive than the leaf path, as we trigger
        // the leaf op, and then decide what we need to do.
        //
        // - locate the node we need to work on.
        let node_txid = node.txid;
        let nmref = Arc::get_mut(node).unwrap().as_mut_branch();
        let anode_idx = nmref.locate_node(&k);
        let mut anode = nmref.get_mut_idx(anode_idx);
        match clone_and_insert(&mut anode, txid, k, v) {
            CRInsertState::Clone(_, _) => {
                #[cfg(test)]
                println!("TEMP -> {:?}", anode.nid);
                unimplemented!();
            }
            CRInsertState::NoClone(res) => {
                // If our descendant did not clone, then we don't have to either.
                debug_assert!(txid == node.txid);
                CRInsertState::NoClone(res)
            }
            CRInsertState::Split(_) => {
                #[cfg(test)]
                println!("TEMP -> {:?}", anode.nid);
                unimplemented!();
            }
            CRInsertState::CloneSplit(lnode, rnode) => {
                // So we updated our descendant at anode_idx, and it cloned and split.
                // This means we need to take a number of steps.

                // First, we need to determine if we require cloning.
                if txid == node_txid {
                    // work inplace.
                    // Second, we update anode_idx node with our lnode as the new clone.
                    nmref.replace_by_idx(anode_idx, lnode);

                    // Third we insert rnode - perfect world it's at anode_idx + 1, but
                    // we use the normal insert routine for now.
                    unimplemented!();
                } else {
                    let mut cnode = node.req_clone(txid);
                    let nmref = Arc::get_mut(&mut cnode).unwrap().as_mut_branch();

                    // Second, we update anode_idx node with our lnode as the new clone.
                    nmref.replace_by_idx(anode_idx, lnode);

                    // Third we insert rnode - perfect world it's at anode_idx + 1, but
                    // we use the normal insert routine for now.
                    match nmref.add_node(rnode) {
                        BRInsertState::Ok => CRInsertState::Clone(None, cnode),
                        BRInsertState::Split(clnode, crnode) => {
                            // Create a new branch to hold these children.
                            let nrnode = Node::new_branch(txid, clnode, crnode);
                            // Return it
                            CRInsertState::CloneSplit(cnode, nrnode)
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::constants::L_CAPACITY;
    use super::super::leaf::Leaf;
    use super::super::node::{ABNode, Node};
    use super::CursorWrite;
    use std::sync::Arc;

    fn create_leaf_node(v: usize) -> ABNode<usize, usize> {
        let mut node = Arc::new(Box::new(Node::new_leaf(0)));
        {
            let nmut = Arc::get_mut(&mut node).unwrap().as_mut().as_mut_leaf();
            nmut.insert_or_update(v, v);
        }
        node
    }

    fn create_leaf_node_full(vbase: usize) -> ABNode<usize, usize> {
        assert!(vbase % 10 == 0);
        let mut node = Arc::new(Box::new(Node::new_leaf(0)));
        {
            let nmut = Arc::get_mut(&mut node).unwrap().as_mut().as_mut_leaf();
            for idx in 0..L_CAPACITY {
                let v = vbase + idx;
                nmut.insert_or_update(v, v);
            }
        }
        node
    }

    #[test]
    fn test_bptree_cursor_insert_leaf() {
        // First create the node + cursor
        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node);
        let prev_txid = wcurs.root_txid();

        // Now insert - the txid should be different.
        let r = wcurs.insert(1, 1);
        assert!(r.is_none());
        let r1_txid = wcurs.root_txid();
        assert!(r1_txid == prev_txid + 1);

        // Now insert again - the txid should be the same.
        let r = wcurs.insert(2, 2);
        assert!(r.is_none());
        let r2_txid = wcurs.root_txid();
        assert!(r2_txid == r1_txid);
        // The clones worked as we wanted!
        assert!(wcurs.verify());
    }

    #[test]
    fn test_bptree_cursor_insert_split_1() {
        // Given a leaf at max, insert such that:
        //
        // leaf
        //
        // leaf -> split leaf
        //
        //
        //      root
        //     /    \
        //  leaf    split leaf
        //
        // It's worth noting that this is testing the CloneSplit path
        // as leaf needs a clone AND to split to achieve the new root.

        let node = create_leaf_node_full(10);
        let mut wcurs = CursorWrite::new(node);
        let prev_txid = wcurs.root_txid();

        let r = wcurs.insert(1, 1);
        assert!(r.is_none());
        let r1_txid = wcurs.root_txid();
        assert!(r1_txid == prev_txid + 1);
        assert!(wcurs.verify());
        println!("{:?}", wcurs);
    }

    #[test]
    fn test_bptree_cursor_insert_split_2() {
        // Similar to split_1, but test the Split only path. This means
        // leaf needs to be below max to start, and we insert enough in-txn
        // to trigger a clone of leaf AND THEN to cause the split.
        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node);

        for v in 1..(L_CAPACITY + 1) {
            println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        println!("{:?}", wcurs);
    }

    #[test]
    fn test_bptree_cursor_insert_split_3() {
        //      root
        //     /    \
        //  leaf    split leaf
        //       ^
        //        \----- nnode
        //
        //  Check leaf split inbetween l/sl (new txn)
        let lnode = create_leaf_node_full(10);
        let rnode = create_leaf_node_full(20);
        let root = Node::new_branch(0, lnode, rnode);
        let mut wcurs = CursorWrite::new(root);
        assert!(wcurs.verify());
        println!("{:?}", wcurs);

        let r = wcurs.insert(19, 19);
        assert!(r.is_none());
        assert!(wcurs.verify());
        println!("{:?}", wcurs);
    }

    #[test]
    fn test_bptree_cursor_insert_split_4() {
        //      root
        //     /    \
        //  leaf    split leaf
        //                       ^
        //                        \----- nnode
        //
        //  Check leaf split of sl (new txn)
        //
        let lnode = create_leaf_node_full(10);
        let rnode = create_leaf_node_full(20);
        let root = Node::new_branch(0, lnode, rnode);
        let mut wcurs = CursorWrite::new(root);
        assert!(wcurs.verify());

        let r = wcurs.insert(29, 29);
        assert!(r.is_none());
        assert!(wcurs.verify());
        println!("{:?}", wcurs);
    }

    #[test]
    fn test_bptree_cursor_insert_split_5() {
        //      root
        //     /    \
        //  leaf    split leaf
        //       ^
        //        \----- nnode
        //
        //  Check leaf split inbetween l/sl (same txn)
        //
    }

    #[test]
    fn test_bptree_cursor_insert_split_6() {
        //      root
        //     /    \
        //  leaf    split leaf
        //                       ^
        //                        \----- nnode
        //
        //  Check leaf split of sl (same txn)
        //
    }

    #[test]
    fn test_bptree_cursor_insert_stress_1() {
        // Insert ascending
    }

    #[test]
    fn test_bptree_cursor_insert_stress_2() {
        // Insert descending
    }

    #[test]
    fn test_bptree_cursor_insert_stress_3() {
        // Insert random
    }
}
