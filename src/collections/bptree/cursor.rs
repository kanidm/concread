// The cursor is what actually knits a tree together from the parts
// we have, and has an important role to keep the system consistent.
//
// Additionally, the cursor also is responsible for general movement
// throughout the structure and how to handle that effectively

use super::node::{ABNode, Node};
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;

// use super::branch::Branch;
use super::states::{BLInsertState, BRInsertState, CRInsertState};

#[derive(Debug)]
pub(crate) struct CursorRead<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    length: usize,
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
    length: usize,
    root: ABNode<K, V>,
}

pub(crate) trait CursorReadOps<K: Clone + Ord + Debug, V: Clone> {
    fn get_root_ref(&self) -> &ABNode<K, V>;

    fn len(&self) -> usize;

    fn get_tree_density(&self) -> (usize, usize) {
        // Walk the tree and calculate the packing effeciency.
        let rref = self.get_root_ref();
        rref.tree_density()
    }

    fn search<'a>(&'a self, k: &'a K) -> Option<&'a V> {
        // Search for and return if a value exists at key.
        let rref = self.get_root_ref();
        rref.get_ref(k)
    }

    fn contains_key(&self, k: &K) -> bool {
        match self.search(k) {
            Some(_) => true,
            None => false,
        }
    }
}

impl<K: Clone + Ord + Debug, V: Clone> CursorWrite<K, V> {
    pub(crate) fn new(root: ABNode<K, V>, length: usize) -> Self {
        let txid = root.txid + 1;
        // TODO: Check that txid < usize max.
        assert!(txid < usize::max_value());

        CursorWrite {
            txid: txid,
            length: length,
            root: root,
        }
    }

    pub(crate) fn finalise(self) -> (ABNode<K, V>, usize) {
        // Return the new root for replacement into the txn manager.
        (self.root, self.length)
    }

    pub(crate) fn clear(&mut self) {
        // Reset the values in this tree.
        let mut nroot = Node::new_ableaf(self.txid);
        mem::swap(&mut self.root, &mut nroot);
    }

    // Functions as insert_or_update
    pub(crate) fn insert(&mut self, k: K, v: V) -> Option<V> {
        let r = match clone_and_insert(&mut self.root, self.txid, k, v) {
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
        };
        // If this is none, it means a new slot is now occupied.
        if r.is_none() {
            self.length += 1;
        }
        r
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

    pub(crate) fn tree_density(&self) -> (usize, usize) {
        self.root.tree_density()
    }

    pub(crate) fn compact(&self) {
        // This will rebuild the tree in a compacted form. The idea is that instead of
        // rebalancing "all the time" via an amortized cost, we allow those ops to be
        // faster to pay a smaller "all at once" cost to compress if required. Generally
        // this is only triggered if tree_density is lower than 50%
        //
        // A traditional b+tree tries to balance all the time, such that when a node splits
        // it takes 50% of it's values. This means that generally with 3value leaves, your
        // tree only ends up at ~60% occupation at the leaves as each leave tends to 2/3
        // capacity. With a 7 value leaf, this tends to 4/7 (57%). That's a lot of excess
        // memory used!
        //
        // We instead do minimal splits, which means on a linear insert we have 100%
        // occupation at the leaves, and the branches tend to 60% on 3 value, and 85% on 7 value.
        //
        // The down side is on random inserts, we tend to be around 50%-60% population - the
        // same as a b+tree. On reverse order inserts we have about 20% occupation. Our branches
        // remain at the higher rates as above for random, and reverse shows the same low
        // occupation rates.
        //
        // Obviously we want to tend to >60% if possible. So instead of "amortizing" everything
        // to be 60%, and always balancing in the write, we allow a compact operation to be
        // called that will check the density of the tree, and then compact if required to
        // achieve near 100% occupation.
        //
        // Basically, this means you can chose to pay occasional once-off penalties to improve
        // your search times, rather than having every write always pay that time penalty.

        unimplemented!();
    }
}

impl<K: Clone + Ord + Debug, V: Clone> CursorRead<K, V> {
    pub(crate) fn new(root: ABNode<K, V>, length: usize) -> Self {
        CursorRead {
            root: root,
            length: length,
        }
    }
}

impl<K: Clone + Ord + Debug, V: Clone> CursorReadOps<K, V> for CursorRead<K, V> {
    fn get_root_ref(&self) -> &ABNode<K, V> {
        &self.root
    }

    fn len(&self) -> usize {
        self.length
    }
}

impl<K: Clone + Ord + Debug, V: Clone> CursorReadOps<K, V> for CursorWrite<K, V> {
    fn get_root_ref(&self) -> &ABNode<K, V> {
        &self.root
    }

    fn len(&self) -> usize {
        self.length
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
            CRInsertState::Clone(res, lnode) => {
                // Do we require cloning?
                if txid == node_txid {
                    // This is reached by a sibling leaf already cloning, then a second leaf
                    // updates.
                    nmref.replace_by_idx(anode_idx, lnode);
                    CRInsertState::NoClone(res)
                } else {
                    let mut cnode = node.req_clone(txid);
                    let nmref = Arc::get_mut(&mut cnode).unwrap().as_mut_branch();
                    nmref.replace_by_idx(anode_idx, lnode);
                    // We have be cloned and done the needed replace of the child, so now we
                    // pass our cloned state back up.
                    CRInsertState::Clone(res, cnode)
                }
            }
            CRInsertState::NoClone(res) => {
                // If our descendant did not clone, then we don't have to either.
                debug_assert!(txid == node.txid);
                CRInsertState::NoClone(res)
            }
            CRInsertState::Split(rnode) => {
                // Our child has split, but was already cloned in this txn. Now we need to decide
                // what is needed.
                if txid == node_txid {
                    match nmref.add_node(rnode) {
                        // Similar to CloneSplit - we are either okay, and the insert was happy.
                        BRInsertState::Ok => CRInsertState::NoClone(None),
                        // Or *we* split as well, and need to return a new sibling branch.
                        BRInsertState::Split(clnode, crnode) => {
                            // Create a new branch to hold these children.
                            let nrnode = Node::new_branch(txid, clnode, crnode);
                            // Return it
                            CRInsertState::Split(nrnode)
                        }
                    }
                } else {
                    // I think
                    unreachable!("This represents a corrupt tree state");
                }
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
                    match nmref.add_node(rnode) {
                        // Similar to CloneSplit - we are either okay, and the insert was happy.
                        BRInsertState::Ok => CRInsertState::NoClone(None),
                        // Or *we* split as well, and need to return a new sibling branch.
                        BRInsertState::Split(clnode, crnode) => {
                            // Create a new branch to hold these children.
                            let nrnode = Node::new_branch(txid, clnode, crnode);
                            // Return it
                            CRInsertState::Split(nrnode)
                        }
                    }
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
    use super::super::node::{check_drop_count, ABNode, Node};
    use super::{CursorReadOps, CursorWrite};
    use rand::prelude::*;
    use rand::seq::SliceRandom;
    use std::mem;
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
        let mut wcurs = CursorWrite::new(node, 0);
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
        let mut wcurs = CursorWrite::new(node, 0);
        let prev_txid = wcurs.root_txid();

        let r = wcurs.insert(1, 1);
        assert!(r.is_none());
        let r1_txid = wcurs.root_txid();
        assert!(r1_txid == prev_txid + 1);
        assert!(wcurs.verify());
        println!("{:?}", wcurs);
        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_insert_split_2() {
        // Similar to split_1, but test the Split only path. This means
        // leaf needs to be below max to start, and we insert enough in-txn
        // to trigger a clone of leaf AND THEN to cause the split.
        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node, 0);

        for v in 1..(L_CAPACITY + 1) {
            println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        println!("{:?}", wcurs);
        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
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
        let mut wcurs = CursorWrite::new(root, 0);
        assert!(wcurs.verify());
        println!("{:?}", wcurs);

        let r = wcurs.insert(19, 19);
        assert!(r.is_none());
        assert!(wcurs.verify());
        println!("{:?}", wcurs);

        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
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
        let mut wcurs = CursorWrite::new(root, 0);
        assert!(wcurs.verify());

        let r = wcurs.insert(29, 29);
        assert!(r.is_none());
        assert!(wcurs.verify());
        println!("{:?}", wcurs);

        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
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
        let lnode = create_leaf_node(10);
        let rnode = create_leaf_node(20);
        let root = Node::new_branch(0, lnode, rnode);
        let mut wcurs = CursorWrite::new(root, 0);
        assert!(wcurs.verify());

        // Now insert to trigger the needed actions.
        // Remember, we only need L_CAPACITY because there is already a
        // value in the leaf.
        for idx in 0..(L_CAPACITY) {
            let v = 10 + 1 + idx;
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        println!("{:?}", wcurs);

        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
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
        let lnode = create_leaf_node(10);
        let rnode = create_leaf_node(20);
        let root = Node::new_branch(0, lnode, rnode);
        let mut wcurs = CursorWrite::new(root, 0);
        assert!(wcurs.verify());

        // Now insert to trigger the needed actions.
        // Remember, we only need L_CAPACITY because there is already a
        // value in the leaf.
        for idx in 0..(L_CAPACITY) {
            let v = 20 + 1 + idx;
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        println!("{:?}", wcurs);

        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_insert_split_7() {
        //      root
        //     /    \
        //  leaf    split leaf
        // Insert to leaf then split leaf such that root has cloned
        // in step 1, but doesn't need clone in 2.
        let lnode = create_leaf_node(10);
        let rnode = create_leaf_node(20);
        let root = Node::new_branch(0, lnode, rnode);
        let mut wcurs = CursorWrite::new(root, 0);
        assert!(wcurs.verify());

        let r = wcurs.insert(11, 11);
        assert!(r.is_none());
        assert!(wcurs.verify());

        let r = wcurs.insert(21, 21);
        assert!(r.is_none());
        assert!(wcurs.verify());

        println!("{:?}", wcurs);

        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_insert_split_8() {
        //      root
        //     /    \
        //  leaf    split leaf
        //        ^               ^
        //        \---- nnode 1    \----- nnode 2
        //
        //  Check double leaf split of sl (same txn). This is to
        // take the clonesplit path in the branch case where branch already
        // cloned.
        //
        let lnode = create_leaf_node_full(10);
        let rnode = create_leaf_node_full(20);
        let root = Node::new_branch(0, lnode, rnode);
        let mut wcurs = CursorWrite::new(root, 0);
        assert!(wcurs.verify());

        let r = wcurs.insert(19, 19);
        assert!(r.is_none());
        assert!(wcurs.verify());

        let r = wcurs.insert(29, 29);
        assert!(r.is_none());
        assert!(wcurs.verify());

        println!("{:?}", wcurs);

        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_insert_stress_1() {
        // Insert ascending - we want to ensure the tree is a few levels deep
        // so we do this to a reasonable number.
        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node, 0);

        for v in 1..(L_CAPACITY << 4) {
            println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        println!("{:?}", wcurs);
        println!("DENSITY -> {:?}", wcurs.get_tree_density());
        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_insert_stress_2() {
        // Insert descending
        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node, 0);

        for v in (1..(L_CAPACITY << 4)).rev() {
            println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        println!("{:?}", wcurs);
        println!("DENSITY -> {:?}", wcurs.get_tree_density());
        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_insert_stress_3() {
        // Insert random
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (1..(L_CAPACITY << 4)).collect();
        ins.shuffle(&mut rng);

        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node, 0);

        for v in ins.into_iter() {
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        println!("{:?}", wcurs);
        println!("DENSITY -> {:?}", wcurs.get_tree_density());
        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
    }

    // Add transaction-ised versions.
    #[test]
    fn test_bptree_cursor_insert_stress_4() {
        // Insert ascending - we want to ensure the tree is a few levels deep
        // so we do this to a reasonable number.
        let mut node = create_leaf_node(0);

        for v in 1..(L_CAPACITY << 4) {
            let mut wcurs = CursorWrite::new(node, 0);
            println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
            let (n, _) = wcurs.finalise();
            node = n;
        }
        println!("{:?}", node);
        // On shutdown, check we dropped all as needed.
        mem::drop(node);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_insert_stress_5() {
        // Insert descending
        let mut node = create_leaf_node(0);

        for v in (1..(L_CAPACITY << 4)).rev() {
            let mut wcurs = CursorWrite::new(node, 0);
            println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
            let (n, _) = wcurs.finalise();
            node = n;
        }
        println!("{:?}", node);
        // On shutdown, check we dropped all as needed.
        mem::drop(node);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_insert_stress_6() {
        // Insert random
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (1..(L_CAPACITY << 4)).collect();
        ins.shuffle(&mut rng);

        let mut node = create_leaf_node(0);

        for v in ins.into_iter() {
            let mut wcurs = CursorWrite::new(node, 0);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
            let (n, _) = wcurs.finalise();
            node = n;
        }
        println!("{:?}", node);
        // On shutdown, check we dropped all as needed.
        mem::drop(node);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_search_1() {
        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node, 0);

        for v in 1..(L_CAPACITY << 4) {
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            let r = wcurs.search(&v);
            assert!(r.unwrap() == &v);
        }

        for v in 1..(L_CAPACITY << 4) {
            let r = wcurs.search(&v);
            assert!(r.unwrap() == &v);
        }
        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_length_1() {
        // Check the length is consistent on operations.
        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node, 1);

        for v in 1..(L_CAPACITY << 4) {
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
        }
        println!("{} == {}", wcurs.len(), L_CAPACITY << 4);
        assert!(wcurs.len() == L_CAPACITY << 4);
    }
}
