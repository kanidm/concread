// The cursor is what actually knits a tree together from the parts
// we have, and has an important role to keep the system consistent.
//
// Additionally, the cursor also is responsible for general movement
// throughout the structure and how to handle that effectively

use super::node::ABNode;
use std::collections::LinkedList;
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;

use super::leaf::Leaf;
// use super::branch::Branch;
use super::states::{BLInsertState, BNClone};

pub(crate) struct CursorRead<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    root: ABNode<K, V>,
}

pub(crate) struct CursorWrite<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    // Need to build a stack as we go - of what, I'm not sure ...
    txid: usize,
    root: ABNode<K, V>,
    stack: LinkedList<()>,
}

impl<K: Clone + Ord + Debug, V: Clone> CursorWrite<K, V> {
    pub(crate) fn new(root: ABNode<K, V>) -> Self {
        let txid = root.txid + 1;
        // TODO: Check that txid < usize max.
        assert!(txid < usize::max_value());

        CursorWrite {
            txid: txid,
            root: root,
            stack: LinkedList::new(),
        }
    }

    pub(crate) fn finalise(self) -> ABNode<K, V> {
        // Return the new root.
        self.root
    }

    fn clone_and_position(&mut self) -> *mut Leaf<K, V> {
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
         * To achieve this, there are some possible states we have to account.
         * - the root is the leaf
         * - the root is a branch with arbitrary branch nodes until leaf
         *
         * We must, effectively handle these situtations.
         *
         * As we are having to clone the path, this means we need access to
         * the previous node *and* the next node. Why? Because if we need
         * to clone the next node, we must update the value pointer to our
         * node in the previous node to keep the cloned path consistent.
         *
         * This is why we need to consider the root is leaf situation - there
         * is no previous node!
         */

        // If this isn't 0, we didn't unwind fully before!
        assert!(self.stack.len() == 0);
        /*
         * [🔥🔥🔥]
         * To start this process, we begin by cloning the root if required.
         */
        match self.root.req_clone(self.txid) {
            BNClone::Ok => {}
            BNClone::Clone(broot) => {
                let mut abroot = Arc::new(broot);
                mem::swap(&mut self.root, &mut abroot);
                println!("Swapped root in cursor!");
            }
        };

        /*
         * [🔥🔥🔥]
         * If the root is a leaf, then we return at this point. This is
         * because we are now positioned to update the leaf, and we have
         * no path to unwind (the empty list). The leaf is certainly from
         * this transaction.
         */
        debug_assert!(self.root.txid == self.txid);
        if self.root.is_leaf() {
            // Turn it into a mut inner. This involves arc shenangians.
            /*
            let mref = if cfg!(test) {
                Arc::get_mut(&mut self.root).unwrap()
            } else {
                Arc::get_mut_unchecked(&mut self.root)
            };
            */
            let mref = Arc::get_mut(&mut self.root).unwrap();
            // We know this will live long enough.
            return mref.as_mut_leaf() as *mut Leaf<K, V>;
        }

        /*
         * [🔥🔥🔥]
         * If the root is a branch, then we must now begin to clone on the path.
         * We have to search to find where the next node should be. We store the
         * index of the node, so we have:
         *
         *  prev (cloned)
         *   | .. some node idx
         *   v
         *  work_node
         *
         * The reason to retain the node idx of prev, is so that if work_node is
         * cloned, then we can look back to prev and replace the previous arc
         * with the new node. We'll also need to stash the idx and the prev as
         * a tuple for unwinding.
         *
         *               prev (cloned)
         *                | .. some node idx
         *                v
         *  work_node   new_node (cloned)
         *    ^
         *    \---- this will be referenced by the prev from a former transaction
         *          so won't be dropped yet.
         */

        // Now that we have the root setup, we can do:
        let mut prev_node = &self.root;
        unimplemented!();
        /*
        while !work_node.is_leaf() {
            unimplemented!();
        }
        */
        // Work_node must contain a leaf.
        // Prep it
    }

    // Functions as insert_or_update
    pub(crate) fn insert(&mut self, k: K, v: V) -> Option<V> {
        // Walk down the tree to the k location, cloning on the path.
        let leaf = self.clone_and_position();
        // Now insert and unwind.
        let mut r = unsafe { (&mut *leaf).insert_or_update(k, v) };
        /*
         * Now begin to unwind, this is entirely dictated by r.
         * Because r is from leaf, it's going to set the course.
         */
        let mut nnode = match r {
            BLInsertState::Ok(ir) => {
                /*
                 * If this is Ok -> Simply clear the ll and return. Done.
                 */
                self.stack.clear();
                return ir;
            }
            BLInsertState::Split(k, v) => {
                /*
                 *
                 * If this is split - we may need to split and walk back all the way
                 * back up the tree, this is why we stored all those pointers and
                 * indexes!
                 */
                // Make a new node, and return
                unimplemented!();
            }
        };

        // Now, while we keep getting BRInsertState::Split, keep
        // walking back up and inserting as needed.
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

#[cfg(test)]
mod tests {
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

}
