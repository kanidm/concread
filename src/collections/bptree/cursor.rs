// The cursor is what actually knits a tree together from the parts
// we have, and has an important role to keep the system consistent.
//
// Additionally, the cursor also is responsible for general movement
// throughout the structure and how to handle that effectively

use super::node::{ABNode, Node};
use std::borrow::Borrow;
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;

// use super::branch::Branch;
use super::iter::{Iter, KeyIter, ValueIter};
use super::states::{
    BLInsertState, BLPruneState, BLRemoveState, BRInsertState, BRShrinkState, BRTrimState,
    CRInsertState, CRRemoveState, CRTrimState,
};
use super::states::{CRCloneState, CRPruneState};
use std::iter::Extend;

#[derive(Debug, Clone)]
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

    fn search<'a, 'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        // Search for and return if a value exists at key.
        let rref = self.get_root_ref();
        rref.get_ref(k)
            // You know, I don't even want to talk about the poor life decisions
            // that lead to this code existing.
            .map(|v| unsafe {
                let x = v as *const V;
                &*x as &V
            })
    }

    fn contains_key<'a, 'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        match self.search(k) {
            Some(_) => true,
            None => false,
        }
    }

    fn kv_iter(&self) -> Iter<K, V> {
        Iter::new(self.get_root_ref(), self.len())
    }

    fn k_iter(&self) -> KeyIter<K, V> {
        KeyIter::new(self.get_root_ref(), self.len())
    }

    fn v_iter(&self) -> ValueIter<K, V> {
        ValueIter::new(self.get_root_ref(), self.len())
    }

    fn verify(&self) -> bool {
        let (l, _) = self.get_tree_density();
        assert!(l == self.len());
        self.get_root_ref().verify()
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

    pub(crate) fn remove(&mut self, k: &K) -> Option<V> {
        let r = match clone_and_remove(&mut self.root, self.txid, k) {
            CRRemoveState::NoClone(res) => res,
            CRRemoveState::Clone(res, mut nnode) => {
                mem::swap(&mut self.root, &mut nnode);
                res
            }
            CRRemoveState::Shrink(res) => {
                if self.root.is_leaf() {
                    // No action - we have an empty tree.
                    res
                } else {
                    // Root is being demoted, get the last branch and
                    // promote it to the root.
                    let rmut = Arc::get_mut(&mut self.root).unwrap().as_mut_branch();
                    let mut pnode = rmut.extract_last_node();
                    mem::swap(&mut self.root, &mut pnode);
                    res
                }
            }
            CRRemoveState::CloneShrink(res, mut nnode) => {
                if nnode.is_leaf() {
                    // The tree is empty, but we cloned the root to get here.
                    mem::swap(&mut self.root, &mut nnode);
                    res
                } else {
                    // Our root is getting demoted here, get the remaining branch
                    let rmut = Arc::get_mut(&mut nnode).unwrap().as_mut_branch();
                    let mut pnode = rmut.extract_last_node();
                    // Promote it to the new root
                    mem::swap(&mut self.root, &mut pnode);
                    res
                }
            }
        };
        if r.is_some() {
            self.length -= 1;
        }
        r
    }

    #[cfg(test)]
    pub(crate) fn path_clone(&mut self, k: &K) {
        match path_clone(&mut self.root, self.txid, k) {
            CRCloneState::Clone(mut nroot) => {
                // We cloned the root, so swap it.
                mem::swap(&mut self.root, &mut nroot);
            }
            CRCloneState::NoClone => {}
        };
    }

    pub(crate) fn get_mut_ref(&mut self, k: &K) -> Option<&mut V> {
        match path_clone(&mut self.root, self.txid, k) {
            CRCloneState::Clone(mut nroot) => {
                // We cloned the root, so swap it.
                mem::swap(&mut self.root, &mut nroot);
            }
            CRCloneState::NoClone => {}
        };
        // Now get the ref.
        path_get_mut_ref(&mut self.root, k)
    }

    pub(crate) fn split_off_lt(&mut self, k: &K) {
        // Remove all the values less than from the top of the tree.
        loop {
            let result = clone_and_split_off_trim_lt(&mut self.root, self.txid, k);
            // println!("clone_and_split_off_trim_lt -> {:?}", result);
            match result {
                CRTrimState::Complete => break,
                CRTrimState::Clone(mut nroot) => {
                    // We cloned the root as we changed it, but don't need
                    // to recurse.
                    mem::swap(&mut self.root, &mut nroot);
                    break;
                }
                CRTrimState::Promote(mut nroot) => {
                    mem::swap(&mut self.root, &mut nroot);
                    // This will continue and try again.
                }
            }
        }

        // Now work up the tree and clean up the remaining path inbetween
        let result = clone_and_split_off_prune_lt(&mut self.root, self.txid, k);
        // println!("clone_and_split_off_prune_lt -> {:?}", result);
        match result {
            CRPruneState::OkNoClone => {}
            CRPruneState::OkClone(mut nroot) => {
                mem::swap(&mut self.root, &mut nroot);
            }
            CRPruneState::Prune => {
                if self.root.is_leaf() {
                    // No action, the tree is now empty.
                } else {
                    // Root is being demoted, get the last branch and
                    // promote it to the root.
                    let rmut = Arc::get_mut(&mut self.root).unwrap().as_mut_branch();
                    let mut pnode = rmut.extract_last_node();
                    mem::swap(&mut self.root, &mut pnode);
                }
            }
            CRPruneState::ClonePrune(mut clone) => {
                if self.root.is_leaf() {
                    mem::swap(&mut self.root, &mut clone);
                } else {
                    let rmut = Arc::get_mut(&mut clone).unwrap().as_mut_branch();
                    let mut pnode = rmut.extract_last_node();
                    mem::swap(&mut self.root, &mut pnode);
                }
            }
        };

        // Iterate over the remaining kv's to fix our k,v count.
        let newsize = self.kv_iter().count();
        self.length = newsize;
    }

    #[cfg(test)]
    pub(crate) fn root_txid(&self) -> usize {
        self.root.txid
    }

    pub(crate) fn tree_density(&self) -> (usize, usize) {
        self.root.tree_density()
    }
}

impl<K: Clone + Ord + Debug, V: Clone> Extend<(K, V)> for CursorWrite<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        iter.into_iter().for_each(|(k, v)| {
            let _ = self.insert(k, v);
        });
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
            match mref.as_mut_leaf().insert_or_update(txid, k, v) {
                BLInsertState::Ok(res) => CRInsertState::NoClone(res),
                BLInsertState::Split(rnode) => {
                    // We split, but left is already part of the txn group, so lets
                    // just return what's new.
                    // let rnode = Node::new_leaf_ins(txid, sk, sv);
                    CRInsertState::Split(rnode)
                }
            }
        } else {
            // Clone required.
            let mut cnode = node.req_clone(txid);
            let mref = Arc::get_mut(&mut cnode).unwrap();
            // insert to the new node.
            match mref.as_mut_leaf().insert_or_update(txid, k, v) {
                BLInsertState::Ok(res) => CRInsertState::Clone(res, cnode),
                BLInsertState::Split(rnode) => {
                    // let rnode = Node::new_leaf_ins(txid, sk, sv);
                    CRInsertState::CloneSplit(cnode, rnode)
                }
            }
        }
    } else {
        // Branch path
        // Decide if we need to clone - we do this as we descend due to a quirk in Arc
        // get_mut, because we don't have access to get_mut_unchecked (and this api may
        // never be stabilised anyway). When we change this to *mut + garbage lists we
        // could consider restoring the reactive behaviour that clones up, rather than
        // cloning down the path.

        if node.txid == txid {
            let nmref = Arc::get_mut(node).unwrap().as_mut_branch();
            let anode_idx = nmref.locate_node(&k);
            let mut anode = nmref.get_mut_idx(anode_idx);

            match clone_and_insert(&mut anode, txid, k, v) {
                CRInsertState::Clone(res, lnode) => {
                    nmref.replace_by_idx(anode_idx, lnode);
                    // We did not clone, and no further work needed.
                    CRInsertState::NoClone(res)
                }
                CRInsertState::NoClone(res) => {
                    // If our descendant did not clone, then we don't have to do any adjustments
                    // or further work.
                    debug_assert!(txid == node.txid);
                    CRInsertState::NoClone(res)
                }
                CRInsertState::Split(rnode) => {
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
                }
                CRInsertState::CloneSplit(lnode, rnode) => {
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
                }
            }
        } else {
            // Not same txn, clone instead.
            let mut cnode = node.req_clone(txid);
            let nmref = Arc::get_mut(&mut cnode).unwrap().as_mut_branch();
            let anode_idx = nmref.locate_node(&k);
            let mut anode = nmref.get_mut_idx(anode_idx);

            match clone_and_insert(&mut anode, txid, k, v) {
                CRInsertState::Clone(res, lnode) => {
                    nmref.replace_by_idx(anode_idx, lnode);
                    // Pass back up that we cloned.
                    CRInsertState::Clone(res, cnode)
                }
                CRInsertState::NoClone(_res) => {
                    // If our descendant did not clone, then we don't have to either.
                    debug_assert!(txid == node.txid);
                    unreachable!("Shoud never be possible.");
                    // CRInsertState::NoClone(res)
                }
                CRInsertState::Split(_rnode) => {
                    // I think
                    unreachable!("This represents a corrupt tree state");
                }
                CRInsertState::CloneSplit(lnode, rnode) => {
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
            } // end match
        } // end if node txn
    } // end if leaf
}

fn path_clone<'a, K: Clone + Ord + Debug, V: Clone>(
    node: &'a mut ABNode<K, V>,
    txid: usize,
    k: &K,
) -> CRCloneState<K, V> {
    if node.is_leaf() {
        if txid == node.txid {
            // No clone, just return.
            CRCloneState::NoClone
        } else {
            let cnode = node.req_clone(txid);
            CRCloneState::Clone(cnode)
        }
    } else {
        // We are in a branch, so locate our descendent and prepare
        // to clone if needed.
        let node_txid = node.txid;
        let nmref = Arc::get_mut(node).unwrap().as_mut_branch();
        let anode_idx = nmref.locate_node(&k);
        let mut anode = nmref.get_mut_idx(anode_idx);
        match path_clone(&mut anode, txid, k) {
            CRCloneState::Clone(cnode) => {
                // Do we need to clone?
                if txid == node_txid {
                    // Nope, just insert and unwind.
                    nmref.replace_by_idx(anode_idx, cnode);
                    CRCloneState::NoClone
                } else {
                    // We require to be cloned.
                    let mut acnode = node.req_clone(txid);
                    let nmref = Arc::get_mut(&mut acnode).unwrap().as_mut_branch();
                    nmref.replace_by_idx(anode_idx, cnode);
                    CRCloneState::Clone(acnode)
                }
            }
            CRCloneState::NoClone => {
                // Did not clone, move on.
                CRCloneState::NoClone
            }
        }
    }
}

fn clone_and_remove<'a, K: Clone + Ord + Debug, V: Clone>(
    node: &'a mut ABNode<K, V>,
    txid: usize,
    k: &K,
) -> CRRemoveState<K, V> {
    if node.is_leaf() {
        if node.txid == txid {
            let mref = Arc::get_mut(node).unwrap();
            match mref.as_mut_leaf().remove(k) {
                BLRemoveState::Ok(res) => CRRemoveState::NoClone(res),
                BLRemoveState::Shrink(res) => CRRemoveState::Shrink(res),
            }
        } else {
            let mut cnode = node.req_clone(txid);
            let mref = Arc::get_mut(&mut cnode).unwrap();
            match mref.as_mut_leaf().remove(k) {
                BLRemoveState::Ok(res) => CRRemoveState::Clone(res, cnode),
                BLRemoveState::Shrink(res) => CRRemoveState::CloneShrink(res, cnode),
            }
        }
    } else {
        // Locate the node we need to work on and then react if it
        // requests a shrink.

        if node.txid == txid {
            let nmref = Arc::get_mut(node).unwrap().as_mut_branch();
            let anode_idx = nmref.locate_node(&k);
            let mut anode = nmref.get_mut_idx(anode_idx);
            match clone_and_remove(&mut anode, txid, k) {
                CRRemoveState::NoClone(res) => {
                    debug_assert!(txid == node.txid);
                    CRRemoveState::NoClone(res)
                }
                CRRemoveState::Clone(res, lnode) => {
                    nmref.replace_by_idx(anode_idx, lnode);
                    CRRemoveState::NoClone(res)
                }
                CRRemoveState::Shrink(res) => {
                    let right_idx = nmref.clone_sibling_idx(txid, anode_idx);
                    match nmref.shrink_decision(right_idx) {
                        BRShrinkState::Balanced | BRShrinkState::Merge => {
                            // K:V were distributed through left and right,
                            // so no further action needed.
                            // -- OR --
                            // Right was merged to left, and we remain
                            // valid
                            CRRemoveState::NoClone(res)
                        }
                        BRShrinkState::Shrink => {
                            // Right was merged to left, but we have now falled under the needed
                            // amount of values, so we begin to shrink up.
                            CRRemoveState::Shrink(res)
                        }
                    }
                }
                CRRemoveState::CloneShrink(res, nnode) => {
                    // We don't need to clone, just work on the nmref we have.
                    //
                    // Swap in the cloned node to the correct location.
                    nmref.replace_by_idx(anode_idx, nnode);
                    // Now setup the sibling, to the left *or* right.
                    let right_idx = nmref.clone_sibling_idx(txid, anode_idx);
                    match nmref.shrink_decision(right_idx) {
                        BRShrinkState::Balanced | BRShrinkState::Merge => {
                            // K:V were distributed through left and right,
                            // so no further action needed.
                            // -- OR --
                            // Right was merged to left, and we remain
                            // valid
                            CRRemoveState::NoClone(res)
                        }
                        BRShrinkState::Shrink => {
                            // Right was merged to left, but we have now falled under the needed
                            // amount of values.
                            CRRemoveState::Shrink(res)
                        }
                    }
                }
            }
        } else {
            let mut cnode = node.req_clone(txid);
            let nmref = Arc::get_mut(&mut cnode).unwrap().as_mut_branch();
            let anode_idx = nmref.locate_node(&k);
            let mut anode = nmref.get_mut_idx(anode_idx);
            match clone_and_remove(&mut anode, txid, k) {
                CRRemoveState::NoClone(_res) => {
                    debug_assert!(txid == node.txid);
                    unreachable!("Should never occur");
                    // CRRemoveState::NoClone(res)
                }
                CRRemoveState::Clone(res, lnode) => {
                    nmref.replace_by_idx(anode_idx, lnode);
                    CRRemoveState::Clone(res, cnode)
                }
                CRRemoveState::Shrink(_res) => {
                    unreachable!("This represents a corrupt tree state");
                }
                CRRemoveState::CloneShrink(res, nnode) => {
                    // Put our cloned child into the tree at the correct location, don't worry,
                    // the shrink_decision will deal with it.
                    nmref.replace_by_idx(anode_idx, nnode);

                    // Now setup the sibling, to the left *or* right.
                    let right_idx = nmref.clone_sibling_idx(txid, anode_idx);

                    // Okay, now work out what we need to do.
                    match nmref.shrink_decision(right_idx) {
                        BRShrinkState::Balanced | BRShrinkState::Merge => {
                            // K:V were distributed through left and right,
                            // so no further action needed.
                            // -- OR --
                            // Right was merged to left, and we remain
                            // valid
                            CRRemoveState::Clone(res, cnode)
                        }
                        BRShrinkState::Shrink => {
                            // Right was merged to left, but we have now falled under the needed
                            // amount of values.
                            CRRemoveState::CloneShrink(res, cnode)
                        }
                    }
                }
            }
        } // end node.txid
    }
}

fn path_get_mut_ref<'a, K: Clone + Ord + Debug, V: Clone>(
    node: &'a mut ABNode<K, V>,
    k: &K,
) -> Option<&'a mut V> {
    if node.is_leaf() {
        Arc::get_mut(node).unwrap().as_mut_leaf().get_mut_ref(k)
    } else {
        // This nmref binds the life of thte reference ...
        let nmref = Arc::get_mut(node).unwrap().as_mut_branch();
        let anode_idx = nmref.locate_node(&k);
        let mut anode = nmref.get_mut_idx(anode_idx);
        // That we get here. So we can't just return it, and we need to 'strip' the
        // lifetime so that it's bound to the lifetime of the outer node
        // rather than the nmref.
        let r: Option<*mut V> = path_get_mut_ref(&mut anode, k).map(|v| v as *mut V);

        // I solemly swear I am up to no good.
        r.map(|v| unsafe { &mut *v as &mut V })
    }
}

fn clone_and_split_off_trim_lt<'a, K: Clone + Ord + Debug, V: Clone>(
    node: &'a mut ABNode<K, V>,
    txid: usize,
    k: &K,
) -> CRTrimState<K, V> {
    if node.is_leaf() {
        // No action, it's a leaf. Prune will do it.
        CRTrimState::Complete
    } else {
        // remove_lt_idx
        if node.txid == txid {
            let nmref = Arc::get_mut(node).unwrap().as_mut_branch();

            match nmref.trim_lt_key(k) {
                BRTrimState::Complete => CRTrimState::Complete,
                BRTrimState::Promote(node) => CRTrimState::Promote(node),
            }
        } else {
            let mut cnode = node.req_clone(txid);
            let nmref = Arc::get_mut(&mut cnode).unwrap().as_mut_branch();

            match nmref.trim_lt_key(k) {
                BRTrimState::Complete => CRTrimState::Clone(cnode),
                BRTrimState::Promote(nnode) => CRTrimState::Promote(nnode),
            }
        }
    }
}

fn clone_and_split_off_prune_lt<'a, K: Clone + Ord + Debug, V: Clone>(
    node: &'a mut ABNode<K, V>,
    txid: usize,
    k: &K,
) -> CRPruneState<K, V> {
    if node.is_leaf() {
        // I think this should be do nothing, the up walk will clean.
        if node.txid == txid {
            let nmref = Arc::get_mut(node).unwrap().as_mut_leaf();
            match nmref.remove_lt(k) {
                BLPruneState::Ok => CRPruneState::OkNoClone,
                BLPruneState::Prune => CRPruneState::Prune,
            }
        } else {
            let mut cnode = node.req_clone(txid);
            let nmref = Arc::get_mut(&mut cnode).unwrap().as_mut_leaf();
            match nmref.remove_lt(k) {
                BLPruneState::Ok => CRPruneState::OkClone(cnode),
                BLPruneState::Prune => CRPruneState::ClonePrune(cnode),
            }
        }
    } else {
        if node.txid == txid {
            let nmref = Arc::get_mut(node).unwrap().as_mut_branch();
            let anode_idx = nmref.locate_node(&k);
            let anode = nmref.get_mut_idx(anode_idx);
            let result = clone_and_split_off_prune_lt(anode, txid, k);
            // println!("== clone_and_split_off_prune_lt --> {:?}", result);
            match result {
                CRPruneState::OkNoClone => {
                    match nmref.prune(anode_idx) {
                        Ok(_) => {
                            // Okay, the branch remains valid, return that we are okay, and
                            // no clone is needed.
                            CRPruneState::OkNoClone
                        }
                        Err(_) => CRPruneState::Prune,
                    }
                }
                CRPruneState::OkClone(clone) => {
                    // Our child cloned, so replace it.
                    nmref.replace_by_idx(anode_idx, clone);
                    // Check our node for anything else to be removed.
                    match nmref.prune(anode_idx) {
                        Ok(_) => {
                            // Okay, the branch remains valid, return that we are okay, and
                            // no clone is needed.
                            CRPruneState::OkNoClone
                        }
                        Err(_) => CRPruneState::Prune,
                    }
                }
                CRPruneState::Prune => {
                    match nmref.prune_decision(txid, anode_idx) {
                        Ok(_) => {
                            // Okay, the branch remains valid. Now we need to trim any
                            // excess if possible.
                            CRPruneState::OkNoClone
                        }
                        Err(_) => CRPruneState::Prune,
                    }
                }
                CRPruneState::ClonePrune(clone) => {
                    // Our child cloned, and intends to be removed.
                    nmref.replace_by_idx(anode_idx, clone);
                    // Now make the prune decision.
                    match nmref.prune_decision(txid, anode_idx) {
                        Ok(_) => {
                            // Okay, the branch remains valid. Now we need to trim any
                            // excess if possible.
                            CRPruneState::OkNoClone
                        }
                        Err(_) => CRPruneState::Prune,
                    }
                }
            }
        } else {
            let mut cnode = node.req_clone(txid);
            let nmref = Arc::get_mut(&mut cnode).unwrap().as_mut_branch();
            let anode_idx = nmref.locate_node(&k);
            let anode = nmref.get_mut_idx(anode_idx);
            let result = clone_and_split_off_prune_lt(anode, txid, k);
            // println!("!= clone_and_split_off_prune_lt --> {:?}", result);
            match result {
                CRPruneState::OkNoClone => {
                    // I think this is an impossible state - how can a child be in the
                    // txid if we are not?
                    unreachable!("Impossible tree state")
                }
                CRPruneState::OkClone(clone) => {
                    // Our child cloned, so replace it.
                    nmref.replace_by_idx(anode_idx, clone);
                    // Check our node for anything else to be removed.
                    match nmref.prune(anode_idx) {
                        Ok(_) => {
                            // Okay, the branch remains valid, return that we are okay.
                            CRPruneState::OkClone(cnode)
                        }
                        Err(_) => CRPruneState::ClonePrune(cnode),
                    }
                }
                CRPruneState::Prune => {
                    unimplemented!();
                }
                CRPruneState::ClonePrune(clone) => {
                    // Our child cloned, and intends to be removed.
                    nmref.replace_by_idx(anode_idx, clone);
                    // Now make the prune decision.
                    match nmref.prune_decision(txid, anode_idx) {
                        Ok(_) => {
                            // Okay, the branch remains valid. Now we need to trim any
                            // excess if possible.
                            CRPruneState::OkClone(cnode)
                        }
                        Err(_) => CRPruneState::ClonePrune(cnode),
                    }
                } // end clone prune
            } // end match result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::constants::{BK_CAPACITY, BV_CAPACITY, CAPACITY, L_CAPACITY};
    use super::super::node::{check_drop_count, ABNode, Node};
    use super::super::states::BRInsertState;
    use super::{CursorReadOps, CursorWrite};
    // use rand::prelude::*;
    use rand::seq::SliceRandom;
    use std::mem;
    use std::sync::Arc;

    fn create_leaf_node(v: usize) -> ABNode<usize, usize> {
        let mut node = Arc::new(Node::new_leaf(0));
        {
            let nmut = Arc::get_mut(&mut node).unwrap().as_mut_leaf();
            nmut.insert_or_update(0, v, v);
        }
        node
    }

    fn create_leaf_node_full(vbase: usize) -> ABNode<usize, usize> {
        assert!(vbase % 10 == 0);
        let mut node = Arc::new(Node::new_leaf(0));
        {
            let nmut = Arc::get_mut(&mut node).unwrap().as_mut_leaf();
            for idx in 0..L_CAPACITY {
                let v = vbase + idx;
                nmut.insert_or_update(0, v, v);
            }
        }
        node
    }

    fn create_branch_node_full(vbase: usize) -> ABNode<usize, usize> {
        let l1 = create_leaf_node(vbase);
        let l2 = create_leaf_node(vbase + 10);
        let mut lbranch = Node::new_branch(0, l1, l2);
        let bref = Arc::get_mut(&mut lbranch).unwrap().as_mut_branch();
        for i in 2..BV_CAPACITY {
            let l = create_leaf_node(vbase + (10 * i));
            let r = bref.add_node(l);
            match r {
                BRInsertState::Ok => {}
                _ => panic!(),
            }
        }
        assert!(bref.len() == BK_CAPACITY);
        lbranch
    }

    #[test]
    fn test_bptree_cursor_insert_leaf() {
        // First create the node + cursor
        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node, 1);
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
        let mut wcurs = CursorWrite::new(node, L_CAPACITY);
        let prev_txid = wcurs.root_txid();

        let r = wcurs.insert(1, 1);
        assert!(r.is_none());
        let r1_txid = wcurs.root_txid();
        assert!(r1_txid == prev_txid + 1);
        assert!(wcurs.verify());
        // println!("{:?}", wcurs);
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
        let mut wcurs = CursorWrite::new(node, 1);

        for v in 1..(L_CAPACITY + 1) {
            // println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        // println!("{:?}", wcurs);
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
        let mut wcurs = CursorWrite::new(root, L_CAPACITY * 2);
        assert!(wcurs.verify());
        // println!("{:?}", wcurs);

        let r = wcurs.insert(19, 19);
        assert!(r.is_none());
        assert!(wcurs.verify());
        // println!("{:?}", wcurs);

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
        let mut wcurs = CursorWrite::new(root, L_CAPACITY * 2);
        assert!(wcurs.verify());

        let r = wcurs.insert(29, 29);
        assert!(r.is_none());
        assert!(wcurs.verify());
        // println!("{:?}", wcurs);

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
        let mut wcurs = CursorWrite::new(root, 2);
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
        // println!("{:?}", wcurs);

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
        let mut wcurs = CursorWrite::new(root, 2);
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
        // println!("{:?}", wcurs);

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
        let mut wcurs = CursorWrite::new(root, 2);
        assert!(wcurs.verify());

        let r = wcurs.insert(11, 11);
        assert!(r.is_none());
        assert!(wcurs.verify());

        let r = wcurs.insert(21, 21);
        assert!(r.is_none());
        assert!(wcurs.verify());

        // println!("{:?}", wcurs);

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
        let mut wcurs = CursorWrite::new(root, L_CAPACITY * 2);
        assert!(wcurs.verify());

        let r = wcurs.insert(19, 19);
        assert!(r.is_none());
        assert!(wcurs.verify());

        let r = wcurs.insert(29, 29);
        assert!(r.is_none());
        assert!(wcurs.verify());

        // println!("{:?}", wcurs);

        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_insert_stress_1() {
        // Insert ascending - we want to ensure the tree is a few levels deep
        // so we do this to a reasonable number.
        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node, 1);

        for v in 1..(L_CAPACITY << 4) {
            // println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        // println!("{:?}", wcurs);
        // println!("DENSITY -> {:?}", wcurs.get_tree_density());
        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_insert_stress_2() {
        // Insert descending
        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node, 1);

        for v in (1..(L_CAPACITY << 4)).rev() {
            // println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        // println!("{:?}", wcurs);
        // println!("DENSITY -> {:?}", wcurs.get_tree_density());
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
        let mut wcurs = CursorWrite::new(node, 1);

        for v in ins.into_iter() {
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        // println!("{:?}", wcurs);
        // println!("DENSITY -> {:?}", wcurs.get_tree_density());
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
        let mut count = 1;

        for v in 1..(L_CAPACITY << 4) {
            let mut wcurs = CursorWrite::new(node, count);
            // println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
            let (n, c) = wcurs.finalise();
            node = n;
            count = c;
        }
        // println!("{:?}", node);
        // On shutdown, check we dropped all as needed.
        mem::drop(node);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_insert_stress_5() {
        // Insert descending
        let mut node = create_leaf_node(0);
        let mut count = 1;

        for v in (1..(L_CAPACITY << 4)).rev() {
            let mut wcurs = CursorWrite::new(node, count);
            // println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
            let (n, c) = wcurs.finalise();
            node = n;
            count = c;
        }
        // println!("{:?}", node);
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
        let mut count = 1;

        for v in ins.into_iter() {
            let mut wcurs = CursorWrite::new(node, count);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
            let (n, c) = wcurs.finalise();
            node = n;
            count = c;
        }
        // println!("{:?}", node);
        // On shutdown, check we dropped all as needed.
        mem::drop(node);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_search_1() {
        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node, 1);

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
        // println!("{} == {}", wcurs.len(), L_CAPACITY << 4);
        assert!(wcurs.len() == L_CAPACITY << 4);
    }

    #[test]
    fn test_bptree_cursor_remove_01_p0() {
        // Check that a single value can be removed correctly without change.
        // Check that a missing value is removed as "None".
        // Check that emptying the root is ok.
        // BOTH of these need new txns to check clone, and then re-use txns.
        //
        //
        let lnode = create_leaf_node_full(0);
        let mut wcurs = CursorWrite::new(lnode, L_CAPACITY);
        // println!("{:?}", wcurs);

        for v in 0..L_CAPACITY {
            let x = wcurs.remove(&v);
            // println!("{:?}", wcurs);
            assert!(x == Some(v));
        }

        for v in 0..L_CAPACITY {
            let x = wcurs.remove(&v);
            assert!(x == None);
        }

        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_01_p1() {
        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node, 1);

        let _ = wcurs.remove(&0);
        // println!("{:?}", wcurs);

        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_02() {
        // Given the tree:
        //
        //      root
        //     /    \
        //  leaf    split leaf
        //
        // Remove from "split leaf" and merge left. (new txn)
        let lnode = create_leaf_node(10);
        let rnode = create_leaf_node(20);
        let znode = create_leaf_node(0);
        let mut root = Node::new_branch(0, znode, lnode);
        // Prevent the tree shrinking.
        Arc::get_mut(&mut root)
            .unwrap()
            .as_mut_branch()
            .add_node(rnode);
        let mut wcurs = CursorWrite::new(root, 3);
        // println!("{:?}", wcurs);
        assert!(wcurs.verify());

        wcurs.remove(&20);
        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_03() {
        // Given the tree:
        //
        //      root
        //     /    \
        //  leaf    split leaf
        //
        // Remove from "leaf" and merge right (really left, but you know ...). (new txn)
        let lnode = create_leaf_node(10);
        let rnode = create_leaf_node(20);
        let znode = create_leaf_node(30);
        let mut root = Node::new_branch(0, lnode, rnode);
        // Prevent the tree shrinking.
        Arc::get_mut(&mut root)
            .unwrap()
            .as_mut_branch()
            .add_node(znode);
        let mut wcurs = CursorWrite::new(root, 3);
        assert!(wcurs.verify());

        wcurs.remove(&10);
        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_04p0() {
        // Given the tree:
        //
        //      root
        //     /    \
        //  leaf    split leaf
        //
        // Remove from "split leaf" and merge left. (leaf cloned already)
        let lnode = create_leaf_node(10);
        let rnode = create_leaf_node(20);
        let znode = create_leaf_node(0);
        let mut root = Node::new_branch(0, znode, lnode);
        // Prevent the tree shrinking.
        Arc::get_mut(&mut root)
            .unwrap()
            .as_mut_branch()
            .add_node(rnode);
        let mut wcurs = CursorWrite::new(root, 3);
        assert!(wcurs.verify());

        // Setup sibling leaf to already be cloned.
        wcurs.path_clone(&10);
        assert!(wcurs.verify());

        wcurs.remove(&20);
        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_04p1() {
        // Given the tree:
        //
        //      root
        //     /    \
        //  leaf    split leaf
        //
        // Remove from "split leaf" and merge left. (leaf cloned already)
        let lnode = create_leaf_node(10);
        let rnode = create_leaf_node(20);
        let znode = create_leaf_node(0);
        let mut root = Node::new_branch(0, znode, lnode);
        // Prevent the tree shrinking.
        Arc::get_mut(&mut root)
            .unwrap()
            .as_mut_branch()
            .add_node(rnode);
        let mut wcurs = CursorWrite::new(root, 3);
        assert!(wcurs.verify());

        // Setup leaf to already be cloned.
        wcurs.path_clone(&20);
        assert!(wcurs.verify());

        wcurs.remove(&20);
        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_05() {
        // Given the tree:
        //
        //      root
        //     /    \
        //  leaf    split leaf
        //
        // Remove from "leaf" and merge 'right'. (split leaf cloned already)
        let lnode = create_leaf_node(10);
        let rnode = create_leaf_node(20);
        let znode = create_leaf_node(30);
        let mut root = Node::new_branch(0, lnode, rnode);
        // Prevent the tree shrinking.
        Arc::get_mut(&mut root)
            .unwrap()
            .as_mut_branch()
            .add_node(znode);
        let mut wcurs = CursorWrite::new(root, 3);
        assert!(wcurs.verify());

        // Setup leaf to already be cloned.
        wcurs.path_clone(&20);

        wcurs.remove(&10);
        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_06() {
        // Given the tree:
        //
        //          root
        //        /      \
        //   lbranch     rbranch
        //     /    \     /    \
        //    l1    l2   r1    r2
        //
        //   conditions:
        //   lbranch - 2node
        //   rbranch - 2node
        //   txn     - new
        //
        //   when remove from rbranch, mergc left to lbranch.
        //   should cause tree height reduction.
        let l1 = create_leaf_node(0);
        let l2 = create_leaf_node(10);
        let r1 = create_leaf_node(20);
        let r2 = create_leaf_node(30);
        let lbranch = Node::new_branch(0, l1, l2);
        let rbranch = Node::new_branch(0, r1, r2);
        let root = Node::new_branch(0, lbranch, rbranch);
        let mut wcurs = CursorWrite::new(root, 4);
        assert!(wcurs.verify());

        wcurs.remove(&30);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_07() {
        // Given the tree:
        //
        //          root
        //        /      \
        //   lbranch     rbranch
        //     /    \     /    \
        //    l1    l2   r1    r2
        //
        //   conditions:
        //   lbranch - 2node
        //   rbranch - 2node
        //   txn     - new
        //
        //   when remove from lbranch, merge right to rbranch.
        //   should cause tree height reduction.
        let l1 = create_leaf_node(0);
        let l2 = create_leaf_node(10);
        let r1 = create_leaf_node(20);
        let r2 = create_leaf_node(30);
        let lbranch = Node::new_branch(0, l1, l2);
        let rbranch = Node::new_branch(0, r1, r2);
        let root = Node::new_branch(0, lbranch, rbranch);
        let mut wcurs = CursorWrite::new(root, 4);
        assert!(wcurs.verify());

        wcurs.remove(&10);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_08() {
        // Given the tree:
        //
        //          root
        //        /      \
        //   lbranch     rbranch
        //     /    \     /    \
        //    l1    l2   r1    r2
        //
        //   conditions:
        //   lbranch - full
        //   rbranch - 2node
        //   txn     - new
        //
        //   when remove from rbranch, borrow from lbranch
        //   will NOT reduce height
        let lbranch = create_branch_node_full(0);

        let r1 = create_leaf_node(80);
        let r2 = create_leaf_node(90);
        let rbranch = Node::new_branch(0, r1, r2);

        let root = Node::new_branch(0, lbranch, rbranch);
        let mut wcurs = CursorWrite::new(root, L_CAPACITY + 2);
        assert!(wcurs.verify());

        wcurs.remove(&80);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_09() {
        // Given the tree:
        //
        //          root
        //        /      \
        //   lbranch     rbranch
        //     /    \     /    \
        //    l1    l2   r1    r2
        //
        //   conditions:
        //   lbranch - 2node
        //   rbranch - full
        //   txn     - new
        //
        //   when remove from lbranch, borrow from rbranch
        //   will NOT reduce height
        let l1 = create_leaf_node(0);
        let l2 = create_leaf_node(10);
        let lbranch = Node::new_branch(0, l1, l2);

        let rbranch = create_branch_node_full(100);

        let root = Node::new_branch(0, lbranch, rbranch);
        let mut wcurs = CursorWrite::new(root, L_CAPACITY + 2);
        assert!(wcurs.verify());

        wcurs.remove(&10);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_10() {
        // Given the tree:
        //
        //          root
        //        /      \
        //   lbranch     rbranch
        //     /    \     /    \
        //    l1    l2   r1    r2
        //
        //   conditions:
        //   lbranch - 2node
        //   rbranch - 2node
        //   txn     - touch lbranch
        //
        //   when remove from rbranch, mergc left to lbranch.
        //   should cause tree height reduction.
        let l1 = create_leaf_node(0);
        let l2 = create_leaf_node(10);
        let r1 = create_leaf_node(20);
        let r2 = create_leaf_node(30);
        let lbranch = Node::new_branch(0, l1, l2);
        let rbranch = Node::new_branch(0, r1, r2);
        let root = Node::new_branch(0, lbranch, rbranch);
        let mut wcurs = CursorWrite::new(root, 4);
        assert!(wcurs.verify());

        wcurs.path_clone(&0);
        wcurs.path_clone(&10);

        wcurs.remove(&30);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_11() {
        // Given the tree:
        //
        //          root
        //        /      \
        //   lbranch     rbranch
        //     /    \     /    \
        //    l1    l2   r1    r2
        //
        //   conditions:
        //   lbranch - 2node
        //   rbranch - 2node
        //   txn     - touch rbranch
        //
        //   when remove from lbranch, merge right to rbranch.
        //   should cause tree height reduction.
        let l1 = create_leaf_node(0);
        let l2 = create_leaf_node(10);
        let r1 = create_leaf_node(20);
        let r2 = create_leaf_node(30);
        let lbranch = Node::new_branch(0, l1, l2);
        let rbranch = Node::new_branch(0, r1, r2);
        let root = Node::new_branch(0, lbranch, rbranch);
        let mut wcurs = CursorWrite::new(root, 4);
        assert!(wcurs.verify());

        wcurs.path_clone(&20);
        wcurs.path_clone(&30);

        wcurs.remove(&0);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_12() {
        // Given the tree:
        //
        //          root
        //        /      \
        //   lbranch     rbranch
        //     /    \     /    \
        //    l1    l2   r1    r2
        //
        //   conditions:
        //   lbranch - full
        //   rbranch - 2node
        //   txn     - touch lbranch
        //
        //   when remove from rbranch, borrow from lbranch
        //   will NOT reduce height
        let lbranch = create_branch_node_full(0);

        let r1 = create_leaf_node(80);
        let r2 = create_leaf_node(90);
        let rbranch = Node::new_branch(0, r1, r2);

        let root = Node::new_branch(0, lbranch, rbranch);
        let count = BV_CAPACITY + 2;
        let mut wcurs = CursorWrite::new(root, count);
        assert!(wcurs.verify());

        wcurs.path_clone(&0);
        wcurs.path_clone(&10);
        wcurs.path_clone(&20);

        wcurs.remove(&90);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_13() {
        // Given the tree:
        //
        //          root
        //        /      \
        //   lbranch     rbranch
        //     /    \     /    \
        //    l1    l2   r1    r2
        //
        //   conditions:
        //   lbranch - 2node
        //   rbranch - full
        //   txn     - touch rbranch
        //
        //   when remove from lbranch, borrow from rbranch
        //   will NOT reduce height
        let l1 = create_leaf_node(0);
        let l2 = create_leaf_node(10);
        let lbranch = Node::new_branch(0, l1, l2);

        let rbranch = create_branch_node_full(100);

        let root = Node::new_branch(0, lbranch, rbranch);
        let mut wcurs = CursorWrite::new(root, BV_CAPACITY + 2);
        assert!(wcurs.verify());

        for i in 0..BV_CAPACITY {
            let k = 100 + (10 * i);
            wcurs.path_clone(&k);
        }
        assert!(wcurs.verify());

        wcurs.remove(&10);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_14() {
        // Test leaf borrow left
        let lnode = create_leaf_node_full(10);
        let rnode = create_leaf_node(20);
        let root = Node::new_branch(0, lnode, rnode);
        let mut wcurs = CursorWrite::new(root, L_CAPACITY + 1);
        assert!(wcurs.verify());

        wcurs.remove(&20);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_15() {
        // Test leaf borrow right.
        let lnode = create_leaf_node(10);
        let rnode = create_leaf_node_full(20);
        let root = Node::new_branch(0, lnode, rnode);
        let mut wcurs = CursorWrite::new(root, L_CAPACITY + 1);
        assert!(wcurs.verify());

        wcurs.remove(&10);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    fn tree_create_rand() -> ABNode<usize, usize> {
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (1..(L_CAPACITY << 4)).collect();
        ins.shuffle(&mut rng);

        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node, 1);

        for v in ins.into_iter() {
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        let (r, _c) = wcurs.finalise();
        r
    }

    #[test]
    fn test_bptree_cursor_remove_stress_1() {
        // Insert ascending - we want to ensure the tree is a few levels deep
        // so we do this to a reasonable number.
        let node = tree_create_rand();
        let mut wcurs = CursorWrite::new(node, L_CAPACITY << 4);

        for v in 1..(L_CAPACITY << 4) {
            // println!("ITER v {}", v);
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
        }
        // println!("{:?}", wcurs);
        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_stress_2() {
        // Insert descending
        let node = tree_create_rand();
        let mut wcurs = CursorWrite::new(node, L_CAPACITY << 4);

        for v in (1..(L_CAPACITY << 4)).rev() {
            // println!("ITER v {}", v);
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
        }
        // println!("{:?}", wcurs);
        // println!("DENSITY -> {:?}", wcurs.get_tree_density());
        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_stress_3() {
        // Insert random
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (1..(L_CAPACITY << 4)).collect();
        ins.shuffle(&mut rng);

        let node = tree_create_rand();
        let mut wcurs = CursorWrite::new(node, L_CAPACITY << 4);

        for v in ins.into_iter() {
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
        }
        // println!("{:?}", wcurs);
        // println!("DENSITY -> {:?}", wcurs.get_tree_density());
        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        check_drop_count();
    }

    // Add transaction-ised versions.
    #[test]
    fn test_bptree_cursor_remove_stress_4() {
        // Insert ascending - we want to ensure the tree is a few levels deep
        // so we do this to a reasonable number.
        let mut node = tree_create_rand();
        let mut count = L_CAPACITY << 4;

        for v in 1..(L_CAPACITY << 4) {
            let mut wcurs = CursorWrite::new(node, count);
            // println!("ITER v {}", v);
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
            let (n, c) = wcurs.finalise();
            node = n;
            count = c;
        }
        // println!("{:?}", node);
        // On shutdown, check we dropped all as needed.
        mem::drop(node);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_stress_5() {
        // Insert descending
        let mut node = tree_create_rand();
        let mut count = L_CAPACITY << 4;

        for v in (1..(L_CAPACITY << 4)).rev() {
            let mut wcurs = CursorWrite::new(node, count);
            // println!("ITER v {}", v);
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
            let (n, c) = wcurs.finalise();
            node = n;
            count = c;
        }
        // println!("{:?}", node);
        // On shutdown, check we dropped all as needed.
        mem::drop(node);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_stress_6() {
        // Insert random
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (1..(L_CAPACITY << 4)).collect();
        ins.shuffle(&mut rng);

        let mut node = tree_create_rand();
        let mut count = L_CAPACITY << 4;

        for v in ins.into_iter() {
            let mut wcurs = CursorWrite::new(node, count);
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
            let (n, c) = wcurs.finalise();
            node = n;
            count = c;
        }
        // println!("{:?}", node);
        // On shutdown, check we dropped all as needed.
        mem::drop(node);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_remove_stress_7() {
        // Insert random
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (1..10240).collect();

        let mut wcurs = CursorWrite::new(Node::new_ableaf(0), 0);
        wcurs.extend(ins.iter().map(|v| (*v, *v)));

        // ins.shuffle(&mut rng);

        let mut compacts = 0;

        for v in ins.into_iter() {
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
            let (l, m) = wcurs.tree_density();
            if l > 0 && (m / l) > 1 {
                compacts += 1;
            }
        }
        println!("compacts {:?}", compacts);
    }

    // This is for setting up trees that are specialised for the split off tests.
    // This is because we can exercise a LOT of complex edge cases by bracketing
    // within this tree. It also works on both node sizes.
    //
    // This is a 16 node tree, with 4 branches and a root. We have 2 values per leaf to
    // allow some cases to be explored. We also need "gaps" between the values to allow other
    // cases.
    //
    // Effectively this means we can test by splitoff on the values:
    // for i in [0,100,200,300]:
    //     for j in [0, 10, 20, 30]:
    //         t1 = i + j
    //         t2 = i + j + 1
    //         t3 = i + j + 2
    //         t4 = i + j + 3
    //
    fn create_split_off_leaf(base: usize) -> ABNode<usize, usize> {
        let mut l = Arc::new(Node::new_leaf(0));
        let lref = Arc::get_mut(&mut l).unwrap().as_mut_leaf();
        lref.insert_or_update(0, base + 1, base + 1);
        lref.insert_or_update(0, base + 2, base + 2);
        l
    }

    fn create_split_off_branch(base: usize) -> ABNode<usize, usize> {
        // This is a helper for create_split_off_tree to make the sub-branches based
        // on a base.
        let l1 = create_split_off_leaf(base);
        let l2 = create_split_off_leaf(base + 10);
        let l3 = create_split_off_leaf(base + 20);
        let l4 = create_split_off_leaf(base + 30);

        let mut branch = Node::new_branch(0, l1, l2);
        let nref = Arc::get_mut(&mut branch).unwrap().as_mut_branch();
        nref.add_node(l3);
        nref.add_node(l4);

        branch
    }

    fn create_split_off_tree() -> ABNode<usize, usize> {
        let b1 = create_split_off_branch(0);
        let b2 = create_split_off_branch(100);
        let b3 = create_split_off_branch(200);
        let b4 = create_split_off_branch(300);
        let mut root = Node::new_branch(0, b1, b2);
        let nref = Arc::get_mut(&mut root).unwrap().as_mut_branch();
        nref.add_node(b3);
        nref.add_node(b4);

        root
    }

    #[test]
    fn test_bptree_cursor_split_off_lt_01() {
        // Make a tree witth just a leaf
        // Do a split_off_lt.
        let node = create_leaf_node(0);
        let mut wcurs = CursorWrite::new(node, 1);

        wcurs.split_off_lt(&5);

        // Remember, all the cases of the remove_lte are already tested on
        // leaf.
        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_split_off_lt_02() {
        // Make a tree witth just a leaf
        // Do a split_off_lt.
        let node = create_leaf_node_full(10);
        let mut wcurs = CursorWrite::new(node, 1);

        wcurs.split_off_lt(&11);

        // Remember, all the cases of the remove_lte are already tested on
        // leaf.
        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_split_off_lt_03() {
        // Make a tree witth just a leaf
        // Do a split_off_lt.
        let node = create_leaf_node_full(10);
        let mut wcurs = CursorWrite::new(node, 1);

        wcurs.path_clone(&11);
        wcurs.split_off_lt(&11);

        // Remember, all the cases of the remove_lte are already tested on
        // leaf.
        assert!(wcurs.verify());
        mem::drop(wcurs);
        check_drop_count();
    }

    fn run_split_off_test_clone(v: usize, exp: usize) {
        // println!("RUNNING -> {:?}", v);
        let tree = create_split_off_tree();

        let mut wcurs = CursorWrite::new(tree, 32);
        // 0 is min, and not present, will cause no change.
        // clone everything
        let outer: [usize; 4] = [0, 100, 200, 300];
        let inner: [usize; 4] = [0, 10, 20, 30];
        for i in outer.iter() {
            for j in inner.iter() {
                wcurs.path_clone(&(i + j + 1));
            }
        }

        wcurs.split_off_lt(&v);
        assert!(wcurs.verify());
        if v > 0 {
            assert!(!wcurs.contains_key(&(v - 1)));
        }
        // assert!(wcurs.len() == exp);

        // println!("{:?}", wcurs);
        mem::drop(wcurs);
        check_drop_count();
    }

    fn run_split_off_test(v: usize, exp: usize) {
        println!("RUNNING -> {:?}", v);
        let tree = create_split_off_tree();
        println!("START -> {:?}", tree);

        let mut wcurs = CursorWrite::new(tree, 32);
        // 0 is min, and not present, will cause no change.
        wcurs.split_off_lt(&v);
        assert!(wcurs.verify());
        if v > 0 {
            assert!(!wcurs.contains_key(&(v - 1)));
        }
        // assert!(wcurs.len() == exp);

        // println!("{:?}", wcurs);
        mem::drop(wcurs);
        check_drop_count();
    }

    #[test]
    fn test_bptree_cursor_split_off_lt_clone_stress() {
        if CAPACITY < 10 {
            // Can't proceed as the "fake" tree we make is invalid.
            return;
        }
        let outer: [usize; 4] = [0, 100, 200, 300];
        let inner: [usize; 4] = [0, 10, 20, 30];
        for i in outer.iter() {
            for j in inner.iter() {
                run_split_off_test_clone(i + j, 32);
                run_split_off_test_clone(i + j + 1, 32);
                run_split_off_test_clone(i + j + 2, 32);
                run_split_off_test_clone(i + j + 3, 32);
            }
        }
    }

    #[test]
    fn test_bptree_cursor_split_off_lt_stress() {
        if CAPACITY < 10 {
            // Can't proceed as the "fake" tree we make is invalid.
            return;
        }
        let outer: [usize; 4] = [0, 100, 200, 300];
        let inner: [usize; 4] = [0, 10, 20, 30];
        for i in outer.iter() {
            for j in inner.iter() {
                run_split_off_test(i + j, 32);
                run_split_off_test(i + j + 1, 32);
                run_split_off_test(i + j + 2, 32);
                run_split_off_test(i + j + 3, 32);
            }
        }
    }

    #[test]
    fn test_bptree_cursor_split_off_lt_random_stress() {
        let data: Vec<isize> = (0..1024).collect();

        for v in data.iter() {
            let mut wcurs = CursorWrite::new(Node::new_ableaf(0), 0);
            wcurs.extend(data.iter().map(|v| (v, v)));

            if v > &0 {
                assert!(wcurs.contains_key(&(v - 1)));
            }

            wcurs.split_off_lt(&v);
            assert!(!wcurs.contains_key(&(v - 1)));
            assert!(wcurs.verify());
            let contents: Vec<_> = wcurs.k_iter().collect();
            assert!(contents[0] == &v);
            assert!(contents.len() as isize == (1024 - v));
        }
    }

    /*
    #[test]
    fn test_bptree_cursor_get_mut_ref_1() {
        // Test that we can clone a path (new txn)
        // Test that we don't re-clone.
        let lnode = create_leaf_node_full(10);
        let rnode = create_leaf_node_full(20);
        let root = Node::new_branch(0, lnode, rnode);
        let mut wcurs = CursorWrite::new(root, 0);
        assert!(wcurs.verify());

        let r1 = wcurs.get_mut_ref(&10);
        std::mem::drop(r1);
        let r1 = wcurs.get_mut_ref(&10);
        std::mem::drop(r1);
    }
    */

}
