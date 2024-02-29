// The cursor is what actually knits a tree together from the parts
// we have, and has an important role to keep the system consistent.
//
// Additionally, the cursor also is responsible for general movement
// throughout the structure and how to handle that effectively

use super::node::*;
use crate::internals::lincowcell::LinCowCellCapable;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::mem;

use super::iter::{Iter, KeyIter, RangeIter, ValueIter};
use super::mutiter::RangeMutIter;
use super::states::*;
use std::ops::RangeBounds;

use std::sync::Mutex;

/// The internal root of the tree, with associated garbage lists etc.
#[derive(Debug)]
pub(crate) struct SuperBlock<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    root: *mut Node<K, V>,
    size: usize,
    txid: u64,
}

unsafe impl<K: Clone + Ord + Debug + Send + 'static, V: Clone + Send + 'static> Send
    for SuperBlock<K, V>
{
}
unsafe impl<K: Clone + Ord + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static> Sync
    for SuperBlock<K, V>
{
}

impl<K: Clone + Ord + Debug, V: Clone> LinCowCellCapable<CursorRead<K, V>, CursorWrite<K, V>>
    for SuperBlock<K, V>
{
    fn create_reader(&self) -> CursorRead<K, V> {
        // This sets up the first reader.
        CursorRead::new(self)
    }

    fn create_writer(&self) -> CursorWrite<K, V> {
        // Create a writer.
        CursorWrite::new(self)
    }

    fn pre_commit(
        &mut self,
        mut new: CursorWrite<K, V>,
        prev: &CursorRead<K, V>,
    ) -> CursorRead<K, V> {
        let mut prev_last_seen = prev.last_seen.lock().unwrap();
        debug_assert!((*prev_last_seen).is_empty());

        let new_last_seen = &mut new.last_seen;
        // swap the two lists. We should now have "empty"
        std::mem::swap(&mut (*prev_last_seen), &mut (*new_last_seen));
        debug_assert!((*new_last_seen).is_empty());
        // Now when the lock is dropped, both sides see the correct info and garbage for drops.

        // We are done, time to seal everything.
        new.first_seen.iter().for_each(|n| unsafe {
            (**n).make_ro();
        });
        // Clear first seen, we won't be dropping them from here.
        new.first_seen.clear();

        // == Push data into our sb. ==
        self.root = new.root;
        self.size = new.length;
        self.txid = new.txid;

        // Create the new reader.
        CursorRead::new(self)
    }
}

impl<K: Clone + Ord + Debug, V: Clone> SuperBlock<K, V> {
    /// This is UNSAFE because you *MUST* understand how to manage the transactions
    /// of this type and to give a correct linearised transaction manager the ability
    /// to control this.
    ///
    /// More than likely, you WILL NOT do this so you should RUN AWAY and try to forget
    /// you ever saw this function at all.
    pub unsafe fn new() -> Self {
        let leaf: *mut Leaf<K, V> = Node::new_leaf(1);
        SuperBlock {
            root: leaf as *mut Node<K, V>,
            size: 0,
            txid: 1,
        }
    }

    #[cfg(test)]
    pub(crate) fn new_test(txid: u64, root: *mut Node<K, V>) -> Self {
        assert!(txid < (TXID_MASK >> TXID_SHF));
        assert!(txid > 0);
        // let last_seen: Vec<*mut Node<K, V>> = Vec::with_capacity(16);
        let mut first_seen = Vec::with_capacity(16);
        // Do a pre-verify to be sure it's sane.
        assert!(unsafe { (*root).verify() });
        // Collect anythinng from root into this txid if needed.
        // Set txid to txid on all tree nodes from the root.
        first_seen.push(root);
        unsafe { (*root).sblock_collect(&mut first_seen) };

        // Lock them all
        first_seen.iter().for_each(|n| unsafe {
            (**n).make_ro();
        });

        // Determine our count internally.
        let (length, _) = unsafe { (*root).tree_density() };

        // Good to go!
        SuperBlock {
            txid,
            size: length,
            root,
        }
    }
}

#[derive(Debug)]
pub(crate) struct CursorRead<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    txid: u64,
    length: usize,
    root: *mut Node<K, V>,
    last_seen: Mutex<Vec<*mut Node<K, V>>>,
}

unsafe impl<K: Clone + Ord + Debug + Send + 'static, V: Clone + Send + 'static> Send
    for CursorRead<K, V>
{
}
unsafe impl<K: Clone + Ord + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static> Sync
    for CursorRead<K, V>
{
}

#[derive(Debug)]
pub(crate) struct CursorWrite<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    txid: u64,
    length: usize,
    root: *mut Node<K, V>,
    last_seen: Vec<*mut Node<K, V>>,
    first_seen: Vec<*mut Node<K, V>>,
}

unsafe impl<K: Clone + Ord + Debug + Send + 'static, V: Clone + Send + 'static> Send
    for CursorWrite<K, V>
{
}
unsafe impl<K: Clone + Ord + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static> Sync
    for CursorWrite<K, V>
{
}

pub(crate) trait CursorReadOps<K: Clone + Ord + Debug, V: Clone> {
    #[allow(unused)]
    fn get_root_ref(&self) -> &Node<K, V>;

    fn get_root(&self) -> *mut Node<K, V>;

    fn len(&self) -> usize;

    fn get_txid(&self) -> u64;

    #[cfg(test)]
    fn get_tree_density(&self) -> (usize, usize) {
        // Walk the tree and calculate the packing effeciency.
        let rref = self.get_root_ref();
        rref.tree_density()
    }

    fn search<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut node = self.get_root();
        for _i in 0..65536 {
            if unsafe { (*node).is_leaf() } {
                let lref = leaf_ref!(node, K, V);
                return lref.get_ref(k).map(|v| unsafe {
                    // Strip the lifetime and rebind to the lifetime of `self`.
                    // This is safe because we know that these nodes will NOT
                    // be altered during the lifetime of this txn, so the references
                    // will remain stable.
                    let x = v as *const V;
                    &*x as &V
                });
            } else {
                let bref = branch_ref!(node, K, V);
                let idx = bref.locate_node(k);
                node = bref.get_idx_unchecked(idx);
            }
        }
        panic!("Tree depth exceeded max limit (65536). This may indicate memory corruption.");
    }

    fn contains_key<Q>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.search(k).is_some()
    }

    fn first_key_value(&self) -> Option<(&K, &V)> {
        let mut node = self.get_root();
        for _i in 0..65536 {
            if unsafe { (*node).is_leaf() } {
                let lref = leaf_ref!(node, K, V);
                return lref.min_value();
            } else {
                let bref = branch_ref!(node, K, V);
                node = bref.min_node();
            }
        }
        panic!("Tree depth exceeded max limit (65536). This may indicate memory corruption.");
    }

    fn last_key_value(&self) -> Option<(&K, &V)> {
        let mut node = self.get_root();
        for _i in 0..65536 {
            if unsafe { (*node).is_leaf() } {
                let lref = leaf_ref!(node, K, V);
                return lref.max_value();
            } else {
                let bref = branch_ref!(node, K, V);
                node = bref.max_node();
            }
        }
        panic!("Tree depth exceeded max limit (65536). This may indicate memory corruption.");
    }

    fn range<'n, R, T>(&'n self, range: R) -> RangeIter<'n, '_, K, V>
    where
        K: Borrow<T>,
        T: Ord + ?Sized,
        R: RangeBounds<T>,
    {
        RangeIter::new(self.get_root(), range, self.len())
    }

    fn kv_iter<'n>(&'n self) -> Iter<'n, '_, K, V> {
        Iter::new(self.get_root(), self.len())
    }

    fn k_iter<'n>(&'n self) -> KeyIter<'n, '_, K, V> {
        KeyIter::new(self.get_root(), self.len())
    }

    fn v_iter<'n>(&'n self) -> ValueIter<'n, '_, K, V> {
        ValueIter::new(self.get_root(), self.len())
    }

    #[cfg(test)]
    fn verify(&self) -> bool {
        self.get_root_ref().no_cycles() && self.get_root_ref().verify() && {
            let (l, _) = self.get_tree_density();
            l == self.len()
        }
    }
}

impl<K: Clone + Ord + Debug, V: Clone> CursorWrite<K, V> {
    pub(crate) fn new(sblock: &SuperBlock<K, V>) -> Self {
        let txid = sblock.txid + 1;
        assert!(txid < (TXID_MASK >> TXID_SHF));
        // println!("starting wr txid -> {:?}", txid);
        let length = sblock.size;
        let root = sblock.root;
        // TODO: Could optimise how big these are based
        // on past trends? Or based on % tree size?
        let last_seen = Vec::with_capacity(16);
        let first_seen = Vec::with_capacity(16);

        CursorWrite {
            txid,
            length,
            root,
            last_seen,
            first_seen,
        }
    }

    pub(crate) fn clear(&mut self) {
        // Reset the values in this tree.
        // We need to mark everything as disposable, and create a new root!
        self.last_seen.push(self.root);
        unsafe { (*self.root).sblock_collect(&mut self.last_seen) };
        let nroot: *mut Leaf<K, V> = Node::new_leaf(self.txid);
        let mut nroot = nroot as *mut Node<K, V>;
        self.first_seen.push(nroot);
        mem::swap(&mut self.root, &mut nroot);
        self.length = 0;
    }

    // Functions as insert_or_update
    pub(crate) fn insert(&mut self, k: K, v: V) -> Option<V> {
        let r = match clone_and_insert(
            self.root,
            self.txid,
            k,
            v,
            &mut self.last_seen,
            &mut self.first_seen,
        ) {
            CRInsertState::NoClone(res) => res,
            CRInsertState::Clone(res, mut nnode) => {
                // We have a new root node, swap it in.
                // !!! It's already been cloned and marked for cleaning by the clone_and_insert
                // call.
                // eprintln!("swap: {:?}, {:?}", self.root, nnode);
                mem::swap(&mut self.root, &mut nnode);
                // Return the insert result
                res
            }
            CRInsertState::CloneSplit(lnode, rnode) => {
                // The previous root had to split - make a new
                // root now and put it inplace.
                let mut nroot = Node::new_branch(self.txid, lnode, rnode) as *mut Node<K, V>;
                self.first_seen.push(nroot);
                // The root was cloned as part of clone split
                // This swaps the POINTERS not the content!
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
                let mut nroot = Node::new_branch(self.txid, self.root, rnode) as *mut Node<K, V>;
                self.first_seen.push(nroot);
                // println!("ls push 2");
                // self.last_seen.push(self.root);
                mem::swap(&mut self.root, &mut nroot);
                // As we split, there must NOT have been an existing
                // key to overwrite.
                None
            }
            CRInsertState::RevSplit(lnode) => {
                let mut nroot = Node::new_branch(self.txid, lnode, self.root) as *mut Node<K, V>;
                self.first_seen.push(nroot);
                // println!("ls push 3");
                // self.last_seen.push(self.root);
                mem::swap(&mut self.root, &mut nroot);
                None
            }
            CRInsertState::CloneRevSplit(rnode, lnode) => {
                let mut nroot = Node::new_branch(self.txid, lnode, rnode) as *mut Node<K, V>;
                self.first_seen.push(nroot);
                // root was cloned in the rev split
                // println!("ls push 4");
                // self.last_seen.push(self.root);
                mem::swap(&mut self.root, &mut nroot);
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
        let r = match clone_and_remove(
            self.root,
            self.txid,
            k,
            &mut self.last_seen,
            &mut self.first_seen,
        ) {
            CRRemoveState::NoClone(res) => res,
            CRRemoveState::Clone(res, mut nnode) => {
                mem::swap(&mut self.root, &mut nnode);
                res
            }
            CRRemoveState::Shrink(res) => {
                if self_meta!(self.root).is_leaf() {
                    // No action - we have an empty tree.
                    res
                } else {
                    // Root is being demoted, get the last branch and
                    // promote it to the root.
                    self.last_seen.push(self.root);
                    let rmut = branch_ref!(self.root, K, V);
                    let mut pnode = rmut.extract_last_node();
                    mem::swap(&mut self.root, &mut pnode);
                    res
                }
            }
            CRRemoveState::CloneShrink(res, mut nnode) => {
                if self_meta!(nnode).is_leaf() {
                    // The tree is empty, but we cloned the root to get here.
                    mem::swap(&mut self.root, &mut nnode);
                    res
                } else {
                    // Our root is getting demoted here, get the remaining branch
                    self.last_seen.push(nnode);
                    let rmut = branch_ref!(nnode, K, V);
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
        match path_clone(
            self.root,
            self.txid,
            k,
            &mut self.last_seen,
            &mut self.first_seen,
        ) {
            CRCloneState::Clone(mut nroot) => {
                // We cloned the root, so swap it.
                mem::swap(&mut self.root, &mut nroot);
            }
            CRCloneState::NoClone => {}
        };
    }

    pub(crate) fn get_mut_ref(&mut self, k: &K) -> Option<&mut V> {
        match path_clone(
            self.root,
            self.txid,
            k,
            &mut self.last_seen,
            &mut self.first_seen,
        ) {
            CRCloneState::Clone(mut nroot) => {
                // We cloned the root, so swap it.
                mem::swap(&mut self.root, &mut nroot);
            }
            CRCloneState::NoClone => {}
        };
        // Now get the ref.
        path_get_mut_ref(self.root, k)
    }

    pub(crate) fn split_off_lt(&mut self, k: &K) {
        /*
        // Remove all the values less than from the top of the tree.
        loop {
            let result = clone_and_split_off_trim_lt(
                self.root,
                self.txid,
                k,
                &mut self.last_seen,
                &mut self.first_seen,
            );
            // println!("clone_and_split_off_trim_lt -> {:?}", result);
            match result {
                CRTrimState::Complete => break,
                CRTrimState::Clone(mut nroot) => {
                    // We cloned the root as we changed it, but don't need
                    // to recurse so we break the loop.
                    mem::swap(&mut self.root, &mut nroot);
                    break;
                }
                CRTrimState::Promote(mut nroot) => {
                    mem::swap(&mut self.root, &mut nroot);
                    // This will continue and try again.
                }
            }
        }
        */

        /*
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
        */

        // Get rid of anything else dangling
        let mut rmkeys: Vec<K> = Vec::new();
        for ki in self.k_iter() {
            if ki >= k {
                break;
            }
            rmkeys.push(ki.clone());
        }

        for kr in rmkeys.into_iter() {
            let _ = self.remove(&kr);
        }

        // Iterate over the remaining kv's to fix our k,v count.
        let newsize = self.kv_iter().count();
        self.length = newsize;
    }

    #[cfg(test)]
    pub(crate) fn root_txid(&self) -> u64 {
        self.get_root_ref().get_txid()
    }

    #[cfg(test)]
    pub(crate) fn tree_density(&self) -> (usize, usize) {
        self.get_root_ref().tree_density()
    }

    pub(crate) fn range_mut<'n, R, T>(&'n mut self, range: R) -> RangeMutIter<'n, '_, K, V>
    where
        K: Borrow<T>,
        T: Ord + ?Sized,
        R: RangeBounds<T>,
    {
        RangeMutIter::new(self, range)
    }
}

impl<K: Clone + Ord + Debug, V: Clone> Extend<(K, V)> for CursorWrite<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        iter.into_iter().for_each(|(k, v)| {
            let _ = self.insert(k, v);
        });
    }
}

impl<K: Clone + Ord + Debug, V: Clone> Drop for CursorWrite<K, V> {
    fn drop(&mut self) {
        // If there is content in first_seen, this means we aborted and must rollback
        // of these items!
        // println!("Releasing CW FS -> {:?}", self.first_seen);
        self.first_seen.iter().for_each(|n| Node::free(*n))
    }
}

impl<K: Clone + Ord + Debug, V: Clone> Drop for CursorRead<K, V> {
    fn drop(&mut self) {
        // If there is content in last_seen, a future generation wants us to remove it!
        let last_seen_guard = self
            .last_seen
            .try_lock()
            .expect("Unable to lock, something is horridly wrong!");
        last_seen_guard.iter().for_each(|n| Node::free(*n));
        std::mem::drop(last_seen_guard);
    }
}

impl<K: Clone + Ord + Debug, V: Clone> Drop for SuperBlock<K, V> {
    fn drop(&mut self) {
        // eprintln!("Releasing SuperBlock ...");
        // We must be the last SB and no txns exist. Drop the tree now.
        // TODO: Calc this based on size.
        let mut first_seen = Vec::with_capacity(16);
        // eprintln!("{:?}", self.root);
        first_seen.push(self.root);
        unsafe { (*self.root).sblock_collect(&mut first_seen) };
        first_seen.iter().for_each(|n| Node::free(*n));
    }
}

impl<K: Clone + Ord + Debug, V: Clone> CursorRead<K, V> {
    pub(crate) fn new(sblock: &SuperBlock<K, V>) -> Self {
        // println!("starting rd txid -> {:?}", sblock.txid);
        CursorRead {
            txid: sblock.txid,
            length: sblock.size,
            root: sblock.root,
            last_seen: Mutex::new(Vec::with_capacity(0)),
        }
    }
}

impl<K: Clone + Ord + Debug, V: Clone> CursorReadOps<K, V> for CursorRead<K, V> {
    fn get_root_ref(&self) -> &Node<K, V> {
        unsafe { &*(self.root) }
    }

    fn get_root(&self) -> *mut Node<K, V> {
        self.root
    }

    fn len(&self) -> usize {
        self.length
    }

    fn get_txid(&self) -> u64 {
        self.txid
    }
}

impl<K: Clone + Ord + Debug, V: Clone> CursorReadOps<K, V> for CursorWrite<K, V> {
    fn get_root_ref(&self) -> &Node<K, V> {
        unsafe { &*(self.root) }
    }

    fn get_root(&self) -> *mut Node<K, V> {
        self.root
    }

    fn len(&self) -> usize {
        self.length
    }

    fn get_txid(&self) -> u64 {
        self.txid
    }
}

fn clone_and_insert<K: Clone + Ord + Debug, V: Clone>(
    node: *mut Node<K, V>,
    txid: u64,
    k: K,
    v: V,
    last_seen: &mut Vec<*mut Node<K, V>>,
    first_seen: &mut Vec<*mut Node<K, V>>,
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

    if self_meta!(node).is_leaf() {
        // NOTE: We have to match, rather than map here, as rust tries to
        // move k:v into both closures!

        // Leaf path
        match leaf_ref!(node, K, V).req_clone(txid) {
            Some(cnode) => {
                // println!();
                first_seen.push(cnode);
                // println!("ls push 5");
                last_seen.push(node);
                // Clone was required.
                let mref = leaf_ref!(cnode, K, V);
                // insert to the new node.
                match mref.insert_or_update(k, v) {
                    LeafInsertState::Ok(res) => CRInsertState::Clone(res, cnode),
                    LeafInsertState::Split(rnode) => {
                        first_seen.push(rnode as *mut Node<K, V>);
                        // let rnode = Node::new_leaf_ins(txid, sk, sv);
                        CRInsertState::CloneSplit(cnode, rnode as *mut Node<K, V>)
                    }
                    LeafInsertState::RevSplit(lnode) => {
                        first_seen.push(lnode as *mut Node<K, V>);
                        CRInsertState::CloneRevSplit(cnode, lnode as *mut Node<K, V>)
                    }
                }
            }
            None => {
                // No clone required.
                // simply do the insert.
                let mref = leaf_ref!(node, K, V);
                match mref.insert_or_update(k, v) {
                    LeafInsertState::Ok(res) => CRInsertState::NoClone(res),
                    LeafInsertState::Split(rnode) => {
                        // We split, but left is already part of the txn group, so lets
                        // just return what's new.
                        // let rnode = Node::new_leaf_ins(txid, sk, sv);
                        first_seen.push(rnode as *mut Node<K, V>);
                        CRInsertState::Split(rnode as *mut Node<K, V>)
                    }
                    LeafInsertState::RevSplit(lnode) => {
                        first_seen.push(lnode as *mut Node<K, V>);
                        CRInsertState::RevSplit(lnode as *mut Node<K, V>)
                    }
                }
            }
        } // end match
    } else {
        // Branch path
        // Decide if we need to clone - we do this as we descend due to a quirk in Arc
        // get_mut, because we don't have access to get_mut_unchecked (and this api may
        // never be stabilised anyway). When we change this to *mut + garbage lists we
        // could consider restoring the reactive behaviour that clones up, rather than
        // cloning down the path.
        //
        // NOTE: We have to match, rather than map here, as rust tries to
        // move k:v into both closures!
        match branch_ref!(node, K, V).req_clone(txid) {
            Some(cnode) => {
                first_seen.push(cnode);
                last_seen.push(node);
                // Not same txn, clone instead.
                let nmref = branch_ref!(cnode, K, V);
                let anode_idx = nmref.locate_node(&k);
                let anode = nmref.get_idx_unchecked(anode_idx);

                match clone_and_insert(anode, txid, k, v, last_seen, first_seen) {
                    CRInsertState::Clone(res, lnode) => {
                        nmref.replace_by_idx(anode_idx, lnode);
                        // Pass back up that we cloned.
                        CRInsertState::Clone(res, cnode)
                    }
                    CRInsertState::CloneSplit(lnode, rnode) => {
                        // CloneSplit here, would have already updated lnode/rnode into the
                        // gc lists.
                        // Second, we update anode_idx node with our lnode as the new clone.
                        nmref.replace_by_idx(anode_idx, lnode);

                        // Third we insert rnode - perfect world it's at anode_idx + 1, but
                        // we use the normal insert routine for now.
                        match nmref.add_node(rnode) {
                            BranchInsertState::Ok => CRInsertState::Clone(None, cnode),
                            BranchInsertState::Split(clnode, crnode) => {
                                // Create a new branch to hold these children.
                                let nrnode = Node::new_branch(txid, clnode, crnode);
                                first_seen.push(nrnode as *mut Node<K, V>);
                                // Return it
                                CRInsertState::CloneSplit(cnode, nrnode as *mut Node<K, V>)
                            }
                        }
                    }
                    CRInsertState::CloneRevSplit(nnode, lnode) => {
                        nmref.replace_by_idx(anode_idx, nnode);
                        match nmref.add_node_left(lnode, anode_idx) {
                            BranchInsertState::Ok => CRInsertState::Clone(None, cnode),
                            BranchInsertState::Split(clnode, crnode) => {
                                let nrnode = Node::new_branch(txid, clnode, crnode);
                                first_seen.push(nrnode as *mut Node<K, V>);
                                CRInsertState::CloneSplit(cnode, nrnode as *mut Node<K, V>)
                            }
                        }
                    }
                    CRInsertState::NoClone(_res) => {
                        // If our descendant did not clone, then we don't have to either.
                        unreachable!("Shoud never be possible.");
                        // CRInsertState::NoClone(res)
                    }
                    CRInsertState::Split(_rnode) => {
                        // I think
                        unreachable!("This represents a corrupt tree state");
                    }
                    CRInsertState::RevSplit(_lnode) => {
                        unreachable!("This represents a corrupt tree state");
                    }
                } // end match
            } // end Some,
            None => {
                let nmref = branch_ref!(node, K, V);
                let anode_idx = nmref.locate_node(&k);
                let anode = nmref.get_idx_unchecked(anode_idx);

                match clone_and_insert(anode, txid, k, v, last_seen, first_seen) {
                    CRInsertState::Clone(res, lnode) => {
                        nmref.replace_by_idx(anode_idx, lnode);
                        // We did not clone, and no further work needed.
                        CRInsertState::NoClone(res)
                    }
                    CRInsertState::NoClone(res) => {
                        // If our descendant did not clone, then we don't have to do any adjustments
                        // or further work.
                        CRInsertState::NoClone(res)
                    }
                    CRInsertState::Split(rnode) => {
                        match nmref.add_node(rnode) {
                            // Similar to CloneSplit - we are either okay, and the insert was happy.
                            BranchInsertState::Ok => CRInsertState::NoClone(None),
                            // Or *we* split as well, and need to return a new sibling branch.
                            BranchInsertState::Split(clnode, crnode) => {
                                // Create a new branch to hold these children.
                                let nrnode = Node::new_branch(txid, clnode, crnode);
                                first_seen.push(nrnode as *mut Node<K, V>);
                                // Return it
                                CRInsertState::Split(nrnode as *mut Node<K, V>)
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
                            BranchInsertState::Ok => CRInsertState::NoClone(None),
                            // Or *we* split as well, and need to return a new sibling branch.
                            BranchInsertState::Split(clnode, crnode) => {
                                // Create a new branch to hold these children.
                                let nrnode = Node::new_branch(txid, clnode, crnode);
                                first_seen.push(nrnode as *mut Node<K, V>);
                                // Return it
                                CRInsertState::Split(nrnode as *mut Node<K, V>)
                            }
                        }
                    }
                    CRInsertState::RevSplit(lnode) => match nmref.add_node_left(lnode, anode_idx) {
                        BranchInsertState::Ok => CRInsertState::NoClone(None),
                        BranchInsertState::Split(clnode, crnode) => {
                            let nrnode = Node::new_branch(txid, clnode, crnode);
                            first_seen.push(nrnode as *mut Node<K, V>);
                            CRInsertState::Split(nrnode as *mut Node<K, V>)
                        }
                    },
                    CRInsertState::CloneRevSplit(nnode, lnode) => {
                        nmref.replace_by_idx(anode_idx, nnode);
                        match nmref.add_node_left(lnode, anode_idx) {
                            BranchInsertState::Ok => CRInsertState::NoClone(None),
                            BranchInsertState::Split(clnode, crnode) => {
                                let nrnode = Node::new_branch(txid, clnode, crnode);
                                first_seen.push(nrnode as *mut Node<K, V>);
                                CRInsertState::Split(nrnode as *mut Node<K, V>)
                            }
                        }
                    }
                } // end match
            }
        } // end match branch ref clone
    } // end if leaf
}

fn path_clone<K: Clone + Ord + Debug, V: Clone>(
    node: *mut Node<K, V>,
    txid: u64,
    k: &K,
    last_seen: &mut Vec<*mut Node<K, V>>,
    first_seen: &mut Vec<*mut Node<K, V>>,
) -> CRCloneState<K, V> {
    if unsafe { (*node).is_leaf() } {
        unsafe {
            (*(node as *mut Leaf<K, V>))
                .req_clone(txid)
                .map(|cnode| {
                    // Track memory
                    last_seen.push(node);
                    // println!("ls push 7 {:?}", node);
                    first_seen.push(cnode);
                    CRCloneState::Clone(cnode)
                })
                .unwrap_or(CRCloneState::NoClone)
        }
    } else {
        // We are in a branch, so locate our descendent and prepare
        // to clone if needed.
        // println!("txid -> {:?} {:?}", node_txid, txid);
        let nmref = branch_ref!(node, K, V);
        let anode_idx = nmref.locate_node(k);
        let anode = nmref.get_idx_unchecked(anode_idx);
        match path_clone(anode, txid, k, last_seen, first_seen) {
            CRCloneState::Clone(cnode) => {
                // Do we need to clone?
                nmref
                    .req_clone(txid)
                    .map(|acnode| {
                        // We require to be cloned.
                        last_seen.push(node);
                        // println!("ls push 8");
                        first_seen.push(acnode);
                        let nmref = branch_ref!(acnode, K, V);
                        nmref.replace_by_idx(anode_idx, cnode);
                        CRCloneState::Clone(acnode)
                    })
                    .unwrap_or_else(|| {
                        // Nope, just insert and unwind.
                        nmref.replace_by_idx(anode_idx, cnode);
                        CRCloneState::NoClone
                    })
            }
            CRCloneState::NoClone => {
                // Did not clone, unwind.
                CRCloneState::NoClone
            }
        }
    }
}

fn clone_and_remove<K: Clone + Ord + Debug, V: Clone>(
    node: *mut Node<K, V>,
    txid: u64,
    k: &K,
    last_seen: &mut Vec<*mut Node<K, V>>,
    first_seen: &mut Vec<*mut Node<K, V>>,
) -> CRRemoveState<K, V> {
    if self_meta!(node).is_leaf() {
        leaf_ref!(node, K, V)
            .req_clone(txid)
            .map(|cnode| {
                first_seen.push(cnode);
                // println!("ls push 10 {:?}", node);
                last_seen.push(node);
                let mref = leaf_ref!(cnode, K, V);
                match mref.remove(k) {
                    LeafRemoveState::Ok(res) => CRRemoveState::Clone(res, cnode),
                    LeafRemoveState::Shrink(res) => CRRemoveState::CloneShrink(res, cnode),
                }
            })
            .unwrap_or_else(|| {
                let mref = leaf_ref!(node, K, V);
                match mref.remove(k) {
                    LeafRemoveState::Ok(res) => CRRemoveState::NoClone(res),
                    LeafRemoveState::Shrink(res) => CRRemoveState::Shrink(res),
                }
            })
    } else {
        // Locate the node we need to work on and then react if it
        // requests a shrink.
        branch_ref!(node, K, V)
            .req_clone(txid)
            .map(|cnode| {
                first_seen.push(cnode);
                // println!("ls push 11 {:?}", node);
                last_seen.push(node);
                // Done mm
                let nmref = branch_ref!(cnode, K, V);
                let anode_idx = nmref.locate_node(k);
                let anode = nmref.get_idx_unchecked(anode_idx);
                match clone_and_remove(anode, txid, k, last_seen, first_seen) {
                    CRRemoveState::NoClone(_res) => {
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
                        let right_idx =
                            nmref.clone_sibling_idx(txid, anode_idx, last_seen, first_seen);
                        // Okay, now work out what we need to do.
                        match nmref.shrink_decision(right_idx) {
                            BranchShrinkState::Balanced => {
                                // K:V were distributed through left and right,
                                // so no further action needed.
                                CRRemoveState::Clone(res, cnode)
                            }
                            BranchShrinkState::Merge(dnode) => {
                                // Right was merged to left, and we remain
                                // valid
                                // println!("ls push 20 {:?}", dnode);
                                debug_assert!(!last_seen.contains(&dnode));
                                last_seen.push(dnode);
                                CRRemoveState::Clone(res, cnode)
                            }
                            BranchShrinkState::Shrink(dnode) => {
                                // Right was merged to left, but we have now falled under the needed
                                // amount of values.
                                // println!("ls push 21 {:?}", dnode);
                                debug_assert!(!last_seen.contains(&dnode));
                                last_seen.push(dnode);
                                CRRemoveState::CloneShrink(res, cnode)
                            }
                        }
                    }
                }
            })
            .unwrap_or_else(|| {
                // We are already part of this txn
                let nmref = branch_ref!(node, K, V);
                let anode_idx = nmref.locate_node(k);
                let anode = nmref.get_idx_unchecked(anode_idx);
                match clone_and_remove(anode, txid, k, last_seen, first_seen) {
                    CRRemoveState::NoClone(res) => CRRemoveState::NoClone(res),
                    CRRemoveState::Clone(res, lnode) => {
                        nmref.replace_by_idx(anode_idx, lnode);
                        CRRemoveState::NoClone(res)
                    }
                    CRRemoveState::Shrink(res) => {
                        let right_idx =
                            nmref.clone_sibling_idx(txid, anode_idx, last_seen, first_seen);
                        match nmref.shrink_decision(right_idx) {
                            BranchShrinkState::Balanced => {
                                // K:V were distributed through left and right,
                                // so no further action needed.
                                CRRemoveState::NoClone(res)
                            }
                            BranchShrinkState::Merge(dnode) => {
                                // Right was merged to left, and we remain
                                // valid
                                //
                                // A quirk here is based on how clone_sibling_idx works. We may actually
                                // start with anode_idx of 0, which triggers a right clone, so it's
                                // *already* in the mm lists. But here right is "last seen" now if
                                //
                                // println!("ls push 22 {:?}", dnode);
                                debug_assert!(!last_seen.contains(&dnode));
                                last_seen.push(dnode);
                                CRRemoveState::NoClone(res)
                            }
                            BranchShrinkState::Shrink(dnode) => {
                                // Right was merged to left, but we have now falled under the needed
                                // amount of values, so we begin to shrink up.
                                // println!("ls push 23 {:?}", dnode);
                                debug_assert!(!last_seen.contains(&dnode));
                                last_seen.push(dnode);
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
                        let right_idx =
                            nmref.clone_sibling_idx(txid, anode_idx, last_seen, first_seen);
                        match nmref.shrink_decision(right_idx) {
                            BranchShrinkState::Balanced => {
                                // K:V were distributed through left and right,
                                // so no further action needed.
                                CRRemoveState::NoClone(res)
                            }
                            BranchShrinkState::Merge(dnode) => {
                                // Right was merged to left, and we remain
                                // valid
                                // println!("ls push 24 {:?}", dnode);
                                debug_assert!(!last_seen.contains(&dnode));
                                last_seen.push(dnode);
                                CRRemoveState::NoClone(res)
                            }
                            BranchShrinkState::Shrink(dnode) => {
                                // Right was merged to left, but we have now falled under the needed
                                // amount of values.
                                // println!("ls push 25 {:?}", dnode);
                                debug_assert!(!last_seen.contains(&dnode));
                                last_seen.push(dnode);
                                CRRemoveState::Shrink(res)
                            }
                        }
                    }
                }
            }) // end unwrap_or_else
    }
}

fn path_get_mut_ref<'a, K, V>(node: *mut Node<K, V>, k: &K) -> Option<&'a mut V>
where
    K: Clone + Ord + Debug + 'a,
    V: Clone,
{
    if self_meta!(node).is_leaf() {
        leaf_ref!(node, K, V).get_mut_ref(k)
    } else {
        // This nmref binds the life of the reference ...
        let nmref = branch_ref!(node, K, V);
        let anode_idx = nmref.locate_node(k);
        let anode = nmref.get_idx_unchecked(anode_idx);
        // That we get here. So we can't just return it, and we need to 'strip' the
        // lifetime so that it's bound to the lifetime of the outer node
        // rather than the nmref.
        let r: Option<*mut V> = path_get_mut_ref(anode, k).map(|v| v as *mut V);

        // I solemly swear I am up to no good.
        r.map(|v| unsafe { &mut *v as &mut V })
    }
}

/*
fn clone_and_split_off_trim_lt<K: Clone + Ord + Debug, V: Clone>(
    node: *mut Node<K, V>,
    txid: u64,
    k: &K,
    last_seen: &mut Vec<*mut Node<K, V>>,
    first_seen: &mut Vec<*mut Node<K, V>>,
) -> CRTrimState<K, V> {
    if self_meta!(node).is_leaf() {
        // No action, it's a leaf. Prune will do it.
        CRTrimState::Complete
    } else {
        branch_ref!(node, K, V)
            .req_clone(txid)
            .map(|cnode| {
                let nmref = branch_ref!(cnode, K, V);
                first_seen.push(cnode as *mut Node<K, V>);
                last_seen.push(node as *mut Node<K, V>);
                match nmref.trim_lt_key(k, last_seen, first_seen) {
                    BranchTrimState::Complete => CRTrimState::Clone(cnode),
                    BranchTrimState::Promote(pnode) => {
                        // We just cloned it but oh well, away you go ...
                        last_seen.push(cnode as *mut Node<K, V>);
                        CRTrimState::Promote(pnode)
                    }
                }
            })
            .unwrap_or_else(|| {
                let nmref = branch_ref!(node, K, V);

                match nmref.trim_lt_key(k, last_seen, first_seen) {
                    BranchTrimState::Complete => CRTrimState::Complete,
                    BranchTrimState::Promote(pnode) => {
                        // We are about to remove our node, so mark it as the last time.
                        last_seen.push(node);
                        CRTrimState::Promote(pnode)
                    }
                }
            })
    }
}
*/

/*
fn clone_and_split_off_prune_lt<K: Clone + Ord + Debug, V: Clone>(
    node: &mut ABNode<K, V>,
    txid: usize,
    k: &K,
) -> CRPruneState<K, V> {
    if node.is_leaf() {
        // I think this should be do nothing, the up walk will clean.
        if node.txid == txid {
            let nmref = Arc::get_mut(node).unwrap().as_mut_leaf();
            match nmref.remove_lt(k) {
                LeafPruneState::Ok => CRPruneState::OkNoClone,
                LeafPruneState::Prune => CRPruneState::Prune,
            }
        } else {
            let mut cnode = node.req_clone(txid);
            let nmref = Arc::get_mut(&mut cnode).unwrap().as_mut_leaf();
            match nmref.remove_lt(k) {
                LeafPruneState::Ok => CRPruneState::OkClone(cnode),
                LeafPruneState::Prune => CRPruneState::ClonePrune(cnode),
            }
        }
    } else {
        if node.txid == txid {
            let nmref = Arc::get_mut(node).unwrap().as_mut_branch();
            let anode_idx = nmref.locate_node(&k);
            let anode = nmref.get_idx_unchecked(anode_idx);
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
            let anode = nmref.get_idx_unchecked(anode_idx);
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
*/

#[cfg(test)]
mod tests {
    use super::super::node::*;
    use super::super::states::*;
    use super::SuperBlock;
    use super::{CursorRead, CursorReadOps};
    use crate::internals::lincowcell::LinCowCellCapable;
    use rand::seq::SliceRandom;
    use std::mem;

    fn create_leaf_node(v: usize) -> *mut Node<usize, usize> {
        let node = Node::new_leaf(1);
        {
            let nmut: &mut Leaf<_, _> = leaf_ref!(node, usize, usize);
            nmut.insert_or_update(v, v);
        }
        node as *mut Node<usize, usize>
    }

    fn create_leaf_node_full(vbase: usize) -> *mut Node<usize, usize> {
        assert!(vbase % 10 == 0);
        let node = Node::new_leaf(1);
        {
            let nmut = leaf_ref!(node, usize, usize);
            for idx in 0..L_CAPACITY {
                let v = vbase + idx;
                nmut.insert_or_update(v, v);
            }
            // println!("lnode full {:?} -> {:?}", vbase, nmut);
        }
        node as *mut Node<usize, usize>
    }

    fn create_branch_node_full(vbase: usize) -> *mut Node<usize, usize> {
        let l1 = create_leaf_node(vbase);
        let l2 = create_leaf_node(vbase + 10);
        let lbranch = Node::new_branch(1, l1, l2);
        let bref = branch_ref!(lbranch, usize, usize);
        for i in 2..BV_CAPACITY {
            let l = create_leaf_node(vbase + (10 * i));
            let r = bref.add_node(l);
            match r {
                BranchInsertState::Ok => {}
                _ => debug_assert!(false),
            }
        }
        assert!(bref.count() == L_CAPACITY);
        lbranch as *mut Node<usize, usize>
    }

    #[test]
    fn test_bptree2_cursor_insert_leaf() {
        // First create the node + cursor
        let node = create_leaf_node(0);
        let sb = SuperBlock::new_test(1, node);
        let mut wcurs = sb.create_writer();

        eprintln!("{:?}", wcurs);

        let prev_txid = wcurs.root_txid();
        eprintln!("prev_txid {:?}", prev_txid);

        // Now insert - the txid should be different.
        let r = wcurs.insert(1, 1);
        assert!(r.is_none());
        eprintln!("get_root_ref {:?}", wcurs.get_root_ref().meta.get_txid());
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
    fn test_bptree2_cursor_insert_split_1() {
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
        let sb = SuperBlock::new_test(1, node);
        let mut wcurs = sb.create_writer();
        let prev_txid = wcurs.root_txid();

        let r = wcurs.insert(1, 1);
        assert!(r.is_none());
        let r1_txid = wcurs.root_txid();
        assert!(r1_txid == prev_txid + 1);
        assert!(wcurs.verify());
        // println!("{:?}", wcurs);
        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_insert_split_2() {
        // Similar to split_1, but test the Split only path. This means
        // leaf needs to be below max to start, and we insert enough in-txn
        // to trigger a clone of leaf AND THEN to cause the split.
        let node = create_leaf_node(0);
        let sb = SuperBlock::new_test(1, node);
        let mut wcurs = sb.create_writer();

        for v in 1..(L_CAPACITY + 1) {
            // println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        // println!("{:?}", wcurs);
        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_insert_split_3() {
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
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();

        assert!(wcurs.verify());
        // println!("{:?}", wcurs);

        let r = wcurs.insert(19, 19);
        assert!(r.is_none());
        assert!(wcurs.verify());
        // println!("{:?}", wcurs);

        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_insert_split_4() {
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
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        assert!(wcurs.verify());

        let r = wcurs.insert(29, 29);
        assert!(r.is_none());
        assert!(wcurs.verify());
        // println!("{:?}", wcurs);

        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_insert_split_5() {
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
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
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
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_insert_split_6() {
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
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
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
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_insert_split_7() {
        //      root
        //     /    \
        //  leaf    split leaf
        // Insert to leaf then split leaf such that root has cloned
        // in step 1, but doesn't need clone in 2.
        let lnode = create_leaf_node(10);
        let rnode = create_leaf_node(20);
        let root = Node::new_branch(0, lnode, rnode);
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
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
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_insert_split_8() {
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
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
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
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_insert_stress_1() {
        // Insert ascending - we want to ensure the tree is a few levels deep
        // so we do this to a reasonable number.
        let node = create_leaf_node(0);
        let sb = SuperBlock::new_test(1, node);
        let mut wcurs = sb.create_writer();

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
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_insert_stress_2() {
        // Insert descending
        let node = create_leaf_node(0);
        let sb = SuperBlock::new_test(1, node);
        let mut wcurs = sb.create_writer();

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
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_insert_stress_3() {
        // Insert random
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (1..(L_CAPACITY << 4)).collect();
        ins.shuffle(&mut rng);

        let node = create_leaf_node(0);
        let sb = SuperBlock::new_test(1, node);
        let mut wcurs = sb.create_writer();

        for v in ins.into_iter() {
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        // println!("{:?}", wcurs);
        // println!("DENSITY -> {:?}", wcurs.get_tree_density());
        // On shutdown, check we dropped all as needed.
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    // Add transaction-ised versions.
    #[test]
    fn test_bptree2_cursor_insert_stress_4() {
        // Insert ascending - we want to ensure the tree is a few levels deep
        // so we do this to a reasonable number.
        let mut sb = unsafe { SuperBlock::new() };
        let mut rdr = sb.create_reader();

        for v in 1..(L_CAPACITY << 4) {
            let mut wcurs = sb.create_writer();
            // println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());

            rdr = sb.pre_commit(wcurs, &rdr);
        }
        // println!("{:?}", node);
        // On shutdown, check we dropped all as needed.
        mem::drop(rdr);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_insert_stress_5() {
        // Insert descending
        let mut sb = unsafe { SuperBlock::new() };
        let mut rdr = sb.create_reader();

        for v in (1..(L_CAPACITY << 4)).rev() {
            let mut wcurs = sb.create_writer();
            // println!("ITER v {}", v);
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
            rdr = sb.pre_commit(wcurs, &rdr);
        }
        // println!("{:?}", node);
        // On shutdown, check we dropped all as needed.
        mem::drop(rdr);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_insert_stress_6() {
        // Insert random
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (1..(L_CAPACITY << 4)).collect();
        ins.shuffle(&mut rng);

        let mut sb = unsafe { SuperBlock::new() };
        let mut rdr = sb.create_reader();

        for v in ins.into_iter() {
            let mut wcurs = sb.create_writer();
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
            rdr = sb.pre_commit(wcurs, &rdr);
        }
        // println!("{:?}", node);
        // On shutdown, check we dropped all as needed.
        mem::drop(rdr);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_search_1() {
        let node = create_leaf_node(0);
        let sb = SuperBlock::new_test(1, node);
        let mut wcurs = sb.create_writer();

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
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_length_1() {
        // Check the length is consistent on operations.
        let node = create_leaf_node(0);
        let sb = SuperBlock::new_test(1, node);
        let mut wcurs = sb.create_writer();

        for v in 1..(L_CAPACITY << 4) {
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
        }
        // println!("{} == {}", wcurs.len(), L_CAPACITY << 4);
        assert!(wcurs.len() == L_CAPACITY << 4);
    }

    #[test]
    fn test_bptree2_cursor_remove_01_p0() {
        // Check that a single value can be removed correctly without change.
        // Check that a missing value is removed as "None".
        // Check that emptying the root is ok.
        // BOTH of these need new txns to check clone, and then re-use txns.
        //
        //
        let lnode = create_leaf_node_full(0);
        let sb = SuperBlock::new_test(1, lnode);
        let mut wcurs = sb.create_writer();
        // println!("{:?}", wcurs);

        for v in 0..L_CAPACITY {
            let x = wcurs.remove(&v);
            // println!("{:?}", wcurs);
            assert!(x == Some(v));
        }

        for v in 0..L_CAPACITY {
            let x = wcurs.remove(&v);
            assert!(x.is_none());
        }

        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_01_p1() {
        let node = create_leaf_node(0);
        let sb = SuperBlock::new_test(1, node);
        let mut wcurs = sb.create_writer();

        let _ = wcurs.remove(&0);
        // println!("{:?}", wcurs);

        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_02() {
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
        let root = Node::new_branch(0, znode, lnode);
        // Prevent the tree shrinking.
        unsafe { (*root).add_node(rnode) };
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        // println!("{:?}", wcurs);
        assert!(wcurs.verify());

        wcurs.remove(&20);
        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_03() {
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
        let root = Node::new_branch(0, lnode, rnode);
        // Prevent the tree shrinking.
        unsafe { (*root).add_node(znode) };
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        assert!(wcurs.verify());

        wcurs.remove(&10);
        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_04p0() {
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
        let root = Node::new_branch(0, znode, lnode);
        // Prevent the tree shrinking.
        unsafe { (*root).add_node(rnode) };
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        assert!(wcurs.verify());

        // Setup sibling leaf to already be cloned.
        wcurs.path_clone(&10);
        assert!(wcurs.verify());

        wcurs.remove(&20);
        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_04p1() {
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
        let root = Node::new_branch(0, znode, lnode);
        // Prevent the tree shrinking.
        unsafe { (*root).add_node(rnode) };
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        assert!(wcurs.verify());

        // Setup leaf to already be cloned.
        wcurs.path_clone(&20);
        assert!(wcurs.verify());

        wcurs.remove(&20);
        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_05() {
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
        let root = Node::new_branch(0, lnode, rnode);
        // Prevent the tree shrinking.
        unsafe { (*root).add_node(znode) };
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        assert!(wcurs.verify());

        // Setup leaf to already be cloned.
        wcurs.path_clone(&20);

        wcurs.remove(&10);
        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_06() {
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
        let root: *mut Branch<usize, usize> =
            Node::new_branch(0, lbranch as *mut _, rbranch as *mut _);
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();

        assert!(wcurs.verify());

        wcurs.remove(&30);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_07() {
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
        let root: *mut Branch<usize, usize> =
            Node::new_branch(0, lbranch as *mut _, rbranch as *mut _);
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        assert!(wcurs.verify());

        wcurs.remove(&10);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_08() {
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

        let root: *mut Branch<usize, usize> =
            Node::new_branch(0, lbranch as *mut _, rbranch as *mut _);
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        assert!(wcurs.verify());

        wcurs.remove(&80);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_09() {
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

        let root: *mut Branch<usize, usize> =
            Node::new_branch(0, lbranch as *mut _, rbranch as *mut _);
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        assert!(wcurs.verify());

        wcurs.remove(&10);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_10() {
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
        let root: *mut Branch<usize, usize> =
            Node::new_branch(0, lbranch as *mut _, rbranch as *mut _);
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();

        assert!(wcurs.verify());

        wcurs.path_clone(&0);
        wcurs.path_clone(&10);

        wcurs.remove(&30);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_11() {
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
        let root: *mut Branch<usize, usize> =
            Node::new_branch(0, lbranch as *mut _, rbranch as *mut _);
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        assert!(wcurs.verify());

        wcurs.path_clone(&20);
        wcurs.path_clone(&30);

        wcurs.remove(&0);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_12() {
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

        let root = Node::new_branch(0, lbranch as *mut _, rbranch as *mut _);
        // let count = BV_CAPACITY + 2;
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        assert!(wcurs.verify());

        wcurs.path_clone(&0);
        wcurs.path_clone(&10);
        wcurs.path_clone(&20);

        wcurs.remove(&90);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_13() {
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

        let root = Node::new_branch(0, lbranch as *mut _, rbranch as *mut _);
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        assert!(wcurs.verify());

        for i in 0..BV_CAPACITY {
            let k = 100 + (10 * i);
            wcurs.path_clone(&k);
        }
        assert!(wcurs.verify());

        wcurs.remove(&10);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_14() {
        // Test leaf borrow left
        let lnode = create_leaf_node_full(10);
        let rnode = create_leaf_node(20);
        let root = Node::new_branch(0, lnode, rnode);
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        assert!(wcurs.verify());

        wcurs.remove(&20);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_15() {
        // Test leaf borrow right.
        let lnode = create_leaf_node(10) as *mut Node<usize, usize>;
        let rnode = create_leaf_node_full(20) as *mut Node<usize, usize>;
        let root = Node::new_branch(0, lnode, rnode);
        let sb = SuperBlock::new_test(1, root as *mut Node<usize, usize>);
        let mut wcurs = sb.create_writer();
        assert!(wcurs.verify());

        wcurs.remove(&10);

        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    fn tree_create_rand() -> (SuperBlock<usize, usize>, CursorRead<usize, usize>) {
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (1..(L_CAPACITY << 4)).collect();
        ins.shuffle(&mut rng);

        let mut sb = unsafe { SuperBlock::new() };
        let rdr = sb.create_reader();
        let mut wcurs = sb.create_writer();

        for v in ins.into_iter() {
            let r = wcurs.insert(v, v);
            assert!(r.is_none());
            assert!(wcurs.verify());
        }
        let rdr = sb.pre_commit(wcurs, &rdr);
        (sb, rdr)
    }

    #[test]
    fn test_bptree2_cursor_remove_stress_1() {
        // Insert ascending - we want to ensure the tree is a few levels deep
        // so we do this to a reasonable number.
        let (mut sb, rdr) = tree_create_rand();
        let mut wcurs = sb.create_writer();

        for v in 1..(L_CAPACITY << 4) {
            // println!("-- ITER v {}", v);
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
        }
        // println!("{:?}", wcurs);
        let rdr2 = sb.pre_commit(wcurs, &rdr);
        // On shutdown, check we dropped all as needed.
        std::mem::drop(rdr2);
        std::mem::drop(rdr);
        std::mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_stress_2() {
        // Insert descending
        let (mut sb, rdr) = tree_create_rand();
        let mut wcurs = sb.create_writer();

        for v in (1..(L_CAPACITY << 4)).rev() {
            // println!("ITER v {}", v);
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
        }
        let rdr2 = sb.pre_commit(wcurs, &rdr);
        std::mem::drop(rdr2);
        std::mem::drop(rdr);
        std::mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_stress_3() {
        // Insert random
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (1..(L_CAPACITY << 4)).collect();
        ins.shuffle(&mut rng);

        let (mut sb, rdr) = tree_create_rand();
        let mut wcurs = sb.create_writer();

        for v in ins.into_iter() {
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
        }
        let rdr2 = sb.pre_commit(wcurs, &rdr);
        std::mem::drop(rdr2);
        std::mem::drop(rdr);
        std::mem::drop(sb);
        assert_released();
    }

    // Add transaction-ised versions.
    #[test]
    fn test_bptree2_cursor_remove_stress_4() {
        // Insert ascending - we want to ensure the tree is a few levels deep
        // so we do this to a reasonable number.
        let (mut sb, mut rdr) = tree_create_rand();

        for v in 1..(L_CAPACITY << 4) {
            let mut wcurs = sb.create_writer();
            // println!("ITER v {}", v);
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
            rdr = sb.pre_commit(wcurs, &rdr);
        }
        std::mem::drop(rdr);
        std::mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_stress_5() {
        // Insert descending
        let (mut sb, mut rdr) = tree_create_rand();

        for v in (1..(L_CAPACITY << 4)).rev() {
            let mut wcurs = sb.create_writer();
            // println!("ITER v {}", v);
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
            rdr = sb.pre_commit(wcurs, &rdr);
        }
        std::mem::drop(rdr);
        std::mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_remove_stress_6() {
        // Insert random
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (1..(L_CAPACITY << 4)).collect();
        ins.shuffle(&mut rng);

        let (mut sb, mut rdr) = tree_create_rand();

        for v in ins.into_iter() {
            let mut wcurs = sb.create_writer();
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
            rdr = sb.pre_commit(wcurs, &rdr);
        }
        std::mem::drop(rdr);
        std::mem::drop(sb);
        assert_released();
    }

    /*
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_bptree2_cursor_remove_stress_7() {
        // Insert random
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (1..10240).collect();

        let node: *mut Leaf<usize, usize> = Node::new_leaf(0);
        let mut wcurs = CursorWrite::new_test(1, node as *mut _);
        wcurs.extend(ins.iter().map(|v| (*v, *v)));

        ins.shuffle(&mut rng);

        let compacts = 0;

        for v in ins.into_iter() {
            let r = wcurs.remove(&v);
            assert!(r == Some(v));
            assert!(wcurs.verify());
            // let (l, m) = wcurs.tree_density();
            // if l > 0 && (m / l) > 1 {
            //     compacts += 1;
            // }
        }
        println!("compacts {:?}", compacts);
    }
    */

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
    fn create_split_off_leaf(base: usize) -> *mut Node<usize, usize> {
        let l = Node::new_leaf(0);
        let lref = leaf_ref!(l, usize, usize);
        lref.insert_or_update(base + 1, base + 1);
        lref.insert_or_update(base + 2, base + 2);
        l as *mut _
    }

    fn create_split_off_branch(base: usize) -> *mut Node<usize, usize> {
        // This is a helper for create_split_off_tree to make the sub-branches based
        // on a base.
        let l1 = create_split_off_leaf(base);
        let l2 = create_split_off_leaf(base + 10);
        let l3 = create_split_off_leaf(base + 20);
        let l4 = create_split_off_leaf(base + 30);

        let branch = Node::new_branch(0, l1, l2);
        let nref = branch_ref!(branch, usize, usize);
        nref.add_node(l3);
        nref.add_node(l4);

        branch as *mut _
    }

    fn create_split_off_tree() -> *mut Node<usize, usize> {
        let b1 = create_split_off_branch(0);
        let b2 = create_split_off_branch(100);
        let b3 = create_split_off_branch(200);
        let b4 = create_split_off_branch(300);
        let root = Node::new_branch(0, b1, b2);
        let nref = branch_ref!(root, usize, usize);
        nref.add_node(b3);
        nref.add_node(b4);

        root as *mut _
    }

    #[test]
    fn test_bptree2_cursor_split_off_lt_01() {
        // Make a tree witth just a leaf
        // Do a split_off_lt.
        let node = create_leaf_node(0);
        let sb = SuperBlock::new_test(1, node);
        let mut wcurs = sb.create_writer();

        wcurs.split_off_lt(&5);

        // Remember, all the cases of the remove_lte are already tested on
        // leaf.
        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_split_off_lt_02() {
        // Make a tree witth just a leaf
        // Do a split_off_lt.
        let node = create_leaf_node_full(10);
        let sb = SuperBlock::new_test(1, node);
        let mut wcurs = sb.create_writer();

        wcurs.split_off_lt(&11);

        // Remember, all the cases of the remove_lte are already tested on
        // leaf.
        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_split_off_lt_03() {
        // Make a tree witth just a leaf
        // Do a split_off_lt.
        let node = create_leaf_node_full(10);
        let sb = SuperBlock::new_test(1, node);
        let mut wcurs = sb.create_writer();

        wcurs.path_clone(&11);
        wcurs.split_off_lt(&11);

        // Remember, all the cases of the remove_lte are already tested on
        // leaf.
        assert!(wcurs.verify());
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    fn run_split_off_test_clone(v: usize, _exp: usize) {
        // println!("RUNNING -> {:?}", v);
        let tree = create_split_off_tree();

        let sb = SuperBlock::new_test(1, tree);
        let mut wcurs = sb.create_writer();
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
        mem::drop(sb);
        assert_released();
    }

    fn run_split_off_test(v: usize, _exp: usize) {
        // println!("RUNNING -> {:?}", v);
        let tree = create_split_off_tree();
        // println!("START -> {:?}", tree);

        let sb = SuperBlock::new_test(1, tree);
        let mut wcurs = sb.create_writer();
        // 0 is min, and not present, will cause no change.
        wcurs.split_off_lt(&v);
        assert!(wcurs.verify());
        if v > 0 {
            assert!(!wcurs.contains_key(&(v - 1)));
        }
        // assert!(wcurs.len() == exp);

        // println!("{:?}", wcurs);
        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
    }

    #[test]
    fn test_bptree2_cursor_split_off_lt_clone_stress() {
        // Can't proceed as the "fake" tree we make is invalid.
        debug_assert!(L_CAPACITY >= 4);
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
    fn test_bptree2_cursor_split_off_lt_stress() {
        debug_assert!(L_CAPACITY >= 4);
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
    #[cfg_attr(miri, ignore)]
    fn test_bptree2_cursor_split_off_lt_random_stress() {
        let data: Vec<isize> = (0..1024).collect();

        for v in data.iter() {
            let node: *mut Leaf<isize, isize> = Node::new_leaf(0) as *mut _;
            let sb = SuperBlock::new_test(1, node as *mut Node<isize, isize>);
            let mut wcurs = sb.create_writer();
            wcurs.extend(data.iter().map(|v| (*v, *v)));

            if v > &0 {
                assert!(wcurs.contains_key(&(v - 1)));
            }

            wcurs.split_off_lt(v);
            assert!(!wcurs.contains_key(&(v - 1)));
            if v < &1024 {
                assert!(wcurs.contains_key(v));
            }
            assert!(wcurs.verify());
            let contents: Vec<_> = wcurs.k_iter().collect();
            assert!(contents[0] == v);
            assert!(contents.len() as isize == (1024 - v));
        }
    }

    #[test]
    fn test_bptree_cursor_double_extend() {
        let node: *mut Leaf<isize, isize> = Node::new_leaf(0) as *mut _;
        let sb = SuperBlock::new_test(1, node as *mut Node<isize, isize>);
        let mut wcurs = sb.create_writer();

        wcurs.extend([(0, 0), (1, 1), (2, 2), (3, 3)]);
        assert!(wcurs.len() == 4);
        assert!(wcurs.verify());

        wcurs.extend([(2, 2), (3, 3), (4, 4), (5, 5)]);
        assert!(wcurs.len() == 6);
        assert!(wcurs.verify());

        mem::drop(wcurs);
        mem::drop(sb);
        assert_released();
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
