use std::fmt::{self, Debug, Error};
use std::mem::MaybeUninit;
use std::ptr;
use std::slice;
use std::sync::Arc;

use super::constants::{BK_CAPACITY, BK_CAPACITY_MIN_N1, BV_CAPACITY, L_CAPACITY};
use super::leaf::Leaf;
use super::states::{BRInsertState, BRShrinkState, BRPruneState, BRTrimState};
use super::utils::*;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
use std::sync::Mutex;
#[cfg(test)]
use std::collections::BTreeSet;

#[cfg(test)]
thread_local!(static NODE_COUNTER: AtomicUsize = AtomicUsize::new(1));
#[cfg(test)]
thread_local!(static ALLOC_LIST: Mutex<BTreeSet<usize>> = Mutex::new(BTreeSet::new()));

#[cfg(test)]
fn alloc_nid() -> usize {
    let nid: usize = NODE_COUNTER.with(|nc| nc.fetch_add(1, Ordering::AcqRel));
    ALLOC_LIST.with(|llist| llist.lock().unwrap().insert(nid));
    nid
}

#[cfg(test)]
fn release_nid(nid: usize) {
    let r = ALLOC_LIST.with(|llist| llist.lock().unwrap().remove(&nid));
    assert!(r == true);
}


pub(crate) struct Branch<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    count: usize,
    key: [MaybeUninit<K>; BK_CAPACITY],
    node: [MaybeUninit<Arc<Node<K, V>>>; BV_CAPACITY],
}

#[derive(Debug)]
pub(crate) enum T<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    B(Branch<K, V>),
    L(Leaf<K, V>),
}

pub(crate) struct Node<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    #[cfg(test)]
    pub nid: usize,
    pub txid: usize,
    inner: T<K, V>,
}

pub(crate) type ABNode<K, V> = Arc<Node<K, V>>;

impl<K: Clone + Ord + Debug, V: Clone> Node<K, V> {
    pub(crate) fn new_leaf(txid: usize) -> Self {
        Node {
            #[cfg(test)]
            nid: alloc_nid(),
            txid: txid,
            inner: T::L(Leaf::new()),
        }
    }

    pub(crate) fn new_ableaf(txid: usize) -> ABNode<K, V> {
        Arc::new(Self::new_leaf(txid))
    }

    pub(crate) fn new_branch(txid: usize, l: ABNode<K, V>, r: ABNode<K, V>) -> ABNode<K, V> {
        Arc::new(Node {
            #[cfg(test)]
            nid: alloc_nid(),
            txid: txid,
            inner: T::B(Branch::new(l, r)),
        })
    }

    pub(crate) fn new_leaf_ins(txid: usize, k: K, v: V) -> ABNode<K, V> {
        let mut node = Arc::new(Node::new_leaf(txid));
        {
            let nmut = Arc::get_mut(&mut node).unwrap().as_mut_leaf();
            nmut.insert_or_update(k, v);
        }
        node
    }

    pub(crate) fn inner_clone(&self) -> T<K, V> {
        match &self.inner {
            T::L(leaf) => T::L(leaf.clone()),
            T::B(branch) => T::B(branch.clone()),
        }
    }

    pub(crate) fn req_clone(&self, txid: usize) -> ABNode<K, V> {
        debug_assert!(txid != self.txid);
        // Do we need to clone this node before we work on it?
        Arc::new(Node {
            #[cfg(test)]
            nid: alloc_nid(),
            txid: txid,
            inner: self.inner_clone(),
        })
    }

    #[cfg(test)]
    pub(crate) fn verify(&self) -> bool {
        match &self.inner {
            T::L(leaf) => leaf.verify(),
            T::B(branch) => branch.verify(),
        }
    }

    #[cfg(tesst)]
    pub(crate) fn len(&self) -> usize {
        match &self.inner {
            T::L(leaf) => leaf.len(),
            T::B(branch) => branch.len(),
        }
    }

    fn min(&self) -> &K {
        match &self.inner {
            T::L(leaf) => leaf.min(),
            T::B(branch) => branch.min(),
        }
    }

    #[cfg(test)]
    fn max(&self) -> &K {
        match &self.inner {
            T::L(leaf) => leaf.max(),
            T::B(branch) => branch.max(),
        }
    }

    pub(crate) fn get_ref(&self, k: &K) -> Option<&V> {
        match &self.inner {
            T::L(leaf) => leaf.get_ref(k),
            T::B(branch) => branch.get_ref(k),
        }
    }

    pub(crate) fn is_leaf(&self) -> bool {
        match &self.inner {
            T::L(_leaf) => true,
            T::B(_branch) => false,
        }
    }

    pub(crate) fn as_mut_leaf(&mut self) -> &mut Leaf<K, V> {
        match &mut self.inner {
            T::L(ref mut leaf) => leaf,
            T::B(_) => panic!(),
        }
    }

    pub(crate) fn as_leaf(&self) -> &Leaf<K, V> {
        match &self.inner {
            T::L(ref leaf) => leaf,
            T::B(_) => panic!(),
        }
    }

    pub(crate) fn as_mut_branch(&mut self) -> &mut Branch<K, V> {
        match &mut self.inner {
            T::L(_) => panic!(),
            T::B(ref mut branch) => branch,
        }
    }

    pub(crate) fn as_branch(&self) -> &Branch<K, V> {
        match &self.inner {
            T::L(_) => panic!(),
            T::B(ref branch) => branch,
        }
    }

    pub(crate) fn tree_density(&self) -> (usize, usize) {
        match &self.inner {
            T::L(leaf) => leaf.tree_density(),
            T::B(branch) => branch.tree_density(),
        }
    }

    pub(crate) fn leaf_count(&self) -> usize {
        match &self.inner {
            T::L(_leaf) => 1,
            T::B(branch) => branch.leaf_count(),
        }
    }
}

impl<K: Clone + Ord + Debug, V: Clone> Debug for Node<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), Error> {
        match &self.inner {
            T::L(leaf) => leaf.fmt(f),
            T::B(branch) => branch.fmt(f),
        }
    }
}

impl<K: Clone + Ord + Debug, V: Clone> Branch<K, V> {
    pub fn new(left: ABNode<K, V>, right: ABNode<K, V>) -> Self {
        let pivot: K = (*right.min()).clone();
        let mut new = Branch {
            count: 1,
            key: unsafe { MaybeUninit::uninit().assume_init() },
            node: unsafe { MaybeUninit::uninit().assume_init() },
        };
        unsafe {
            new.key[0].as_mut_ptr().write(pivot);
            new.node[0].as_mut_ptr().write(left);
            new.node[1].as_mut_ptr().write(right);
        }
        new
    }

    // Add a new pivot + node.
    pub(crate) fn add_node(&mut self, node: ABNode<K, V>) -> BRInsertState<K, V> {
        // Do we have space?
        if self.count == BK_CAPACITY {
            // if no space ->
            //    split and send two nodes back for new branch
            //
            // There are three possible states that this causes.
            // 1 * The inserted node is a low/middle value, causing max and max -1 to be returned.
            // 2 * The inserted node is the greater than all current values, causing l(max, node)
            //     to be returned.
            // 3 * The inserted node is between max - 1 and max, causing l(node, max) to be returned.
            //
            let kr: &K = node.min();
            // bst and find when min-key < key[idx]
            let r = {
                let (left, _) = self.key.split_at(self.count);
                let inited: &[K] =
                    unsafe { slice::from_raw_parts(left.as_ptr() as *const K, left.len()) };
                inited.binary_search(kr)
            };
            let ins_idx = r.unwrap_err();
            let res = match ins_idx {
                // Case 2
                BK_CAPACITY => {
                    // Greater than all current values, so we'll just return max and node.
                    let max = unsafe {
                        ptr::read(self.node.get_unchecked(BV_CAPACITY - 1)).assume_init()
                    };
                    // Drop the key between them.
                    let _kdrop =
                        unsafe { ptr::read(self.key.get_unchecked(BK_CAPACITY - 1)).assume_init() };
                    // Now setup the ret val
                    BRInsertState::Split(max, node)
                }
                // Case 3
                BK_CAPACITY_MIN_N1 => {
                    // Greater than all but max, so we return max and node in the correct order.
                    let max = unsafe {
                        ptr::read(self.node.get_unchecked(BV_CAPACITY - 1)).assume_init()
                    };
                    // Drop the key between them.
                    let _kdrop =
                        unsafe { ptr::read(self.key.get_unchecked(BK_CAPACITY - 1)).assume_init() };
                    // Now setup the ret val NOTICE compared to case 2 that we swap node and max?
                    BRInsertState::Split(node, max)
                }
                // Case 1
                ins_idx => {
                    // Get the max - 1 and max nodes out.
                    let maxn1 = unsafe {
                        ptr::read(self.node.get_unchecked(BV_CAPACITY - 2)).assume_init()
                    };
                    let max = unsafe {
                        ptr::read(self.node.get_unchecked(BV_CAPACITY - 1)).assume_init()
                    };
                    // Drop the key between them.
                    let _kdrop =
                        unsafe { ptr::read(self.key.get_unchecked(BK_CAPACITY - 1)).assume_init() };
                    // Drop the key before us that we are about to replace.
                    let _kdrop =
                        unsafe { ptr::read(self.key.get_unchecked(BK_CAPACITY - 2)).assume_init() };
                    // Add node and it's key to the correct location.
                    let k: K = kr.clone();
                    let leaf_ins_idx = ins_idx + 1;
                    unsafe {
                        slice_insert(&mut self.key, MaybeUninit::new(k), ins_idx);
                        slice_insert(&mut self.node, MaybeUninit::new(node), leaf_ins_idx);
                    }

                    BRInsertState::Split(maxn1, max)
                }
            };

            // Adjust the count, because we always remove at least 1 from the keys.
            self.count -= 1;
            res
        } else {
            // if space ->
            // Get the nodes min-key - we clone it because we'll certainly be inserting it!
            let k: K = node.min().clone();
            // bst and find when min-key < key[idx]
            let r = {
                let (left, _) = self.key.split_at(self.count);
                let inited: &[K] =
                    unsafe { slice::from_raw_parts(left.as_ptr() as *const K, left.len()) };
                inited.binary_search(&k)
            };
            // if r is ever found, I think this is a bug, because we should never be able to
            // add a node with an existing min.
            //
            //       [ 5 ]
            //        / \
            //    [0,]   [5,]
            //
            // So if we added here to [0, ], and it had to overflow to split, then everything
            // must be < 5. Why? Because to get to [0,] as your insert target, you must be < 5.
            // if we added to [5,] then a split must be greater than, or the insert would replace 5.
            //
            // if we consider
            //
            //       [ 5 ]
            //        / \
            //    [0,]   [7,]
            //
            // Now we insert 5, and 7, splits. 5 would remain in the tree and we'd split 7 to the right
            //
            // As a result, any "Ok(idx)" must represent a corruption of the tree.
            // debug_assert!(r.is_err());
            let ins_idx = r.unwrap_err();
            let leaf_ins_idx = ins_idx + 1;
            // So why do we only need to insert right? Because the left-most
            // leaf when it grows, it splits to the right. That importantly
            // means that we only need to insert to replace the min and it's
            // right leaf, or anything higher. As a result, we are always
            // targetting ins_idx and leaf_ins_idx = ins_idx + 1.
            //
            // We have a situation like:
            //
            //   [1, 3, 9, 18]
            //
            // and ins_idx is 2. IE:
            //
            //   [1, 3, 9, 18]
            //          ^-- k=6
            //
            // So this we need to shift those r-> and insert.
            //
            //   [1, 3, x, 9, 18]
            //          ^-- k=6
            //
            //   [1, 3, 6, 9, 18]
            //
            // Now we need to consider the leaves too:
            //
            //   [1, 3, 9, 18]
            //   | |  |  |   |
            //   v v  v  v   v
            //   0 1  3  9   18
            //
            // So that means we need to move leaf_ins_idx = (ins_idx + 1)
            // right also
            //
            //   [1, 3, x, 9, 18]
            //   | |  |  |  |   |
            //   v v  v  v  v   v
            //   0 1  3  x  9   18
            //           ^-- leaf for k=6 will go here.
            //
            // Now to talk about the right expand issue - lets say 0 conducted
            // a split, it returns the new right node - which would push
            // 3 to the right to insert a new right hand side as required. So we
            // really never need to consider the left most leaf to have to be
            // replaced in any conditions.
            //
            // Magic!
            unsafe {
                slice_insert(&mut self.key, MaybeUninit::new(k), ins_idx);
                slice_insert(&mut self.node, MaybeUninit::new(node), leaf_ins_idx);
            }
            // finally update the count
            self.count += 1;
            // Return that we are okay to go!
            BRInsertState::Ok
        }
    }

    pub(crate) fn shrink_decision(&mut self, ridx: usize) -> BRShrinkState {
        // Given two nodes, we need to decide what to do with them!
        //
        // Remember, this isn't happening in a vacuum. This is really a manipulation of
        // the following structure:
        //
        //      parent (self)
        //     /     \
        //   left    right
        //
        //   We also need to consider the following situation too:
        //
        //          root
        //         /    \
        //    lbranch   rbranch
        //    /    \    /     \
        //   l1    l2  r1     r2
        //
        //  Imagine we have exhausted r2, so we need to merge:
        //
        //          root
        //         /    \
        //    lbranch   rbranch
        //    /    \    /     \
        //   l1    l2  r1 <<-- r2
        //
        // This leaves us with a partial state of
        //
        //          root
        //         /    \
        //    lbranch   rbranch (invalid!)
        //    /    \    /
        //   l1    l2  r1
        //
        // This means rbranch issues a cloneshrink to root. clone shrink must contain the remainer
        // so that it can be reparented:
        //
        //          root
        //         /
        //    lbranch --
        //    /    \    \
        //   l1    l2   r1
        //
        // Now root has to shrink too.
        //
        //     root  --
        //    /    \    \
        //   l1    l2   r1
        //
        // So, we have to analyse the situation.
        //  * Have left or right been emptied? (how to handle when branches
        //  *
        //  * Is left or right belowe a reasonable threshold?
        //  * Does the opposite have capacity to remain valid?
        //  *
        //

        let (left, right) = self.get_mut_pair(ridx);

        if left.is_leaf() {
            debug_assert!(right.is_leaf());
            debug_assert!(Arc::strong_count(left) == 1);
            debug_assert!(Arc::strong_count(right) == 1);
            let lmut = Arc::get_mut(left).unwrap().as_mut_leaf();
            let rmut = Arc::get_mut(right).unwrap().as_mut_leaf();

            if lmut.len() == L_CAPACITY {
                lmut.take_from_l_to_r(rmut);
                self.rekey_by_idx(ridx);
                BRShrinkState::Balanced
            } else if rmut.len() == L_CAPACITY {
                lmut.take_from_r_to_l(rmut);
                self.rekey_by_idx(ridx);
                BRShrinkState::Balanced
            } else {
                // merge
                lmut.merge(rmut);
                // drop our references
                // mem::drop(lmut)
                // mem::drop(rmut)
                // remove the right node from parent
                let _ = self.remove_by_idx(ridx);
                // What is our capacity?
                if self.count == 0 {
                    // We now need to be merged across as we only contain a single
                    // value now.
                    BRShrinkState::Shrink
                } else {
                    BRShrinkState::Merge
                    // We are complete!
                }
            }
        } else {
            // right or left is now in a "corrupt" state with a single value that we need to relocate
            // to left - or we need to borrow from left and fix it!
            debug_assert!(!right.is_leaf());
            debug_assert!(Arc::strong_count(left) == 1);
            debug_assert!(Arc::strong_count(right) == 1);
            let lmut = Arc::get_mut(left).unwrap().as_mut_branch();
            let rmut = Arc::get_mut(right).unwrap().as_mut_branch();
            debug_assert!(rmut.len() == 0 || lmut.len() == 0);
            if lmut.len() == BK_CAPACITY {
                lmut.take_from_l_to_r(rmut);
                self.rekey_by_idx(ridx);
                BRShrinkState::Balanced
            } else if rmut.len() == BK_CAPACITY {
                lmut.take_from_r_to_l(rmut);
                self.rekey_by_idx(ridx);
                BRShrinkState::Balanced
            } else {
                // merge the right to tail of left.
                lmut.merge(rmut);
                // Reduce our count
                let _ = self.remove_by_idx(ridx);
                if self.count == 0 {
                    // We now need to be merged across as we also only contain a single
                    // value now.
                    BRShrinkState::Shrink
                } else {
                    BRShrinkState::Merge
                    // We are complete!
                }
            }
        }
    }

    pub(crate) fn prune_decision(&mut self, txid: usize, anode_idx: usize) -> Result<(), ()> {
        // So this means there are quite a few cases
        //
        //    [   k1, k2, k3, k4, k5, k6   ]
        //    [ v1, v2, v3, v4, v5, v6, v7 ]
        //
        // * anode_idx is maximum (self.count - 1), so we also now are empty.
        // * anything else, so we can shift down.

        // This is subtely different to prune, in that we handle branch rebalancing.
        println!("prune_decision -> {:?}", anode_idx);
        println!("{:?}", self);

        if anode_idx == self.count {
            /* We've hit a situation that shouldn't occur? */
            unimplemented!();
        }

        // First, clean up any excess we hold.
        self.prune(anode_idx)
            .expect("Invalid branch state!");

        // We now can assert that 0 is the node we are about to act upon.
        let ridx = self.clone_sibling_idx(txid, 0);
        let (left, right) = self.get_mut_pair(ridx);

        if left.is_leaf() {
            debug_assert!(right.is_leaf());
            debug_assert!(Arc::strong_count(left) == 1);
            debug_assert!(Arc::strong_count(right) == 1);
            let lmut = Arc::get_mut(left).unwrap().as_mut_leaf();
            let rmut = Arc::get_mut(right).unwrap().as_mut_leaf();

            if lmut.len() == L_CAPACITY {
                unreachable!("We should never be able to borrow from left!");
            } else if rmut.len() == L_CAPACITY {
                lmut.take_from_r_to_l(rmut);
                self.rekey_by_idx(ridx);
                Ok(())
            } else {
                // merge
                lmut.merge(rmut);
                // drop our references
                // mem::drop(lmut)
                // mem::drop(rmut)
                // remove the right node from parent
                let _ = self.remove_by_idx(ridx);
                // What is our capacity?
                if self.count == 0 {
                    // We now need to be merged across as we only contain a single
                    // value now.
                    Err(())
                } else {
                    Ok(())
                    // We are complete!
                }
            }
        } else {
            unimplemented!();
        }
    }

    pub(crate) fn prune(&mut self, idx: usize) -> Result<(), ()> {
        // idx is where we modified, so we know anything "less" must be less than k
        // so we can remove these.
        //
        // idx is the idx of the node we worked on, so that means if you have say:
        //
        //    [   k1, k2, k3, k4, k5, k6   ]
        //    [ v1, v2, v3, v4, v5, v6, v7 ]
        //                   ^--
        //
        //    so if we worked on idx 3, that means we need to remove 0 -> 2 in both
        // k/v sets.
        //
        debug_assert!(idx <= self.count);
        println!("slide and drop! c: {:?} i:{:?}", self.count, idx);

        // If the idx is 0, we do not need to act.
        if idx == 0 {
            return Ok(())
        } else if idx == self.count {
            // We must have to remove v6 and lower, so we need to be merge to a neighbor.
            unsafe {
                // We just drop all the keys.
                for kidx in 0..self.count {
                    ptr::drop_in_place(self.key[kidx].as_mut_ptr());
                    ptr::drop_in_place(self.node[kidx].as_mut_ptr());
                }
                // Move the last node to the bottom.
                ptr::swap(self.node[0].as_mut_ptr(), self.node[self.count].as_mut_ptr());
            }
            self.count = 0;
            // This node is now invalid, but has a single remaining pointer in position 0;
            unimplemented!();
            Err(())
        } else {
            // We still have enough to be a valid node, move on
            self.count = self.count - idx;
            unsafe {
                slice_slide_and_drop(&mut self.key, idx - 1, self.count);
                slice_slide_and_drop(&mut self.node, idx - 1, self.count + 1);
            }
            Ok(())
        }
    }

    pub(crate) fn clone_sibling_idx(&mut self, txid: usize, idx: usize) -> usize {
        // This clones and sets up for a subsequent
        // merge.
        if idx == 0 {
            // If we are zero, we clone our right sibling.
            // Clone idx 1
            self.clone_idx(txid, 1);
            // And increment the idx to 1
            1
        } else {
            // Otherwise we clone to the left
            self.clone_idx(txid, idx - 1);
            // And return as is.
            idx
        }
    }

    fn clone_idx(&mut self, txid: usize, idx: usize) {
        // Do we actually need to clone?
        let prev_ptr = self.get_idx(idx);
        // Do we really need to clone?
        if prev_ptr.txid == txid {
            // No, we already cloned this txn
            debug_assert!(Arc::strong_count(prev_ptr) == 1);
            return;
        }
        // Now do the clone
        let prev = unsafe { ptr::read(self.node.get_unchecked(idx)).assume_init() };
        let cnode = prev.req_clone(txid);
        debug_assert!(Arc::strong_count(&cnode) == 1);
        unsafe { ptr::write(self.node.get_unchecked_mut(idx), MaybeUninit::new(cnode)) };
        debug_assert!(
            {
                let r = unsafe { &mut *self.node[idx].as_mut_ptr() };
                Arc::strong_count(&r)
            } == 1
        )
    }

    // get a node containing some K - need to return our related idx.
    pub(crate) fn locate_node(&self, k: &K) -> usize {
        let r = {
            let (left, _) = self.key.split_at(self.count);
            let inited: &[K] =
                unsafe { slice::from_raw_parts(left.as_ptr() as *const K, left.len()) };
            inited.binary_search(&k)
        };

        // If the value is Ok(idx), then that means
        // we were located to the right node. This is because we
        // exactly hit and located on the key.
        //
        // If the value is Err(idx), then we have the exact index already.
        // as branches is one more.
        match r {
            Ok(v) => v + 1,
            Err(v) => v,
        }
    }

    pub(crate) fn extract_last_node(&mut self) -> ABNode<K, V> {
        debug_assert!(self.count == 0);
        unsafe {
            // We could just use get_unchecked instead.
            slice_remove(&mut self.node, 0).assume_init()
        }
    }

    pub(crate) fn merge(&mut self, right: &mut Self) {
        if right.len() == 0 {
            let node = right.extract_last_node();
            let k: K = node.min().clone();
            let ins_idx = self.count;
            let leaf_ins_idx = ins_idx + 1;
            unsafe {
                slice_insert(&mut self.key, MaybeUninit::new(k), ins_idx);
                slice_insert(&mut self.node, MaybeUninit::new(node), leaf_ins_idx);
            }
            self.count += 1;
        } else {
            debug_assert!(self.len() == 0);
            unsafe {
                // Move all the nodes from right.
                slice_merge(&mut self.node, 1, &mut right.node, right.count + 1);
                // Move the related keys.
                slice_merge(&mut self.key, 1, &mut right.key, right.count);
            }
            // Set our count correctly.
            self.count = right.count + 1;
            // Set rightt len to 0
            right.count = 0;
            // rekey the lowest pointer.
            unsafe {
                let nptr = &*self.node[1].as_ptr();
                let rekey = (*nptr.min()).clone();
                self.key[0].as_mut_ptr().write(rekey);
            }
            // done!
        }
    }

    pub(crate) fn take_from_l_to_r(&mut self, right: &mut Self) {
        debug_assert!(self.len() > right.len());
        // Starting index of where we move from. We work normally from a branch
        // with only zero (but the base) branch item, but we do the math anyway
        // to be sure incase we change later.
        //
        // So, self.len must be larger, so let's give a few examples here.
        //  4 = 7 - (7 + 0) / 2 (will move 4, 5, 6)
        //  3 = 6 - (6 + 0) / 2 (will move 3, 4, 5)
        //  3 = 5 - (5 + 0) / 2 (will move 3, 4)
        //  2 = 4 ....          (will move 2, 3)
        //
        let count = (self.len() + right.len()) / 2;
        let start_idx = self.len() - count;
        // Move the remaining element from r to the correct location.

        unsafe {
            ptr::swap(
                right.node.get_unchecked_mut(0),
                right.node.get_unchecked_mut(count),
            )
        }

        // Move our values.
        unsafe {
            slice_move(&mut right.node, 0, &mut self.node, start_idx + 1, count);
        }
        // Remove the keys from left.
        //
        // If we have:
        //    [   k1, k2, k3, k4, k5, k6   ]
        //    [ v1, v2, v3, v4, v5, v6, v7 ] -> [ v8, ------- ]
        //
        // We would move 3 now to:
        //
        //    [   k1, k2, k3, k4, k5, k6   ]    [   --, --, --, --, ...
        //    [ v1, v2, v3, v4, --, --, -- ] -> [ v5, v6, v7, v8, --, ...
        //
        // So we need to remove the corresponding keys. so that we get.
        //
        //    [   k1, k2, k3, --, --, --   ]    [   --, --, --, --, ...
        //    [ v1, v2, v3, v4, --, --, -- ] -> [ v5, v6, v7, v8, --, ...
        //
        // This means it's start_idx - 1 up to BK cap

        for kidx in (start_idx - 1)..BK_CAPACITY {
            let _pk = unsafe { ptr::read(self.key.get_unchecked(kidx)).assume_init() };
            // They are dropped now.
        }
        // Adjust both counts - we do this before rekey to ensure that the safety
        // checks hold in debugging.
        right.count = count;
        self.count = start_idx;
        // Rekey right
        for kidx in 1..(count + 1) {
            right.rekey_by_idx(kidx);
        }
        // Done!
    }

    pub(crate) fn take_from_r_to_l(&mut self, right: &mut Self) {
        debug_assert!(right.len() >= self.len());

        let count = (self.len() + right.len()) / 2;
        let start_idx = right.len() - count;

        // We move count from right to left.
        unsafe {
            slice_move(&mut self.node, 1, &mut right.node, 0, count);
        }

        // Pop the excess keys in right
        // So say we had 6/7 in right, and 0/1 in left.
        //
        // We have a start_idx of 4, and count of 3.
        //
        // We moved 3 values from right, leaving 4. That means we need to remove
        // keys 0, 1, 2. The remaining keys are moved down.
        for kidx in 0..count {
            let _pk = unsafe { ptr::read(right.key.get_unchecked(kidx)).assume_init() };
            // They are dropped now.
        }

        // move keys down in right
        unsafe {
            ptr::copy(
                right.key.as_ptr().add(count),
                right.key.as_mut_ptr(),
                start_idx,
            );
        }
        // move nodes down in right
        unsafe {
            ptr::copy(
                right.node.as_ptr().add(count),
                right.node.as_mut_ptr(),
                start_idx + 1,
            );
        }

        // update counts
        right.count = start_idx;
        self.count = count;
        // Rekey left
        for kidx in 1..(count + 1) {
            self.rekey_by_idx(kidx);
        }
        // Done!
    }

    // remove a node by idx.
    pub(crate) fn remove_by_idx(&mut self, idx: usize) -> ABNode<K, V> {
        debug_assert!(idx <= self.count);
        debug_assert!(idx > 0);
        // remove by idx.
        let _pk = unsafe { slice_remove(&mut self.key, idx - 1).assume_init() };
        let pn = unsafe { slice_remove(&mut self.node, idx).assume_init() };
        self.count -= 1;
        pn
    }

    pub(crate) fn trim_lt_key(&mut self, k: &K) -> BRTrimState<K, V> {
        // The possible states of a branch are
        //
        // [  0,  4,  8,  12  ]
        // [n1, n2, n3, n4, n5]
        //
        let r = {
            let (left, _) = self.key.split_at(self.count);
            let inited: &[K] =
                unsafe { slice::from_raw_parts(left.as_ptr() as *const K, left.len()) };
            inited.binary_search(&k)
        };

        match r {
            Ok(idx) => {
                if idx == self.count {
                    // * The key exactly matches the max value, so we can remove all lower
                    //   and promote the maximum
                    unimplemented!();
                } else {
                    // * A key matches exactly a value. IE k is 4. This means we cane remove
                    //   n1 and n2 because we know 4 must be in n3 as the min.
                    unimplemented!();
                }
            }
            Err(idx) => {
                if idx == 0 {
                    // * The key is less than min. IE it wants to remove "inside" n1. we simply
                    //   return, as no trimming is possible.
                    BRTrimState::Complete
                } else if idx > self.count {
                    // * The value is greater than all. This means we have to remove everything BUT
                    //   the maximal node.
                    unimplemented!();
                } else {
                    // * A key is between two values. We can remove everything less, but not
                    //   the assocated. For example, remove 6 would cause n1, n2 to be removed, but
                    //   the prune/walk will have to examine n3 to know about further changes.
                    unimplemented!();
                }
            }
        }



        /*
        if idx == self.count {
            // We would remove everything.
            BRPruneState::Prune
        } else if idx == self.count - 1 {
            // Remove all but one, IE we now shrink
            // drop self.count by one.
            // drop the associated key.
            unimplemented!();
            // let node = ...?
            // BRPruneState::Shrink(node)
        } else if idx == 0 {
            // Nothing to do.
            BRPruneState::Ok
        } else {
            // Okay, actually do some shuffle, but our branch will survive.
            BRPruneState::Ok
        }
        */
    }

    pub(crate) fn replace_by_idx(&mut self, idx: usize, mut node: ABNode<K, V>) -> () {
        debug_assert!(idx <= self.count);
        // We have to swap the value at idx with this node, then ensure that the
        // prev is droped.
        unsafe {
            ptr::swap(self.node[idx].as_mut_ptr(), &mut node as *mut ABNode<K, V>);
        }
    }

    pub(crate) fn rekey_by_idx(&mut self, idx: usize) {
        debug_assert!(idx <= self.count);
        debug_assert!(idx > 0);
        // For the node listed, rekey it.
        let nref = self.get_idx(idx);
        let nkey = (*nref.min()).clone();
        unsafe {
            self.key[idx - 1].as_mut_ptr().write(nkey);
        }
    }

    pub(crate) fn get_mut_idx(&mut self, idx: usize) -> &mut ABNode<K, V> {
        debug_assert!(idx <= self.count);
        let v = unsafe { &mut *self.node[idx].as_mut_ptr() };
        // We can't assert that our child's count is 1 here, because we clone down as we
        // go now.
        // debug_assert!(Arc::strong_count(&v) == 1);
        v
    }

    pub(crate) fn get_mut_pair(&mut self, idx: usize) -> (&mut ABNode<K, V>, &mut ABNode<K, V>) {
        debug_assert!(idx <= self.count);
        let l = unsafe { &mut *self.node[idx - 1].as_mut_ptr() };
        let r = unsafe { &mut *self.node[idx].as_mut_ptr() };
        debug_assert!(Arc::ptr_eq(&l, &r) == false);
        debug_assert!(Arc::strong_count(&l) == 1);
        debug_assert!(Arc::strong_count(&r) == 1);
        (l, r)
    }

    pub(crate) fn get_idx(&self, idx: usize) -> &ABNode<K, V> {
        debug_assert!(idx <= self.count);
        unsafe { &*self.node[idx].as_ptr() }
    }

    pub(crate) fn get_idx_checked(&self, idx: usize) -> Option<&ABNode<K, V>> {
        if idx <= self.count {
            Some(unsafe { &*self.node[idx].as_ptr() })
        } else {
            None
        }
    }

    pub(crate) fn min(&self) -> &K {
        unsafe { (*self.node[0].as_ptr()).min() }
    }

    #[cfg(test)]
    pub(crate) fn max(&self) -> &K {
        unsafe { (*self.node[self.count].as_ptr()).max() }
    }

    pub(crate) fn len(&self) -> usize {
        self.count
    }

    pub(crate) fn get_ref(&self, k: &K) -> Option<&V> {
        // Which node should hold this value?
        let idx = self.locate_node(k);
        unsafe { (*self.node[idx].as_ptr()).get_ref(k) }
    }

    pub(crate) fn tree_density(&self) -> (usize, usize) {
        let mut lcount = 0; // leaf populated
        let mut mcount = 0; // leaf max possible
        for idx in 0..(self.count + 1) {
            let (l, m) = unsafe { (*self.node[idx].as_ptr()).tree_density() };
            lcount += l;
            mcount += m;
        }
        (lcount, mcount)
    }

    pub(crate) fn leaf_count(&self) -> usize {
        let mut lcount = 0;
        for idx in 0..(self.count + 1) {
            lcount += unsafe { (*self.node[idx].as_ptr()).leaf_count() };
        }
        lcount
    }

    #[cfg(test)]
    fn check_sorted(&self) -> bool {
        // check the pivots are sorted.
        if self.count == 0 {
            panic!();
            false
        } else {
            let mut lk: &K = unsafe { &*self.key[0].as_ptr() };
            for work_idx in 1..self.count {
                let rk: &K = unsafe { &*self.key[work_idx].as_ptr() };
                if lk >= rk {
                    panic!();
                    return false;
                }
                lk = rk;
            }
            // println!("Passed sorting");
            true
        }
    }

    #[cfg(test)]
    fn check_descendents_valid(&self) -> bool {
        for work_idx in 0..self.count {
            // get left max and right min
            let lnode = unsafe { &*self.node[work_idx].as_ptr() };
            let rnode = unsafe { &*self.node[work_idx + 1].as_ptr() };

            let pkey = unsafe { &*self.key[work_idx].as_ptr() };
            let lkey = lnode.max();
            let rkey = rnode.min();
            if lkey >= pkey || pkey > rkey {
                println!("++++++");
                println!("out of order key found {}", work_idx);
                println!("{:?}", lnode);
                println!("{:?}", rnode);
                println!("{:?}", self);
                panic!();
                return false;
            }
        }
        // println!("Passed descendants");
        true
    }

    #[cfg(test)]
    fn verify_children(&self) -> bool {
        // For each child node call verify on it.
        for work_idx in 0..self.count {
            let node = unsafe { &*self.node[work_idx].as_ptr() };
            if !node.verify() {
                println!("Failed children");
                panic!();
                return false;
            }
        }
        // println!("Passed children");
        true
    }

    #[cfg(test)]
    pub(crate) fn verify(&self) -> bool {
        self.check_sorted() && self.check_descendents_valid() && self.verify_children()
    }
}

impl<K: Clone + Ord + Debug, V: Clone> Clone for Branch<K, V> {
    fn clone(&self) -> Self {
        let mut nkey: [MaybeUninit<K>; BK_CAPACITY] =
            unsafe { MaybeUninit::uninit().assume_init() };
        let mut nnode: [MaybeUninit<Arc<Node<K, V>>>; BV_CAPACITY] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for idx in 0..self.count {
            unsafe {
                let lkey = (*self.key[idx].as_ptr()).clone();
                nkey[idx].as_mut_ptr().write(lkey);
            }
            unsafe {
                let lnode = (*self.node[idx].as_ptr()).clone();
                nnode[idx].as_mut_ptr().write(lnode);
            }
        }
        // clone the last node.
        unsafe {
            let lnode = (*self.node[self.count].as_ptr()).clone();
            nnode[self.count].as_mut_ptr().write(lnode);
        }

        Branch {
            count: self.count,
            key: nkey,
            node: nnode,
        }
    }
}

impl<K: Clone + Ord + Debug, V: Clone> Debug for Branch<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), Error> {
        write!(f, "Branch -> {}\n", self.count)?;
        write!(f, "  \\-> [  |")?;
        for idx in 0..self.count {
            write!(f, "{:^6?}|", unsafe { &*self.key[idx].as_ptr() })?;
        }
        #[cfg(test)]
        {
            write!(f, " ]\n")?;
            write!(f, " nids [{:^6?}", unsafe { (*self.node[0].as_ptr()).nid })?;
            for idx in 0..self.count {
                write!(f, "{:^7?}", unsafe { (*self.node[idx + 1].as_ptr()).nid })?;
            }
            write!(f, " ]\n")?;
            write!(f, " mins [{:^6?}", unsafe {
                (*self.node[0].as_ptr()).min()
            })?;
            for idx in 0..self.count {
                write!(f, "{:^7?}", unsafe { (*self.node[idx + 1].as_ptr()).min() })?;
            }
        }
        write!(f, " ]\n")
    }
}

impl<K: Clone + Ord + Debug, V: Clone> Drop for Branch<K, V> {
    fn drop(&mut self) {
        // Due to the use of maybe uninit we have to drop any contained values.
        if self.count > 0 {
            for idx in 0..self.count {
                unsafe {
                    ptr::drop_in_place(self.key[idx].as_mut_ptr());
                }
            }
            // Remember, a branch ALWAYS has two nodes per key, which means
            // it's N+1,so we have to increase this to ensure we drop them
            // all.
            for idx in 0..(self.count + 1) {
                unsafe {
                    ptr::drop_in_place(self.node[idx].as_mut_ptr());
                }
            }
        }
    }
}

#[cfg(test)]
impl<K: Clone + Ord + Debug, V: Clone> Drop for Node<K, V> {
    fn drop(&mut self) {
        if cfg!(test) {
            println!("fn drop -> {:?}", self.nid);
            release_nid(self.nid)
        }
    }
}

#[cfg(test)]
pub(crate) fn check_drop_count() {
    let size = ALLOC_LIST.with(|llist| {
        let guard = llist.lock().unwrap();
        let size = guard.len();
        if size > 0 {
            println!("failed to drop nodes!");
            println!("{:?}", guard);
        }
        size
    });
    assert!(size == 0);
}

#[cfg(test)]
mod tests {
    use super::super::constants::BV_CAPACITY;
    use super::super::states::BRInsertState;
    use super::{ABNode, Branch, Node};
    use std::sync::Arc;

    #[test]
    fn test_bptree_node_new() {
        let mut left = Arc::new(Node::new_leaf(0));
        let mut right = Arc::new(Node::new_leaf(0));

        // add some k, vs to each.
        {
            let lmut = Arc::get_mut(&mut left).unwrap().as_mut_leaf();
            lmut.insert_or_update(0, 0);
            lmut.insert_or_update(1, 1);
        }
        {
            let rmut = Arc::get_mut(&mut right).unwrap().as_mut_leaf();
            rmut.insert_or_update(5, 5);
            rmut.insert_or_update(6, 6);
        }

        let branch = Branch::new(left, right);

        assert!(branch.verify());
    }

    fn create_branch_one_three() -> Branch<usize, usize> {
        let mut left = Arc::new(Node::new_leaf(0));
        let mut right = Arc::new(Node::new_leaf(0));
        {
            let lmut = Arc::get_mut(&mut left).unwrap().as_mut_leaf();
            lmut.insert_or_update(1, 1);
            let rmut = Arc::get_mut(&mut right).unwrap().as_mut_leaf();
            rmut.insert_or_update(3, 3);
        }
        Branch::new(left, right)
    }

    fn create_branch_one_three_max() -> Branch<usize, usize> {
        let mut branch = create_branch_one_three();
        // We - 3 here because we have two nodes from before
        // and we need 1 to be 100 so we know the max.
        assert!(BV_CAPACITY >= 3);
        for idx in 0..(BV_CAPACITY - 3) {
            let node = create_node(idx + 10);
            branch.add_node(node);
        }
        let node = create_node(100);
        branch.add_node(node);
        branch
    }

    fn create_node(v: usize) -> ABNode<usize, usize> {
        let mut node = Arc::new(Node::new_leaf(0));
        {
            let nmut = Arc::get_mut(&mut node).unwrap().as_mut_leaf();
            nmut.insert_or_update(v, v);
        }
        node
    }

    /*
    #[test]
    fn test_bptree_node_add_min() {
        // Add a new node which is a new minimum. In theory this situation
        // should *never* occur as we always split *right*. But we handle it
        // for completeness sake.
        let node = create_node(0);
        let mut branch = create_branch_one_three();
        let r = branch.add_node(node);
        match r {
            BRInsertState::Ok => {}
            _ => panic!(),
        };
        // ALERT ALERT ALERT WARNING ATTENTION DANGER WILL ROBINSON
        // THIS IS ASSERTING THAT THE NODE IS NOW CORRUPTED AS INSERT MIN
        // SHOULD NEVER OCCUR!!!
        assert!(branch.verify() == false);
    }
    */

    #[test]
    fn test_bptree_node_add_middle() {
        // Add a new node in "the middle" of existing nodes.
        let node = create_node(2);
        let mut branch = create_branch_one_three();
        let r = branch.add_node(node);
        match r {
            BRInsertState::Ok => {}
            _ => panic!(),
        };
        assert!(branch.verify());
    }

    #[test]
    fn test_bptree_node_add_max() {
        // Add a new max node.
        let node = create_node(4);
        let mut branch = create_branch_one_three();
        let r = branch.add_node(node);
        match r {
            BRInsertState::Ok => {}
            _ => panic!(),
        };
        assert!(branch.verify());
    }

    #[test]
    fn test_bptree_node_add_split_min() {
        // We don't test this, it should never occur.
        //
        assert!(true);
    }

    #[test]
    fn test_bptree_node_add_split_middle() {
        // Add a new middle node that wuld cause a split
        let node = create_node(4);
        let mut branch = create_branch_one_three_max();
        println!("test ins");
        let r = branch.add_node(node);
        match r {
            BRInsertState::Split(_, _) => {}
            _ => panic!(),
        };
        assert!(branch.verify());
    }

    #[test]
    fn test_bptree_node_add_split_max() {
        // Add a new max node that would cause this branch to split.
        let node = create_node(101);
        let mut branch = create_branch_one_three_max();
        println!("test ins");
        let r = branch.add_node(node);
        match r {
            BRInsertState::Split(_, r) => {
                assert!(r.min() == &101);
            }
            _ => panic!(),
        };
        assert!(branch.verify());
    }

    #[test]
    fn test_bptree_node_add_split_n1max() {
        // Add a value that is one before max that would trigger a split.
        let node = create_node(99);
        let mut branch = create_branch_one_three_max();
        println!("test ins");
        let r = branch.add_node(node);
        match r {
            BRInsertState::Split(l, _) => {
                assert!(l.min() == &99);
            }
            _ => panic!(),
        };
        assert!(branch.verify());
    }

    #[test]
    fn test_bptree_node_prune_states() {
        // Will the locate_node being eq tto go right affect?
        // remove none (caller needs to descend down left most)
        // purge all -> I think this becomes a promote of the rightmost node
        // remove middle (exact), enough rem
        // remove middle (non-exact), enough rem
        // remove middle (exact), one rem
        // remove middle (non-exact), one rem
    }
}
