use std::borrow::Borrow;
use std::fmt::{self, Debug, Error};
use std::mem::MaybeUninit;
use std::ptr;
use std::slice;

use super::constants::{L_CAPACITY, L_MAX_IDX};
use super::node::Node;
#[cfg(test)]
use super::states::BLPruneState;
use super::states::{BLInsertState, BLRemoveState};
use super::utils::*;

pub(crate) struct Leaf<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    count: usize,
    key: [MaybeUninit<K>; L_CAPACITY],
    value: [MaybeUninit<V>; L_CAPACITY],
}

impl<K: Clone + Ord + Debug, V: Clone> Leaf<K, V> {
    pub fn new() -> Self {
        Leaf {
            count: 0,
            key: unsafe { MaybeUninit::uninit().assume_init() },
            value: unsafe { MaybeUninit::uninit().assume_init() },
        }
    }

    pub(crate) fn insert_or_update(&mut self, txid: usize, k: K, v: V) -> BLInsertState<K, V> {
        // Update the node, and split if required.
        // There are three possible paths
        let r = {
            let (left, _) = self.key.split_at(self.count);
            let inited: &[K] =
                unsafe { slice::from_raw_parts(left.as_ptr() as *const K, left.len()) };
            // inited.binary_search(&k)
            slice_search_linear(inited, &k)
        };
        match r {
            Ok(idx) => {
                // * some values (but not full) exist, and we need to update the value that does exist
                let prev = unsafe { self.value[idx].as_mut_ptr().replace(v) };
                // v now contains the original value, return it!
                return BLInsertState::Ok(Some(prev));
            }
            Err(idx) => {
                if self.count == L_CAPACITY {
                    // * The node is full, so we must indicate as such.
                    if idx >= self.count {
                        // The requested insert is larger than our max key.
                        let rnode = Node::new_leaf_ins(txid, k, v);
                        BLInsertState::Split(rnode)
                    } else if idx == 0 {
                        // If this value is smaller than all else, we want to handle this special
                        // case. The reason is we don't want to heavily fragment in the reverse
                        // insert case, so we support spliting left, but it requires a more complex
                        // parent operation to support.
                        //
                        // This does require a rekey to the parent however, which adds extra states
                        // etc.
                        let lnode = Node::new_leaf_ins(txid, k, v);
                        BLInsertState::RevSplit(lnode)
                    } else {
                        // The requested insert in within our range, return current
                        // max.
                        let pk = unsafe { slice_remove(&mut self.key, L_MAX_IDX).assume_init() };
                        let pv = unsafe { slice_remove(&mut self.value, L_MAX_IDX).assume_init() };
                        unsafe {
                            slice_insert(&mut self.key, MaybeUninit::new(k), idx);
                            slice_insert(&mut self.value, MaybeUninit::new(v), idx);
                        }
                        let rnode = Node::new_leaf_ins(txid, pk, pv);
                        BLInsertState::Split(rnode)
                    }
                } else {
                    // We have space, insert at the correct location after shifting.
                    unsafe {
                        slice_insert(&mut self.key, MaybeUninit::new(k), idx);
                        slice_insert(&mut self.value, MaybeUninit::new(v), idx);
                    }
                    self.count += 1;
                    BLInsertState::Ok(None)
                }
            }
        }
    }

    pub(crate) fn remove(&mut self, k: &K) -> BLRemoveState<V> {
        // We already were empty - should never occur, but let's be paranoid.
        if self.count == 0 {
            return BLRemoveState::Shrink(None);
        }

        // Find the value
        // * if not found, return Ok(None).
        match self.get_idx(k) {
            // Count must be greater than 0, and we didn't find it, so return ok.
            None => BLRemoveState::Ok(None),
            // We found it, let's shuffle stuff.
            Some(idx) => {
                // Get the k/v out. These slots will be over-written, and pk/pv
                // are now subject to drop handling.
                let _pk = unsafe { slice_remove(&mut self.key, idx).assume_init() };
                let pv = unsafe { slice_remove(&mut self.value, idx).assume_init() };
                // drop our count, as we have removed a k/v
                self.count -= 1;
                // Based on the count, indicate if we should be shrunk
                if self.count == 0 {
                    BLRemoveState::Shrink(Some(pv))
                } else {
                    BLRemoveState::Ok(Some(pv))
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn remove_lt(&mut self, k: &K) -> BLPruneState {
        // println!("Start remove_lt on {:?}", self);
        // Remove everything less than or equal to a value.
        if self.count == 0 {
            return BLPruneState::Prune;
        }

        // Find the pivot point
        let r = {
            let (left, _) = self.key.split_at(self.count);
            let inited: &[K] =
                unsafe { slice::from_raw_parts(left.as_ptr() as *const K, left.len()) };
            // inited.binary_search(&k)
            slice_search_linear(inited, k)
        };

        match r {
            Err(0) => {
                // No action, nothing to remove.
                BLPruneState::Ok
            }
            // Is our removal including an item?
            Err(idx) => {
                if idx >= self.count {
                    // Remove everything.
                    for didx in 0..self.count {
                        unsafe {
                            ptr::drop_in_place(self.key[didx].as_mut_ptr());
                            ptr::drop_in_place(self.value[didx].as_mut_ptr());
                        }
                    }
                    // Set the count to zero.
                    self.count = 0;
                    BLPruneState::Prune
                } else {
                    unsafe {
                        slice_slide_and_drop(&mut self.key, idx, self.count - idx);
                        slice_slide_and_drop(&mut self.value, idx, self.count - idx);
                    }
                    // Only remove self.count - idx
                    self.count = self.count - idx;
                    BLPruneState::Ok
                }
            }
            Ok(idx) => {
                if idx == 0 {
                    // Do nothing, this is min
                    BLPruneState::Ok
                } else {
                    // Remove idx - 1?
                    let upto = idx - 1;

                    // Split and move
                    unsafe {
                        slice_slide_and_drop(&mut self.key, upto, self.count - idx);
                        slice_slide_and_drop(&mut self.value, upto, self.count - idx);
                    }

                    self.count = self.count - idx;
                    BLPruneState::Ok
                }
            }
        }
    }

    pub(crate) fn merge(&mut self, right: &mut Self) {
        unsafe {
            slice_merge(&mut self.key, self.count, &mut right.key, right.count);
            slice_merge(&mut self.value, self.count, &mut right.value, right.count);
        }
        self.count = self.count + right.count;
        right.count = 0;
    }

    pub(crate) fn take_from_l_to_r(&mut self, right: &mut Self) {
        debug_assert!(right.len() == 0);
        let count = self.len() / 2;
        let start_idx = self.len() - count;

        //move key and values
        unsafe {
            slice_move(&mut right.key, 0, &mut self.key, start_idx, count);
            slice_move(&mut right.value, 0, &mut self.value, start_idx, count);
        }

        // update the counts
        self.count = start_idx;
        right.count = count;
    }

    pub(crate) fn take_from_r_to_l(&mut self, right: &mut Self) {
        debug_assert!(self.len() == 0);
        let count = right.len() / 2;
        let start_idx = right.len() - count;

        // Move values from right to left.
        unsafe {
            slice_move(&mut self.key, 0, &mut right.key, 0, count);
            slice_move(&mut self.value, 0, &mut right.value, 0, count);
        }
        // Shift the values in right down.
        unsafe {
            ptr::copy(
                right.key.as_ptr().add(count),
                right.key.as_mut_ptr(),
                start_idx,
            );
            ptr::copy(
                right.value.as_ptr().add(count),
                right.value.as_mut_ptr(),
                start_idx,
            );
        }

        // Fix the counts.
        self.count = count;
        right.count = start_idx;
    }

    fn max_idx(&self) -> usize {
        debug_assert!(self.count > 0);
        self.count - 1
    }

    pub(crate) fn min(&self) -> &K {
        unsafe { &*self.key[0].as_ptr() }
    }

    pub(crate) fn max(&self) -> &K {
        let idx = self.max_idx();
        unsafe { &*self.key[idx].as_ptr() }
    }

    pub(crate) fn get_idx<Q: ?Sized>(&self, k: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        match {
            let (left, _) = self.key.split_at(self.count);
            let inited: &[K] =
                unsafe { slice::from_raw_parts(left.as_ptr() as *const K, left.len()) };

            slice_search_linear(inited, k)
            // inited.binary_search(k)
        } {
            Ok(idx) => Some(idx),
            Err(_) => None,
        }
    }

    pub(crate) fn get_ref<Q: ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        self.get_idx(k)
            .map(|idx| unsafe { &*self.value[idx].as_ptr() })
    }

    /*
    pub(crate) fn get_mut_ref_idx(&mut self, idx: usize) -> &mut V {
        unsafe { &mut *self.value[idx].as_mut_ptr() }
    }
    */

    pub(crate) fn get_mut_ref(&mut self, k: &K) -> Option<&mut V> {
        self.get_idx(k)
            .map(|idx| unsafe { &mut *self.value[idx].as_mut_ptr() })
    }

    pub(crate) fn get_kv_idx_checked(&self, idx: usize) -> Option<(&K, &V)> {
        if idx < self.count {
            Some((unsafe { &*self.key[idx].as_ptr() }, unsafe {
                &*self.value[idx].as_ptr()
            }))
        } else {
            None
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.count
    }

    #[cfg(test)]
    fn check_sorted(&self) -> bool {
        if self.count == 0 {
            true
        } else {
            let mut lk: &K = unsafe { &*self.key[0].as_ptr() };
            for work_idx in 1..self.count {
                let rk: &K = unsafe { &*self.key[work_idx].as_ptr() };
                if lk >= rk {
                    println!("{:?}", self);
                    debug_assert!(false);
                    return false;
                }
                lk = rk;
            }
            // println!("Leaf passed sorting");
            true
        }
    }

    #[cfg(test)]
    pub(crate) fn verify(&self) -> bool {
        self.check_sorted()
    }

    pub(crate) fn tree_density(&self) -> (usize, usize) {
        (self.count, L_CAPACITY)
    }
}

impl<K: Ord + Clone, V: Clone> Clone for Leaf<K, V> {
    fn clone(&self) -> Self {
        let mut nkey: [MaybeUninit<K>; L_CAPACITY] = unsafe { MaybeUninit::uninit().assume_init() };
        let mut nvalue: [MaybeUninit<V>; L_CAPACITY] =
            unsafe { MaybeUninit::uninit().assume_init() };

        for idx in 0..self.count {
            // Clone all the keys.
            unsafe {
                let lkey = (*self.key[idx].as_ptr()).clone();
                nkey[idx].as_mut_ptr().write(lkey);
            }

            // Clone the values.
            unsafe {
                let lvalue = (*self.value[idx].as_ptr()).clone();
                nvalue[idx].as_mut_ptr().write(lvalue);
            }
        }

        Leaf {
            count: self.count,
            key: nkey,
            value: nvalue,
        }
    }
}

impl<K: Ord + Clone, V: Clone> Drop for Leaf<K, V> {
    fn drop(&mut self) {
        // Due to the use of maybe uninit we have to drop any contained values.
        for idx in 0..self.count {
            unsafe {
                ptr::drop_in_place(self.key[idx].as_mut_ptr());
                ptr::drop_in_place(self.value[idx].as_mut_ptr());
            }
        }
    }
}

impl<K: Ord + Clone + Debug, V: Clone> Debug for Leaf<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), Error> {
        write!(f, "Leaf -> {}", self.count)?;
        write!(f, "  \\-> [ ")?;
        for idx in 0..self.count {
            write!(f, "{:?}, ", unsafe { &*self.key[idx].as_ptr() })?;
        }
        write!(f, " ]")
    }
}

#[cfg(test)]
mod tests {
    use super::super::constants::L_CAPACITY;
    use super::super::states::{BLInsertState, BLRemoveState};
    use super::Leaf;

    // test insert in order
    #[test]
    fn test_bptree_leaf_insert_order() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();

        for kv in 0..L_CAPACITY {
            let r = leaf.insert_or_update(0, kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
            let gr = leaf.get_ref(&kv);
            assert!(gr == Some(&kv));
        }
        assert!(leaf.verify());
    }

    // test insert and update to over-write in order.
    #[test]
    fn test_bptree_leaf_update_order() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();

        for kv in 0..L_CAPACITY {
            let r = leaf.insert_or_update(0, kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
            let gr = leaf.get_ref(&kv);
            assert!(gr == Some(&kv));
        }

        for kv in 0..L_CAPACITY {
            let r = leaf.insert_or_update(0, kv, kv + 1);
            match r {
                // Check for some kv, that was the former value.
                BLInsertState::Ok(Some(_kv)) => {}
                _ => panic!(),
            }
            let gr = leaf.get_ref(&kv);
            // Check the new value is incremented.
            assert!(gr == Some(&(kv + 1)));
        }
        assert!(leaf.verify());
    }

    // test insert out of order
    #[test]
    fn test_bptree_leaf_insert_out_of_order() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();

        let kvs = [7, 5, 1, 6, 2, 3, 0, 8, 4, 9];

        for idx in 0..L_CAPACITY {
            let kv = kvs[idx];
            let r = leaf.insert_or_update(0, kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
            let gr = leaf.get_ref(&kv);
            assert!(gr == Some(&kv));
        }
        assert!(leaf.verify());
    }

    // test insert and update to over-write out of order.
    #[test]
    fn test_bptree_leaf_update_out_of_order() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();

        let kvs = [7, 5, 1, 6, 2, 3, 0, 8, 4, 9];

        for idx in 0..L_CAPACITY {
            let kv = kvs[idx];
            let r = leaf.insert_or_update(0, kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
            let gr = leaf.get_ref(&kv);
            assert!(gr == Some(&kv));
        }
        assert!(leaf.verify());

        for idx in 0..L_CAPACITY {
            let kv = kvs[idx];
            let r = leaf.insert_or_update(0, kv, kv + 1);
            match r {
                BLInsertState::Ok(Some(_kv)) => {}
                _ => panic!(),
            }
            let gr = leaf.get_ref(&kv);
            assert!(gr == Some(&(kv + 1)));
        }
        assert!(leaf.verify());
    }

    // assert min-max bounds correctly are found.
    #[test]
    fn test_bptree_leaf_max() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();

        let kvs = [1, 3, 2, 6, 4, 5, 9, 8, 7, 0];
        let max = [1, 3, 3, 6, 6, 6, 9, 9, 9, 9];

        for idx in 0..L_CAPACITY {
            let kv = kvs[idx];
            let r = leaf.insert_or_update(0, kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
            let gr = leaf.max();
            assert!(*gr == max[idx]);
        }
        assert!(leaf.verify());
    }

    #[test]
    fn test_bptree_leaf_min() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();

        let kvs = [3, 2, 6, 4, 5, 1, 9, 8, 7, 0];
        let min = [3, 2, 2, 2, 2, 1, 1, 1, 1, 0];

        for idx in 0..L_CAPACITY {
            let kv = kvs[idx];
            let r = leaf.insert_or_update(0, kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
            let gr = leaf.min();
            assert!(*gr == min[idx]);
        }
        assert!(leaf.verify());
    }

    // insert to split.
    #[test]
    fn test_bptree_leaf_insert_split() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();
        let high = L_CAPACITY + 2;
        // First we insert from 1 to capacity + 1.
        for kv in 1..(L_CAPACITY + 1) {
            let r = leaf.insert_or_update(0, kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
        }
        // Then we insert capacity + 2, and should get that back.
        let r_over = leaf.insert_or_update(0, high, high);
        match r_over {
            BLInsertState::Split(_) => {}
            _ => panic!(),
        }
        // Then we insert 0, and we should get capacity + 1 back
        /*
        let r_under = leaf.insert_or_update(0, 0, 0);
        match r_under {
            BLInsertState::Split(_) => assert!(L_CAPACITY == high),
            _ => panic!(),
        }
        assert!(leaf.len() == L_CAPACITY);
        */
    }

    // remove in order
    #[test]
    fn test_bptree_leaf_remove_order() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();
        for kv in 0..L_CAPACITY {
            let _ = leaf.insert_or_update(0, kv, kv);
        }
        // Remove all but one!
        for kv in 0..(L_CAPACITY - 1) {
            let r = leaf.remove(&kv);
            match r {
                BLRemoveState::Ok(Some(_kv)) => {}
                _ => panic!(),
            }
        }
        println!("{:?}", leaf.max());
        assert!(leaf.max() == &(L_CAPACITY - 1));

        // Remove non-existant
        let r = leaf.remove(&(L_CAPACITY + 20));
        match r {
            BLRemoveState::Ok(None) => {}
            _ => panic!(),
        }
        // Remove the last item.
        let r = leaf.remove(&(L_CAPACITY - 1));
        match r {
            BLRemoveState::Shrink(Some(_)) => {}
            _ => panic!(),
        }
        // Remove non-existant post shrink
        let r = leaf.remove(&0);
        match r {
            BLRemoveState::Shrink(None) => {}
            _ => panic!(),
        }
    }

    // remove out of order
    #[test]
    fn test_bptree_leaf_remove_out_of_order() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();
        for kv in 0..L_CAPACITY {
            let _ = leaf.insert_or_update(0, kv, kv);
        }
        // Remove all but one!
        for kv in (L_CAPACITY / 2)..(L_CAPACITY - 1) {
            let r = leaf.remove(&kv);
            assert!(leaf.verify());
            match r {
                BLRemoveState::Ok(_) => {}
                _ => panic!(),
            }
        }

        for kv in 0..(L_CAPACITY / 2) {
            let r = leaf.remove(&kv);
            match r {
                BLRemoveState::Ok(_) => {}
                _ => panic!(),
            }
        }
    }

    #[test]
    fn test_bptree_leaf_remove_lt() {
        // Test removing less than or equal.
        // No op
        let mut leaf1: Leaf<usize, usize> = Leaf::new();
        for kv in 0..L_CAPACITY {
            let _ = leaf1.insert_or_update(0, kv + 10, kv);
        }

        leaf1.remove_lt(&5);
        // Removes all values
        let mut leaf2: Leaf<usize, usize> = Leaf::new();
        for kv in 0..L_CAPACITY {
            let _ = leaf2.insert_or_update(0, kv + 10, kv);
        }

        leaf2.remove_lt(&50);
        // Removes from middle.
        let mut leaf3: Leaf<usize, usize> = Leaf::new();
        for kv in 0..L_CAPACITY {
            let _ = leaf3.insert_or_update(0, kv + 10, kv);
        }

        leaf3.remove_lt(&((L_CAPACITY / 2) + 10));

        // Remove less then where not in leaf.
        let mut leaf4: Leaf<usize, usize> = Leaf::new();
        let _ = leaf4.insert_or_update(0, 5, 5);
        let _ = leaf4.insert_or_update(0, 15, 15);

        leaf4.remove_lt(&5);
        assert!(leaf4.len() == 2);

        leaf4.remove_lt(&10);
        assert!(leaf4.len() == 1);

        let mut leaf5: Leaf<usize, usize> = Leaf::new();
        let _ = leaf5.insert_or_update(0, 5, 5);
        let _ = leaf5.insert_or_update(0, 15, 15);

        leaf5.remove_lt(&16);
        assert!(leaf5.len() == 0);
    }
}
