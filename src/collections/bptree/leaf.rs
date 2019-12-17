use std::fmt::{self, Debug, Error};
use std::mem::MaybeUninit;
use std::ptr;

use super::constants::L_CAPACITY;
use super::states::{BLInsertState, BLRemoveState};

pub(crate) struct Leaf<K, V> {
    count: usize,
    key: [MaybeUninit<K>; L_CAPACITY],
    value: [MaybeUninit<V>; L_CAPACITY],
}

impl<K: PartialEq + PartialOrd, V> Leaf<K, V> {
    pub fn new() -> Self {
        Leaf {
            count: 0,
            key: unsafe { MaybeUninit::uninit().assume_init() },
            value: unsafe { MaybeUninit::uninit().assume_init() },
        }
    }

    pub(crate) fn insert_or_update(&mut self, k: K, v: V) -> BLInsertState<K, V> {
        // Update the node, and split if required.

        // There are three possible paths
        // * some values (but not full) exist, and we need to update the value that does exist
        for idx in 0..self.count {
            unsafe {
                if *self.key[idx].as_ptr() == k {
                    // Update in place.
                    let prev = self.value[idx].as_mut_ptr().replace(v);
                    // v now contains the original value, return it!
                    return BLInsertState::Ok(Some(prev));
                }
            }
        }
        // If we get here, not found - append or split as needed
        if self.count == L_CAPACITY {
            // * The node is full, so we must indicate as such.
            // Find the largest value of this node
            let midx = self.max_idx();

            if &k > unsafe { &*self.key[midx].as_ptr() } {
                // if new value is larger than this nodes max, return k and v.
                BLInsertState::Split(k, v)
            } else {
                // if old value is larger swap new/old, and then return the existing value
                // to be put to a new neighbor.
                let pk = unsafe { self.key[midx].as_mut_ptr().replace(k) };
                let pv = unsafe { self.value[midx].as_mut_ptr().replace(v) };
                BLInsertState::Split(pk, pv)
            }
        } else {
            // * no values exist yet, so we should simply add the value
            // * some values (but not full) exist, and we need to add the value that does not exist
            // Because self.count will be +1 to idx, then we can use it here before we
            // increment.
            unsafe {
                self.key[self.count].as_mut_ptr().write(k);
                self.value[self.count].as_mut_ptr().write(v);
            }
            self.count += 1;
            BLInsertState::Ok(None)
        }
    }

    fn remove(&mut self, k: &K) -> BLRemoveState<V> {
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
                let pk = unsafe { slice_remove(&mut self.key, idx).assume_init() };
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

    fn min_idx(&self) -> usize {
        debug_assert!(self.count > 0);
        let mut idx: usize = 0;
        let mut tmp_k: &K = unsafe { &*self.key[0].as_ptr() };

        for work_idx in 1..self.count {
            let k = unsafe { &*self.key[work_idx].as_ptr() };
            if k < tmp_k {
                tmp_k = k;
                idx = work_idx
            }
        }

        idx
    }

    fn max_idx(&self) -> usize {
        debug_assert!(self.count > 0);
        let mut idx: usize = 0;
        let mut tmp_k: &K = unsafe { &*self.key[0].as_ptr() };

        for work_idx in 1..self.count {
            let k = unsafe { &*self.key[work_idx].as_ptr() };
            if k > tmp_k {
                tmp_k = k;
                idx = work_idx
            }
        }

        idx
    }

    pub(crate) fn min(&self) -> &K {
        let idx = self.min_idx();
        unsafe { &*self.key[idx].as_ptr() }
    }

    pub(crate) fn max(&self) -> &K {
        let idx = self.max_idx();
        unsafe { &*self.key[idx].as_ptr() }
    }

    fn get_idx(&self, k: &K) -> Option<usize> {
        for idx in 0..self.count {
            unsafe {
                if &*self.key[idx].as_ptr() == k {
                    // Shortcut return.
                    return Some(idx);
                }
            }
        }
        None
    }

    pub(crate) fn get_ref(&self, k: &K) -> Option<&V> {
        self.get_idx(k)
            .map(|idx| unsafe { &*self.value[idx].as_ptr() })
    }

    fn get_mut_ref(&mut self, k: &K) -> Option<&mut V> {
        self.get_idx(k)
            .map(|idx| unsafe { &mut *self.value[idx].as_mut_ptr() })
    }

    pub(crate) fn len(&self) -> usize {
        self.count
    }

    pub(crate) fn verify(&self) -> bool {
        true
    }
}

impl<K: Clone, V: Clone> Clone for Leaf<K, V> {
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

impl<K, V> Drop for Leaf<K, V> {
    fn drop(&mut self) {
        // Due to the use of maybe uninit we have to drop any contained values.
        for idx in 0..self.count {
            unsafe {
                ptr::drop_in_place(self.key[idx].as_mut_ptr());
                ptr::drop_in_place(self.value[idx].as_mut_ptr());
            }
        }
        // println!("leaf dropped {:?}", self.count);
    }
}

impl<K, V> Debug for Leaf<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), Error> {
        write!(f, "Leaf -> {}", self.count)
    }
}

// From std::collections::btree::node.rs
unsafe fn slice_remove<T>(slice: &mut [T], idx: usize) -> T {
    // setup the value to be returned, IE give ownership to ret.
    let ret = ptr::read(slice.get_unchecked(idx));
    ptr::copy(
        slice.as_ptr().add(idx + 1),
        slice.as_mut_ptr().add(idx),
        slice.len() - idx - 1,
    );
    ret
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
            let r = leaf.insert_or_update(kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
            let gr = leaf.get_ref(&kv);
            assert!(gr == Some(&kv));
        }
    }

    // test insert and update to over-write in order.
    #[test]
    fn test_bptree_leaf_update_order() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();

        for kv in 0..L_CAPACITY {
            let r = leaf.insert_or_update(kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
            let gr = leaf.get_ref(&kv);
            assert!(gr == Some(&kv));
        }

        for kv in 0..L_CAPACITY {
            let r = leaf.insert_or_update(kv, kv + 1);
            match r {
                // Check for some kv, that was the former value.
                BLInsertState::Ok(Some(kv)) => {}
                _ => panic!(),
            }
            let gr = leaf.get_ref(&kv);
            // Check the new value is incremented.
            assert!(gr == Some(&(kv + 1)));
        }
    }

    // test insert out of order
    #[test]
    fn test_bptree_leaf_insert_out_of_order() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();

        let kvs = [7, 5, 1, 6, 2, 3, 0, 8, 4, 9];

        for idx in 0..L_CAPACITY {
            let kv = kvs[idx];
            let r = leaf.insert_or_update(kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
            let gr = leaf.get_ref(&kv);
            assert!(gr == Some(&kv));
        }
    }

    // test insert and update to over-write out of order.
    #[test]
    fn test_bptree_leaf_update_out_of_order() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();

        let kvs = [7, 5, 1, 6, 2, 3, 0, 8, 4, 9];

        for idx in 0..L_CAPACITY {
            let kv = kvs[idx];
            let r = leaf.insert_or_update(kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
            let gr = leaf.get_ref(&kv);
            assert!(gr == Some(&kv));
        }

        for idx in 0..L_CAPACITY {
            let kv = kvs[idx];
            let r = leaf.insert_or_update(kv, kv + 1);
            match r {
                BLInsertState::Ok(Some(kv)) => {}
                _ => panic!(),
            }
            let gr = leaf.get_ref(&kv);
            assert!(gr == Some(&(kv + 1)));
        }
    }

    // assert min-max bounds correctly are found.
    #[test]
    fn test_bptree_leaf_max() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();

        let kvs = [1, 3, 2, 6, 4, 5, 9, 8, 7, 0];
        let max = [1, 3, 3, 6, 6, 6, 9, 9, 9, 9];

        for idx in 0..L_CAPACITY {
            let kv = kvs[idx];
            let r = leaf.insert_or_update(kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
            let gr = leaf.max();
            assert!(*gr == max[idx]);
        }
    }

    #[test]
    fn test_bptree_leaf_min() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();

        let kvs = [3, 2, 6, 4, 5, 1, 9, 8, 7, 0];
        let min = [3, 2, 2, 2, 2, 1, 1, 1, 1, 0];

        for idx in 0..L_CAPACITY {
            let kv = kvs[idx];
            let r = leaf.insert_or_update(kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
            let gr = leaf.min();
            assert!(*gr == min[idx]);
        }
    }

    // insert to split.
    #[test]
    fn test_bptree_leaf_insert_split() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();
        let high = L_CAPACITY + 2;
        // First we insert from 1 to capacity + 1.
        for kv in 1..(L_CAPACITY + 1) {
            let r = leaf.insert_or_update(kv, kv);
            match r {
                BLInsertState::Ok(None) => {}
                _ => panic!(),
            }
        }
        // Then we insert capacity + 2, and should get that back.
        let r_over = leaf.insert_or_update(high, high);
        match r_over {
            BLInsertState::Split(high, _) => {}
            _ => panic!(),
        }
        // Then we insert 0, and we should get capacity + 1 back
        let zret = L_CAPACITY + 1;
        let r_under = leaf.insert_or_update(0, 0);
        match r_over {
            BLInsertState::Split(zret, _) => {}
            _ => panic!(),
        }
        assert!(leaf.len() == L_CAPACITY);
    }

    // remove in order
    #[test]
    fn test_bptree_leaf_remove_order() {
        let mut leaf: Leaf<usize, usize> = Leaf::new();
        for kv in 0..L_CAPACITY {
            let _ = leaf.insert_or_update(kv, kv);
        }
        // Remove all but one!
        for kv in 0..(L_CAPACITY - 1) {
            let r = leaf.remove(&kv);
            match r {
                BLRemoveState::Ok(Some(kv)) => {}
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
            let _ = leaf.insert_or_update(kv, kv);
        }
        // Remove all but one!
        for kv in (L_CAPACITY / 2)..(L_CAPACITY - 1) {
            let r = leaf.remove(&kv);
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

}
