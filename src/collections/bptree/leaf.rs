use std::mem::MaybeUninit;
use std::ptr;

use super::constants::L_CAPACITY;
use super::states::{BLInsertState, BLRemoveState};

pub(crate) struct Leaf<K, V> {
    count: usize,
    key: [MaybeUninit<K>; L_CAPACITY],
    value: [MaybeUninit<V>; L_CAPACITY],
}

impl<K: PartialEq, V> Leaf<K, V> {
    pub fn new() -> Self {
        Leaf {
            count: 0,
            key: unsafe { MaybeUninit::uninit().assume_init() },
            value: unsafe { MaybeUninit::uninit().assume_init() },
        }
    }

    fn insert_or_update(&mut self, k: K, v: V) -> BLInsertState<K, V> {
        // Update the node, and split if required.
        // There are four possible states.
        // * no values exist yet, so we should simply add the value
        if self.count == 0 {
            unsafe {
                self.key[0].as_mut_ptr()
                    .write(k);
                self.value[0].as_mut_ptr()
                    .write(v);
            }
            self.count = 1;
            BLInsertState::Ok(None)
        } else if self.count == L_CAPACITY {
            // * the leaf is full, so we need to split and progress.
            unimplemented!();
        } else {
            // * some values (but not full) exist, and we need to add the value that does not exist
            // * some values (but not full) exist, and we need to update the value that doe exist
            for idx in 0..self.count {
                unsafe {
                    if *self.key[idx].as_ptr() == k {
                        // Update in place.
                        let prev = self.value[idx].as_mut_ptr()
                            .replace(v);
                        // v now contains the original value, return it!
                        return BLInsertState::Ok(Some(prev))
                    }
                }
            }
            // If we get here, not found - append.
            // Because self.count will be +1 to idx, then we can use it here before we
            // increment.
            unsafe {
                self.key[self.count].as_mut_ptr()
                    .write(k);
                self.value[self.count].as_mut_ptr()
                    .write(v);
            }
            self.count += 1;
            BLInsertState::Ok(None)
        }
    }

    fn remove(&mut self, k: K, v: V) -> BLRemoveState<V> {
        // Remove a value, and if we are empty, mark that we require
        // to be shrunk.
        unimplemented!();
    }

    fn min(&self) -> &K {
        unimplemented!();
    }

    fn max(&self) -> &K {
        unimplemented!();
    }

    pub(crate) fn get_ref(&self, k: &K) -> Option<&V> {
        for idx in 0..self.count {
            unsafe {
                if &*self.key[idx].as_ptr() == k {
                    // Shortcut return.
                    return Some(&*self.value[idx].as_ptr())
                }
            }
        }
        // Not found, base case.
        None
    }

    fn get_mut_ref(&mut self, k: &K) -> Option<&mut V> {
        for idx in 0..self.count {
            unsafe {
                if &*self.key[idx].as_ptr() == k {
                    // Shortcut return.
                    return Some(&mut *self.value[idx].as_mut_ptr())
                }
            }
        }
        // Not found, base case.
        None
    }

    pub(crate) fn len(&self) -> usize {
        self.count
    }
}

impl<K: Clone, V: Clone> Clone for Leaf<K, V> {
    fn clone(&self) -> Self {
        let mut nkey: [MaybeUninit<K>; L_CAPACITY] = unsafe { MaybeUninit::uninit().assume_init() };
        let mut nvalue: [MaybeUninit<V>; L_CAPACITY] = unsafe { MaybeUninit::uninit().assume_init() };

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
        println!("leaf dropped {:?}", self.count);
    }
}


#[cfg(test)]
mod tests {
    use super::Leaf;
    use super::super::constants::L_CAPACITY;
    use super::super::states::BLInsertState;

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
            println!("len -> {}", leaf.len());
            let gr = leaf.get_ref(&kv);
            println!("{:?}", gr);
            assert!(gr == Some(&kv));
        }
    }

    // test insert and update to over-write in order.

    // test insert out of order

    // test insert and update to over-write out of order.

    // assert min-max bounds correct
    // insert to split.

    // remove

    // remove in order

    // remove out of order

    // remove to empty

}
