use super::utils::M;
use super::constants::CAPACITY;
use std::fmt::Debug;
use std::mem;

#[derive(Debug)]
pub struct SparseLeaf<K, V> {
    // This is an unsorted set of K, V pairs. Insert just appends (if possible),
    //, remove does NOT move the slots. On split, we sort-compact then move some
    // values (if needed).
    key: [M<K>; CAPACITY],
    value: [M<V>; CAPACITY],
}

impl<K, V> SparseLeaf<K, V>
where
    K: Clone + PartialEq + PartialOrd + Debug,
    V: Debug,
{
    pub fn new() -> Self {
        SparseLeaf {
            key: [
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
            ],
            value: [
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
            ],
        }
    }

    // insert takes a key and a value and tries to find a place to
    // insert it, if the sparseLeaf is already full and therefore
    // the key and value can't be inserted None is returned
    pub fn insert(&mut self, k: K, v: V) -> Option<()> {
        // Just find a free slot and insert
        for i in 0..CAPACITY {
            println!("insert: {:?}", self.key[i]);
            if !self.key[i].is_some() {
                // Make the new k: v here
                let mut nk = M::Some(k);
                let mut nv = M::Some(v);
                // swap them
                mem::swap(&mut self.key[i], &mut nk);
                mem::swap(&mut self.value[i], &mut nv);
                // return Some
                return Some(());
            }
        }

        None
    }

    // search tries to find a key that matches k within the sparseLeaf
    // if the key can't be found None is returned
    // if the key is found but the value associated to it is a None value
    // since this is not allowed we panick
    pub fn search(&mut self, k: &K) -> Option<&V> {
        for i in 0..CAPACITY {
            match &self.key[i] {
                M::Some(v) => {
                    if v == k {
                        match &self.value[i] {
                            M::Some(v) => {
                                return Some(v);
                            }
                            M::None => panic!("SparseLeaf - search failure. None value found associated to a valid key, aborting."),
                        }
                    }
                }

                M::None => {}
            }
        }

        None
    }

    // update attempts to change the associated value of a key with a new value
    // if the key isn't found then None is returned and nothing is updated,
    // if the value that was associated with the key before the update was a None
    //  value then we panick because that should not happen
    pub fn update(&mut self, k: K, v: V) -> Option<V> {
        for i in 0..CAPACITY {
            match &self.key[i] {
                M::Some(v) => {
                    if v != &k {
                        continue;
                    }
                }

                M::None => {
                    continue;
                }
            }

            let mut nv = M::Some(v);

            mem::swap(&mut self.value[i], &mut nv);

            match nv {
                M::Some(v) => return Some(v),
                M::None => panic!("SparseLeaf - update failure. None value found associated to a valid key, aborting."),
            }
        }

        None
    }

    // remove attempts to delete a key/value pair from the sparseLeaf
    // if the key is found and the value isn't None then the value is returned
    // if the key isn't found then None is returned
    // if the keys associated value is None then we panic because that shouldn't happen
    pub fn remove(&mut self, k: &K) -> Option<V> {
        for i in 0..CAPACITY {
            println!("remove: {:?}", self.key[i]);

            match &self.key[i] {
                M::Some(v) => {
                    if v != k {
                        continue;
                    }
                }
                M::None => {
                    continue;
                }
            }

            let mut nk = M::None;
            let mut nv = M::None;

            mem::swap(&mut self.key[i], &mut nk);
            mem::swap(&mut self.value[i], &mut nv);

            match nv {
                M::Some(v) => {
                    return Some(v);
                }
                M::None => panic!(
                    "SparseLeaf - remove() None value found associated to a valid key, aborting."
                ),
            }
        }

        None
    }

    // either returns some(k) holding the largest key in the node
    // or none if the node is empty
    pub fn get_max(&self) -> Option<&K> {
        let mut max: &M<K> = &self.key[0];
        let mut key_found: bool = false;

        for key in self.key.iter() {
            match key {
                M::Some(_) => {
                    if key_found == false {
                        max = key;
                        key_found = true;
                    } else if max < key {
                        max = key;
                    }
                }
                M::None => continue,
            }
        }

        match max {
            M::Some(v) => return Some(&v),
            M::None => return None,
        }
    }

    // either returns some(k) holding the smallest key in the node
    // or none if the node is empty
    pub fn get_min(&self) -> Option<&K> {
        let mut min: &M<K> = &self.key[0];
        let mut key_found: bool = false;

        for key in self.key.iter() {
            match key {
                M::Some(_k) => {
                    if key_found == false {
                        min = key;
                        key_found = true;
                    } else if min > key {
                        min = key;
                    }
                }
                M::None => {}
            }
        }

        match min {
            M::Some(k) => return Some(k),
            M::None => return None,
        }
    }

    // This function is used to help verify the validity of the entire tree
    // this function returns true if all keys within the SparseLeaf are within the bounds
    // min to max or equal to min or max or the SparseLeaf node is empty
    // otherwise this returns false
    pub fn check_bounds(&mut self, min_bound: &K, max_bound: &K) -> bool {
        let min = self.get_min();
        let max = self.get_max();

        // if either min or max is None they must both be None
        // if they are both None then the Node MUST be empty and
        // we can return true
        if min == None && max == None {
            return true;
        }

        if min >= Some(&min_bound) && max <= Some(&max_bound) {
            return true;
        }

        false
    }

    // We need to sort *just before* we split if required.
    // This function implements selection sort for a SparseLeaf
    // creates a new SparseLeaf struct, inserts the minimal values
    // one by one then overwrites the old struct with the new one
    //
    // we create the new node so we don't have to deal with the None values
    // being in-between values otherwise the code would be more complex to handle
    // compacting the values and then sorting or vice versa so there is no gaps
    // between actual keys in the underlying array
    pub fn sort(&mut self) {
        let mut smallest_key_index: usize;
        let mut sl: SparseLeaf<K, V> = SparseLeaf::new();
        let mut sl_index: usize = 0;

        // run once for every key in the sparseLeaf
        for _ in 0..CAPACITY {
            smallest_key_index = 0;

            // run a pass over the remaining items to be sorted to find the
            // entry with the smallest key and swap it for the item at currentIndex
            for i in 0..8 {
                match self.key[i] {
                    M::Some(_) => {
                        if self.key[i] < self.key[smallest_key_index] {
                            smallest_key_index = i;
                        }
                    }
                    M::None => continue,
                }
            }

            // swap the found element into the new SparseLeaf with the M::None
            // that is already in the SparseLeaf instead of just using the insert method
            // on the new SparseLeaf so the sorting function will keep working
            //
            // we could also just just insert the values into the new node and set the value of
            // the old node to M::None manually but that would require more code and I figured
            // this was a bit cleaner, thoughts?
            mem::swap(&mut self.key[smallest_key_index], &mut sl.key[sl_index]);
            mem::swap(&mut self.value[smallest_key_index], &mut sl.value[sl_index]);
            sl_index += 1;
        }

        *self = sl;
    }
}


#[cfg(test)]
mod tests{
    use super::SparseLeaf;
    use collections::maple_tree::utils::M;
    use collections::maple_tree::constants::CAPACITY;


    #[test]
    fn test_sparse_leaf_get_max() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();
        let test_vals: [usize; 8] = [3, 8, 7, 4, 2, 1, 5, 6];

        for val in test_vals.iter() {
            sl.insert(*val, *val);
        }

        // check that get_max() works for a full node
        assert!(sl.get_max() == Some(&8));

        //check that get_max() works for a node with Nones inbetween values
        let del_vals: [usize; 4] = [3, 8, 2, 1];
        for val in del_vals.iter() {
            sl.remove(val);
        }

        assert!(sl.get_max() == Some(&7));

        // check that get_min() works for empty nodes
        let sl_empty: SparseLeaf<usize, usize> = SparseLeaf::new();
        assert!(sl_empty.get_max() == None);
    }

    #[test]
    fn test_sparse_leaf_get_min() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();
        let test_vals: [usize; 8] = [3, 8, 7, 4, 2, 1, 5, 6];

        for val in test_vals.iter() {
            sl.insert(*val, *val);
        }

        // check that get_min() works for a full node
        assert!(sl.get_min() == Some(&1));

        //check that get_min() works for a node with Nones inbetween values
        let del_vals: [usize; 4] = [3, 8, 2, 1];
        for val in del_vals.iter() {
            sl.remove(val);
        }

        assert!(sl.get_min() == Some(&4));

        // check that get_min() works for empty nodes
        let sl_empty: SparseLeaf<usize, usize> = SparseLeaf::new();
        assert!(sl_empty.get_min() == None);
    }

    #[test]
    fn test_sparse_leaf_search() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();

        // test valid search
        sl.insert(2, 2);

        assert!(sl.search(&2).is_some());

        // test invalid search
        assert!(sl.search(&3).is_none());

        sl = SparseLeaf::new();

        for i in 0..CAPACITY {
            sl.insert(i, i);
        }

        sl.remove(&3);

        assert!(sl.search(&4).is_some());
    }

    #[test]
    fn test_sparse_leaf_update() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();

        // Insert K:V pair
        sl.insert(2, 2);

        // update inplace.
        sl.update(2, 3);

        // check that the value was correctly changed
        assert!(sl.search(&2) == Some(&3));
    }

    #[test]
    fn test_sparse_leaf_insert() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();

        // insert
        sl.insert(2, 2);

        // test valid search
        assert!(sl.search(&2) == Some(&2));

        // test invalid search
        assert!(sl.search(&1).is_none());

        // test insert after node is already full

        for i in 1..CAPACITY {
            sl.insert(i, i);
        }

        assert!(sl.insert(8, 8).is_none())
    }

    #[test]
    fn test_sparse_leaf_remove() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();

        // check removing a non-existent value fails
        assert!(sl.remove(&0).is_none());

        // check removing a value that exists
        sl.insert(0, 0);
        assert!(sl.remove(&0).is_some());

        // check removing existing values out of order is successfull
        let remove_keys = [3, 7, 8, 1, 4];
        for i in 0..CAPACITY {
            sl.insert(i, i);
        }
        for i in 0..remove_keys.len() {
            assert!(sl.remove(&i).is_some());
        }
    }

    #[test]
    fn test_sparse_leaf_check_bounds() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();

        // test that check_min_max returns true when the sparseLeaf is empty
        assert!(sl.check_bounds(&0, &8));

        // insert 8 values from 0 - 7
        for i in 0..CAPACITY - 3 {
            sl.insert(i, i);
        }

        assert!(sl.check_bounds(&0, &8));

        // test that check_min_max returns some when the values are out of the range
        // and returns the first value that is found outside the range.

        sl.insert(10, 10);
        sl.insert(11, 11);
        sl.insert(12, 12);
        assert!(sl.check_bounds(&0, &8) == false);
    }

    #[test]
    fn test_sparse_leaf_sort() {
        // test sorting full node
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();
        let mut test_vals: [usize; 8] = [3, 8, 7, 4, 2, 1, 5, 6];
        let sorted_test_vals: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

        for val in test_vals.iter() {
            sl.insert(*val, *val);
        }
        for i in 0..CAPACITY {
            match sl.key[i] {
                M::Some(v) => {
                    println!("{0}", v);
                }
                M::None => println!("None"),
            }
        }
        sl.sort();

        for i in 0..CAPACITY {
            // the code inside match is usefull for debuging if a test fails
            match sl.key[i] {
                M::Some(v) => {
                    println!(
                        "(actualValue = {0}) - (sortedTestValue = {1})",
                        v, sorted_test_vals[i]
                    );
                }
                M::None => println!("None - {}", sorted_test_vals[i]),
            }
            assert!(sl.key[i] == M::Some(sorted_test_vals[i]));
        }

        // test sorting half full node with M::None's inbetween each value
        // i.e [3, None, 4, None, 2, None, 1, None]

        test_vals = [3, 8, 4, 6, 2, 7, 1, 5];
        let none_positions: [usize; 4] = [8, 6, 7, 5];
        sl = SparseLeaf::new();

        for val in test_vals.iter() {
            sl.insert(*val, *val);
        }

        // remove every second value from sl
        for val in none_positions.iter() {
            sl.remove(&val);
        }

        sl.sort();

        for i in 0..4 {
            println!("{} <-> ", sorted_test_vals[i]);
            assert!(sl.key[i] == M::Some(sorted_test_vals[i]));
        }
    }

}
