use super::constants::D_CAPACITY;
use super::utils::M;
use std::fmt::Debug;

#[derive(Debug)]
pub struct DenseLeaf<V> {
    value: [M<V>; D_CAPACITY],
}

impl<V> DenseLeaf<V>
where
    V: Clone + PartialEq + Debug,
{
    pub fn new() -> Self {
        DenseLeaf {
            value: [
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
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

    // if index is invalid returns None otherwise
    // returns Some(index)
    //
    // index can neither be lower than min or larger
    // than (min + D_CAPACITY)-1. The former makes sure
    // there can be no usize subtraction overflow
    fn get_index(&mut self, min: &usize, key: &usize) -> Option<usize> {
        if (key >= &(min + D_CAPACITY)) || (key < min) {
            return None;
        }

        Some(key - min)
    }

    // internal function used by insert and update
    // if the index at key - min isn't valid returns false
    //
    // when update is false only inserts the value if the
    // value at key - min is M::None otherwise returns false
    //
    // when update is true inserts the value no matter what the
    // state of the value is at key - min and returns true
    fn _insert(&mut self, min: &usize, key: usize, value: V, update: bool) -> bool {
        let index: usize = match self.get_index(min, &key) {
            Some(i) => i,
            None => return false,
        };

        if let M::Some(_) = self.value[index] {
            if !update {
                return false;
            }
        }
        match self.value[index] {
            M::Some(_) if !update => return false,
            _ => {}
        }

        self.value[index] = M::Some(value);
        true
    }

    // inserts the value at key - min
    // if the index key - min isn't valid return false,
    //
    // only insert if the value at key - min is M::None
    // otherwise return false
    pub fn insert(&mut self, min: &usize, key: usize, value: V) -> bool {
        self._insert(min, key, value, false)
    }

    // updates the value at key - min
    //
    // if the index key - min isn't valid return false,
    // otherwise set the value at key - min to value
    pub fn update(&mut self, min: &usize, key: usize, value: V) -> bool {
        self._insert(min, key, value, true)
    }

    // searches for the value at key - min,
    //
    // if the index key - min isn't valid None is returned, otherwise
    // the value at key - min is returned as an Option<> instead of M<>
    pub fn search(&mut self, min: &usize, key: usize) -> Option<V> {
        let index: usize;

        index = self.get_index(min, &key)?;

        match &self.value[index] {
            M::Some(v) => return Some(v.clone()),
            M::None => return None,
        }
    }

    // delete the value at key - min,
    //
    // if the index key - min or the value it refers to isn't valid
    // return false, otherwise return true
    pub fn delete(&mut self, min: &usize, key: usize) -> bool {
        let index: usize = match self.get_index(min, &key) {
            Some(i) => i,
            None => return false,
        };

        if let M::None = self.value[index] {
            return false;
        }

        self.value[index] = M::None;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::DenseLeaf;
    use collections::maple_tree::constants::D_CAPACITY;
    use collections::maple_tree::utils::M;

    fn check_dense_node_state(values: [usize; D_CAPACITY], dl: &DenseLeaf<usize>) -> bool {
        for (i, val) in values.iter().enumerate() {
            if *val != 0 {
                assert!(dl.value[i] == M::Some(*val));
            } else {
                assert!(dl.value[i] == M::None);
            }
        }

        true
    }

    #[test]
    fn test_dense_leaf_insert() {
        let mut dl: DenseLeaf<usize> = DenseLeaf::new();
        let min: usize = 10;

        // keys cannot be less than min or larger than (min + D_CAPACITY)-1
        assert!(dl.insert(&min, 9, 1) == false);
        assert!(dl.insert(&min, min + D_CAPACITY, 1) == false);

        // insert in the first place
        assert!(dl.insert(&min, min, 1) == true);

        println!("{:?}", dl);
        assert!(
            check_dense_node_state([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &dl) == true
        );

        // insert into the first place again fails
        assert!(dl.insert(&min, min, 1) == false);

        for i in 1..16 {
            assert!(dl.insert(&min, min + i, i + 1) == true);
        }

        assert!(
            check_dense_node_state([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], &dl)
                == true
        );
    }

    #[test]
    fn test_dense_leaf_update() {
        let mut dl: DenseLeaf<usize> = DenseLeaf::new();
        let min: usize = 10;

        // update to a key with M::None still works
        assert!(dl.update(&min, min, 1) == true);

        assert!(dl.update(&min, min, 5) == true);

        assert!(check_dense_node_state(
            [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &dl
        ));
    }

    #[test]
    fn test_dense_leaf_delete() {
        let mut dl: DenseLeaf<usize> = DenseLeaf::new();
        let min: usize = 10;

        for i in 0..16 {
            dl.insert(&min, min + i, i + 1);
        }

        assert!(dl.delete(&min, min) == true);
        assert!(dl.delete(&min, min + 5) == true);
        assert!(dl.delete(&min, min + 10) == true);
        assert!(
            check_dense_node_state([0, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 16], &dl)
                == true
        );

        // trying to delete a value that is M::None should return false
        assert!(dl.delete(&min, min) == false);
        assert!(dl.delete(&min, min + 5) == false);
        assert!(dl.delete(&min, min + 10) == false);
    }

    #[test]
    fn teset_dense_leaf_search() {
        let mut dl: DenseLeaf<usize> = DenseLeaf::new();
        let min: usize = 10;

        for i in 0..16 {
            dl.insert(&min, min + i, i + 1);
        }

        for i in 0..16 {
            assert!(dl.search(&min, min + i) == Some(i + 1));
        }

        assert!(dl.search(&min, 15) == Some(6));

        // search for index that falls out of the acceptable range
        assert!(dl.search(&min, min + D_CAPACITY) == None);
    }
}
