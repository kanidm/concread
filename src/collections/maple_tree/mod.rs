use std::fmt::Debug;
use std::mem;
extern crate num;

// Number of k,v in sparse, and number of range values/links.
const CAPACITY: usize = 8;
// Number of pivots in range
const R_CAPACITY: usize = CAPACITY;
// Number of values in dense.
const D_CAPACITY: usize = CAPACITY * 2;

#[derive(PartialEq, PartialOrd, Clone, Eq, Ord, Debug, Hash)]
enum M<T> {
    Some(T),
    None,
}

impl<T> M<T> {
    fn is_some(&self) -> bool {
        match self {
            M::Some(_) => true,
            M::None => false,
        }
    }

    fn unwrap(self) -> T {
        match self {
            M::Some(v) => v,
            M::None => panic!(),
        }
    }
}

#[derive(Debug)]
struct SparseLeaf<K, V> {
    // This is an unsorted set of K, V pairs. Insert just appends (if possible),
    //, remove does NOT move the slots. On split, we sort-compact then move some
    // values (if needed).
    key: [M<K>; CAPACITY],
    value: [M<V>; CAPACITY],
}

#[derive(Debug)]
struct DenseLeaf<V> {
    value: [M<V>; D_CAPACITY],
}

#[derive(Debug)]
struct RangeLeaf<K, V> {
    pivot: [M<K>; CAPACITY],
    value: [M<V>; CAPACITY],
}

#[derive(Debug, PartialEq)]
enum RangeInsertInfo{
    Range(usize, usize),
    LeftPivot(usize),     
    FirstPivot,
}

#[derive(Debug)]
struct RangeBranch<K, V> {
    // Implied Pivots
    // Cap - 2
    pivot: [M<K>; R_CAPACITY],
    links: [M<*mut Node<K, V>>; CAPACITY],
}

// When K: SliceIndex, allow Dense
// When K: Binary, allow Range.

#[derive(Debug)]
pub enum NodeTag<K, V> {
    SL(SparseLeaf<K, V>),
    DL(DenseLeaf<V>),
    RL(RangeLeaf<K, V>),
    RB(RangeBranch<K, V>),
}

#[derive(Debug)]
struct Node<K, V> {
    tid: u64,
    // checksum: u32,
    inner: NodeTag<K, V>,
}



impl<K, V> RangeLeaf<K, V>
where
    K: Clone + PartialEq + PartialOrd + num::Num + Debug,
    V: Clone + PartialEq + Debug,
{
    pub fn new() -> Self {
        RangeLeaf {
            pivot: [
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

    // returns the number of used pivots
    // we can optimise this later by using the data returned from find_key_info
    // as a starting point to count from as we will know that at least RangeInfo.startIndex
    // or RangeInfo.endIndex(if its set) pivots will be  set
    pub fn len(&mut self) -> usize{
        let mut count: usize = 0;

        for pivot in self.pivot.iter() {
            match pivot {
                M::Some(_) => count += 1,
                // since the node is sorted with M::None's all being at the end of the node
                // when we encounter M::None there are never anymore pivots
                M::None => break,
            }
        }

        count
    }

    // returns the RangeInfo containing the found information
    // where p is located 
    // i.e if the following array is all the pivots of a Range node
    // [5, 6, 10, 11, 15, 20, 60, 90], if we are searching for 70
    // this function would return RangeInfo{ StartIndex: 6, endIndex: Some(7)} 
    pub fn find_key_info(&mut self, p: &K) -> RangeInsertInfo {
        let mut startIndex:usize = 0;
        let mut endIndex:usize = 0;
        let mut found = false;

        // loop over each pivot in the node and check if it 
        // is the start of the range that p is included in
        for i in 0..R_CAPACITY {
            match &self.pivot[i] {
                M::Some(pivot) => {
                    if p >= pivot{
                        // there is definantly a start index for the left pivot 
                        found = true;
                        startIndex = i;
                        continue;
                    }
                    else if pivot > p{
                        // as long as this isn't the first pivot in the node
                        // we found the start and end index of the range
                        if i > 0{
                            found = true;
                            startIndex = i-1;
                            endIndex = i;
                            break;
                        } 
                        // if the first pivot is larger than the p
                        // there can't be a valid range that includes p
                        // so the value will have to be inserted at the start of the
                        // array as the first pivot:value pair
                        else{
                            return RangeInsertInfo::FirstPivot;
                        }
                    }
                },
                M::None => break
            }
        }
       
        if found == false{
            return RangeInsertInfo::FirstPivot;
        }
        // if no endIndex was found it will still equal 0
        // so return None
        else if endIndex == 0{
            return RangeInsertInfo::LeftPivot(startIndex);
        }
        else{
            return RangeInsertInfo::Range(startIndex, endIndex);
        }
    }
    
    // moves the pivot/s at pivotIndex and above to the right x places where x = places variable
    // if the move would push values off the end then false is returned 
    pub fn move_pivots_right(&mut self, pivotIndex: usize, places: usize) -> bool{
        
        let freePivots = R_CAPACITY-self.len();

        // make sure there is enough room to move the pivots into
        if freePivots < places{
            return false;
        }
        // make sure the pivot is greater than 0 and less than the index of the 
        // last pivot as moving that would push it out of the array
        else if pivotIndex < 0 || pivotIndex >= R_CAPACITY-1{
            return false;
            //panic!("PivotIndex isn't within the correct bounds"); 
        }
        else if places > R_CAPACITY{
            return false;
            //panic!("the number of places to move the pivots was to large, must be bellow {}", R_CAPACITY); 
        }

        // must use a variable to hold the pivots and values between swaps as we can't have two
        // mutable borrows of the same array at the same time :(
        let mut pivotHolder: M<K> = M::None;
        let mut valueHolder: M<V> = M::None;
        let lastIndex = (R_CAPACITY-1)-freePivots;
        
        // starts at lastPivotIndex and goes until reaching the pivot at pivotIndex 
        // so moving pivots and values from righ to left so each pivot and value is moved 
        // into a free place in the arrays
        for i in 0..lastIndex+1{
            if lastIndex-i >= pivotIndex {
                match &self.pivot[lastIndex-i]{
                    M::None => {
                        continue; 
                    },
                    M::Some(_) => {  
                            mem::swap(&mut pivotHolder, &mut self.pivot[lastIndex-i]);
                            mem::swap(&mut pivotHolder, &mut self.pivot[lastIndex-i+places]);

                            mem::swap(&mut valueHolder, &mut self.value[lastIndex-i]);
                            mem::swap(&mut valueHolder, &mut self.value[lastIndex-i+places]);
                    }
                }
            }            
        }
        return true;
    }

    pub fn insert_into_range(&mut self, p: K, v: V, startIndex: usize, endIndex: usize, numPivots: usize) -> bool{
        // the end of the range equals the pivot to be inserted, since the ranges are
        // exclusive for the right pivot we need to increment the right pivot by one to
        // include the new pivot 
        if self.pivot[endIndex] == M::Some(p.clone()){

            if self.value[endIndex] == M::Some(v.clone()){

                mem::swap(&mut self.pivot[endIndex], &mut M::Some(p.clone() + K::one()));  
                return true;
            }
            else if numPivots < R_CAPACITY{
                

            }
            else{
                return false;
            }

        }
        // the pivot is already in the node as a range with only that single pivot
        // value so all we need to do is update the value
        else if (endIndex - startIndex) == 1{
            mem::swap(&mut self.value[startIndex], &mut M::Some(v.clone()));
            return true;
        }
        // make sure there are two pivots because we are inserting 2 ranges
        // one between the last pivot and the new range who's value is M::None
        else if numPivots < R_CAPACITY-2{
            mem::swap(&mut self.pivot[startIndex+1], &mut M::Some(p.clone()));  
            mem::swap(&mut self.value[startIndex], &mut M::None);
            mem::swap(&mut self.pivot[startIndex+2], &mut M::Some(p.clone()+K::one())); 
            mem::swap(&mut self.value[startIndex+1], &mut M::Some(v.clone()));
            return true;
        }

        return false;
    }

    // inserts a pivot and value after a specific index, this means that 
    // the pivotIndex is the last pivot with a value in the node 
    pub fn insert_after_pivot(&mut self, p: K, v: V, pivotIndex: usize) -> bool{
       
        // trying to insert the value after the last pivot in the array, this will only work if
        // the pivot to be inserted is one larger than the 
        if pivotIndex == R_CAPACITY-1{
            if M::Some(p.clone()) == self.pivot[pivotIndex] && 
                M::Some(v.clone()) == self.value[pivotIndex]{
                mem::swap(&mut self.pivot[pivotIndex], &mut M::Some(p.clone()));
                return true;
            }
            else{
                return false;
            }
        }
        else if pivotIndex < R_CAPACITY-1{
             if M::Some(p.clone()) == self.pivot[pivotIndex]{ 

                if M::Some(v.clone()) == self.value[pivotIndex-1]{
                    mem::swap(&mut self.pivot[pivotIndex], &mut M::Some(p + K::one()));
                    return true;
                }
                else if self.move_pivots_right(pivotIndex+1, 1){
                     mem::swap(&mut self.pivot[pivotIndex+1], &mut M::Some(p + K::one()));
                     mem::swap(&mut self.value[pivotIndex+1], &mut M::Some(v));
                     return true
                }
                
                return false;
            }   
            else if self.move_pivots_right(pivotIndex+1, 2){
                
                mem::swap(&mut self.pivot[pivotIndex+1], &mut M::Some(p.clone())); 
                mem::swap(&mut self.value[pivotIndex+1], &mut M::Some(v));
                mem::swap(&mut self.pivot[pivotIndex+2], &mut M::Some(p + K::one()));
                return true;
            } 
            
            return false;
        }
        else{
            return false;
        }
    }
  
    pub fn prepend_pivot(&mut self, p: K, v: V) -> bool{
        let upperLimit: K = p.clone() + K::one();
        let len = self.len();

        // check if the pivot to be inserted is one less than the current pivot at index 0     
        // because if it is we only need to either prepend one new pivot or swap the existsing first
        // pivot for the new one
        if M::Some(p.clone()+K::one()) == self.pivot[0]{
            
            // value remains the same for the first range just swap
            // the starting pivot over to the one to be inserted
            if M::Some(v.clone()) == self.value[0]{
                // if the value to be inserted is the same as that for the range between the first
                // pivot and the second pivot then we only need to insert the pivot in place
                mem::swap(&mut M::Some(p), &mut self.pivot[0]); 
                return true;
            }
            // need to insert a new pivot at the start
            else if self.move_pivots_right(0, 1){
                mem::swap(&mut M::Some(p), &mut self.pivot[0]);
                mem::swap(&mut M::Some(v), &mut self.value[0]);
                return true;
            }
            else{
                return false;
            }
            
        }
        else if self.move_pivots_right(0, 2){
            mem::swap(&mut M::Some(p), &mut self.pivot[0]);
            mem::swap(&mut M::Some(upperLimit), &mut self.pivot[1]);
            mem::swap(&mut M::Some(v), &mut self.value[0]);
            return true;
        }
        else{
            return false;
        }
    }

    pub fn insert_first_pivot(&mut self, p: K, v: V) -> bool {
        let upperLimit: K = p.clone() + K::one();

        // make sure the node is empty 
        if self.len() == 0{

            mem::swap(&mut M::Some(p), &mut self.pivot[0]);
            mem::swap(&mut M::Some(upperLimit), &mut self.pivot[1]);
            mem::swap(&mut M::Some(v), &mut self.value[0]);

            return true;
        }

        return false;
    }

    pub fn insert(&mut self, p: K, v: V) -> bool {
        
        let rangeInsertInfoResponse = self.find_key_info(&p);
        // if in RangeInsertInfo startIndex refers to the last pivot in the 
        // range(i.e equals R_CAPACITY-1) then
        // we can't reason about whether we can update the value or not because we don't know how
        // big the range is since the end of the range is in another node 
        let numPivots = self.len();

        if numPivots == R_CAPACITY{
            return false;
        }

        match rangeInsertInfoResponse{
            
            RangeInsertInfo::Range(startIndex, endIndex) => {
                return self.insert_into_range(p, v, startIndex, endIndex, numPivots);          
            },
            RangeInsertInfo::LeftPivot(pivotIndex) => {
                return self.insert_after_pivot(p, v, pivotIndex);    
            },
            RangeInsertInfo::FirstPivot => {
                return self.insert_first_pivot(p, v);
            }
        }
    }
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

    // returns the number of keys in a node
    // should we store a lenth value on a node instead of calculating it on the fly?
    pub fn get_len(&self) -> usize {
        let mut size: usize = 0;
        for key in self.key.iter() {
            match key {
                M::Some(_) => size += 1,
                M::None => continue,
            }
        }

        size
    }

    // either returns some(k) holding the smallest key in the node
    // or none if the node is empty
    pub fn get_min(&self) -> Option<&K> {
        let mut min: &M<K> = &self.key[0];
        let mut key_found: bool = false;

        for key in self.key.iter() {
            match key {
                M::Some(k) => {
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
    pub fn check_bounds(&mut self, minBound: &K, maxBound: &K) -> bool {
        let min = self.get_min();
        let max = self.get_max();

        // if either min or max is None they must both be None
        // if they are both None then the Node MUST be empty and
        // we can return true
        if min == None && max == None {
            return true;
        }

        if min >= Some(&minBound) && max <= Some(&maxBound) {
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
        for current_index in 0..CAPACITY {
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
mod tests {
    use super::SparseLeaf;
    use super::RangeLeaf;
    use super::M;
    use collections::maple_tree::CAPACITY;
    use collections::maple_tree::R_CAPACITY;
    use collections::maple_tree::RangeInsertInfo;

    #[test]
    fn test_sparse_leaf_get_len() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();

        assert!(sl.get_len() == 0);

        let mut test_vals: [usize; 8] = [3, 8, 7, 4, 2, 1, 5, 6];

        for val in test_vals.iter() {
            sl.insert(*val, *val);
        }

        assert!(sl.get_len() == 8);

        let del_vals: [usize; 4] = [3, 8, 2, 1];
        for val in del_vals.iter() {
            sl.remove(val);
        }

        assert!(sl.get_len() == 4);
    }

    #[test]
    fn test_sparse_leaf_get_max() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();
        let mut test_vals: [usize; 8] = [3, 8, 7, 4, 2, 1, 5, 6];

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
        let mut slEmpty: SparseLeaf<usize, usize> = SparseLeaf::new();
        assert!(slEmpty.get_max() == None);
    }

    #[test]
    fn test_sparse_leaf_get_min() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();
        let mut test_vals: [usize; 8] = [3, 8, 7, 4, 2, 1, 5, 6];

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
        let mut slEmpty: SparseLeaf<usize, usize> = SparseLeaf::new();
        assert!(slEmpty.get_min() == None);
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
        let mut sorted_test_vals: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

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

    #[test]
    fn  test_range_node_find_key_info(){
        
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();

        assert!(rn.find_key_info(&10) == RangeInsertInfo::FirstPivot);
        
        // setup internal state manually
        // it will look like this [2, 6, 7, 10, M::None, M::None, M::None, M::None] 
        rn.pivot[0] = M::Some(2);
        rn.pivot[1] = M::Some(6);
        rn.pivot[2] = M::Some(7);
        rn.pivot[3] = M::Some(10);
        
        assert!(rn.find_key_info(&9) == RangeInsertInfo::Range(2, 3));

        assert!(rn.find_key_info(&20) == RangeInsertInfo::LeftPivot(3));

        assert!(rn.find_key_info(&1) == RangeInsertInfo::FirstPivot);
    }

    #[test]
    fn test_range_node_prepend_pivot(){

        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();

        // rn pivot = [7, 10, M::None, ..., M::None]
        // rn value will equal [55, M::None, ..., M::None] 
        rn.pivot[0] = M::Some(7);
        rn.pivot[1] = M::Some(10);
        rn.value[0] = M::Some(55); 
        
        // rn pivot should be [6, 10, M::None, ..., M::None]
        // rn value should be [55, M::None, ..., M::None]
        rn.prepend_pivot(6, 55);
        assert!(rn.pivot[0] == M::Some(6) && rn.pivot[1] == M::Some(10) && rn.value[0] == M::Some(55));
        
        // rn pivot should be [5, 6, 10, M::None, ..., M::None]
        // rn value should be [33, 55, ..., M::None]
        rn.prepend_pivot(5, 33);
        assert!(rn.pivot[0] == M::Some(5) && rn.pivot[1] == M::Some(6) && rn.pivot[2] == M::Some(10) &&
                 rn.value[0] == M::Some(33) && rn.value[1] == M::Some(55));
        
        rn.pivot[2] = M::None;
        rn.value[1] = M::None;
        
        // rn pivot should be [2, 3, 5, 6, M::None, ..., M::None]
        // rn value should be [66, M::None, 33, ..., M::None]
        rn.prepend_pivot(3, 66);
        assert!(rn.pivot[0] == M::Some(3) && rn.pivot[1] == M::Some(4) && rn.pivot[2] == M::Some(5) &&
                rn.pivot[3] == M::Some(6) && rn.value[0] == M::Some(66) && rn.value[1] == M::None &&
                rn.value[2] == M::Some(33));

        rn.pivot[4] = M::Some(7);
        rn.pivot[5] = M::Some(9);
        rn.pivot[6] = M::Some(11);
        rn.pivot[7] = M::Some(15);
        rn.value[3] = M::Some(1);
        rn.value[4] = M::Some(2);
        rn.value[5] = M::Some(3);
        rn.value[6] = M::Some(4);
        rn.value[7] = M::Some(5);

        assert!(rn.prepend_pivot(1, 2) == false);
        assert!(rn.prepend_pivot(2, 2) == false);

    }
    
    #[test]
    fn test_range_node_insert_after_pivot(){
        let mut rn: RangeLeaf<usize,usize> = RangeLeaf::new(); 
        let pivotArray: [usize; 8] = [7, 11, 0, 0, 0, 0, 0, 0]; 
        let valueArray: [usize; 8] = [55, 0, 0, 0, 0, 0, 0, 0];

        // rn pivot = [7, 10, M::None, ..., M::None]
        // rn value will equal [55, M::None, ..., M::None] 
        rn.pivot[0] = M::Some(7);
        rn.pivot[1] = M::Some(10);
        rn.value[0] = M::Some(55); 
        
        assert!(rn.insert_after_pivot(10, 55, 1) == true);
        
        assert!(rn.pivot[0] == M::Some(7) && rn.pivot[1] == M::Some(11) && rn.value[0] == M::Some(55));
    }

    // since RangeNode::move_pivots_right
    // is used in RangeNode::insert we will test this function by
    // inserting the pivots and values manually
    #[test]
    fn test_range_node_move_pivots_right(){
        
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();
        rn.pivot[0] = M::Some(6);
         
        assert!(rn.move_pivots_right(0, 2) == true);
        for i in 0..R_CAPACITY{
            if i == 2{
                assert!(rn.pivot[2] == M::Some(6));
            }
            else{
                assert!(rn.pivot[i] == M::None);
            }
        }

        let testOne: [usize; R_CAPACITY] = [1, 2, 6, 7, 0, 17, 19, 0];  

        rn.pivot[0] = M::Some(1);
        rn.pivot[1] = M::Some(2); 
        rn.pivot[3] = M::Some(7);
        rn.pivot[4] = M::Some(17);
        rn.pivot[5] = M::Some(19);
        rn.pivot[6] = M::None;
        rn.pivot[7] = M::None;
        
        assert!(rn.move_pivots_right(4, 1) == true);


        for i in 0..R_CAPACITY{
            if testOne[i] == 0{
                assert!(rn.pivot[i] == M::None);
            }
            else{
                println!("{}", i);
                assert!(rn.pivot[i] == M::Some(testOne[i]));
            }
        }
        
        let testTwo: [usize; R_CAPACITY] = [1, 2, 6, 7, 10, 15, 20, 30];


        for i in 0..R_CAPACITY{
            rn.pivot[i] = M::Some(testTwo[i]);
        }

        assert!(rn.move_pivots_right(4,1) == false);

        let testThree: [usize; R_CAPACITY] = [1, 2, 6, 7, 10, 15, 0, 0];

        for i in 0.. R_CAPACITY-1{
            if testThree[i] == 0{
                rn.pivot[i] = M::None;
            }
            else{
                rn.pivot[i] = M::Some(testThree[i]);
            }            
        }

        assert!(rn.move_pivots_right(2, 3) == false);

    }
}
