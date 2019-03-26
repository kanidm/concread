
use std::fmt::Debug;
use std::mem;

// Number of k,v in sparse, and number of range values/links.
const CAPACITY: usize = 8;
// Number of pivots in range
const R_CAPACITY: usize = CAPACITY - 1;
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
    pivot: [M<K>; R_CAPACITY],
    value: [M<V>; CAPACITY],
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

impl<K, V> SparseLeaf<K, V> 
    where K: PartialEq + Debug,
          V: Debug,
{
    pub fn new() -> Self {
        SparseLeaf {
            key: [M::None, M::None, M::None, M::None, M::None, M::None, M::None, M::None],
            value: [M::None, M::None, M::None, M::None, M::None, M::None, M::None, M::None],
        }
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<()> {
        // Just find a free slot and insert
        for i in 0..CAPACITY {
            println!("insert: {:?}", self.key[i]);
            if ! self.key[i].is_some() {
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

        // If there is no slot, return a failure?

        unimplemented!();
    }

    pub fn update(&mut self, k: K, v: V) -> Option<V> {
        unimplemented!();
    }

    pub fn remove(&mut self, k: &K) -> Option<V> {
        for i in 0..CAPACITY {
            println!("remove: {:?}", self.key[i]);

            if self.key[i] == k {
                let mut nk = M::None;
                let mut nv = M::None;

                mem::swap(&mut self.key[i], &mut nk);
                mem::swap(&mut self.value[i], &mut nv);
                return Some(nv);
            }
        }
        unimplemented!();
    }

    // We need to sort *just before* we split if required.
}


#[cfg(test)]
mod tests {
    use super::{SparseLeaf};

    #[test]
    fn test_sparse_leaf_basic() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();
        // insert
        sl.insert(0, 0);
        // remove
        sl.remove(&0);
        // Remove non-existant
        sl.remove(&0);

        sl.insert(1, 1);
        sl.insert(2, 2);

        // Insert duplicate
        sl.insert(2, 2); // error?

        // update inplace.
        sl.update(2, 3);

        // split?
        // compact/sort
        // verify
        // clone
        // eq?
        // pass
    }

}


