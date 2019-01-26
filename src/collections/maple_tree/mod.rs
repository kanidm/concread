
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

struct SparseLeaf<K, V> {
    // This is an unsorted set of K, V pairs. Insert just appends (if possible),
    //, remove does NOT move the slots. On split, we sort-compact then move some
    // values (if needed).
    key: [M<K>; CAPACITY],
    value: [M<V>; CAPACITY],
}

struct DenseLeaf<V> {
    value: [M<V>; D_CAPACITY],
}

struct RangeLeaf<K, V> {
    pivot: [M<K>; R_CAPACITY],
    value: [M<V>; CAPACITY],
}

struct RangeBranch<K, V> {
    // Implied Pivots
    // Cap - 2
    pivot: [M<K>; R_CAPACITY],
    links: [M<*mut Node<K, V>>; CAPACITY],
}

// When K: SliceIndex, allow Dense
// When K: Binary, allow Range.

pub enum NodeTag<K, V> {
    SL(SparseLeaf<K, V>),
    DL(DenseLeaf<V>),
    RL(RangeLeaf<K, V>),
    RB(RangeBranch<K, V>),
}

struct Node<K, V> {
    tid: u64,
    // checksum: u32,
    inner: NodeTag<K, V>,
}


impl<K, V> SparseLeaf<K, V> {
    pub fn new() -> Self {
        unimplemented!();
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<()> {
        unimplemented!();
    }

    pub fn update(&mut self, k: K, v: V) -> Option<V> {
        unimplemented!();
    }

    pub fn remove(&mut self, k: &K) -> Option<V> {
        unimplemented!();
    }
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


