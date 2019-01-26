use super::map::BptreeErr;
use std::ptr;
use std::cmp::PartialOrd;

const CAPACITY: usize = 5;
const L_CAPACITY: usize = CAPACITY + 1;

// We have to define our own "Option" style type to provide correct ording with PartialOrd
// as Option puts None before Some.

#[derive(PartialEq, PartialOrd, Clone, Eq, Ord, Debug, Hash)]
enum OptionNode<T> {
    Some(T),
    None,
}

pub struct BptreeLeaf<K, V> {
    /* These options get null pointer optimised for us :D */
    key: [OptionNode<K>; CAPACITY],
    value: [OptionNode<V>; CAPACITY],
    parent: *mut BptreeNode<K, V>,
    parent_idx: u16,
    capacity: u16,
    tid: u64,
}

pub struct BptreeBranch<K, V> {
    key: [OptionNode<K>; CAPACITY],
    links: [*mut BptreeNode<K, V>; L_CAPACITY],
    parent: *mut BptreeNode<K, V>,
    parent_idx: u16,
    capacity: u16,
    tid: u64,
}

pub enum BptreeNode<K, V> {
    Leaf { inner: BptreeLeaf<K, V> },
    Branch { inner: BptreeBranch<K, V> },
}

impl<K, V> BptreeNode<K, V>
where
    K: Clone + PartialEq,
    V: Clone,
{
    pub fn new_leaf(tid: u64) -> Self {
        BptreeNode::Leaf {
            inner: BptreeLeaf {
                key: [OptionNode::None, OptionNode::None, OptionNode::None, OptionNode::None, OptionNode::None],
                value: [OptionNode::None, OptionNode::None, OptionNode::None, OptionNode::None, OptionNode::None],
                parent: ptr::null_mut(),
                parent_idx: 0,
                capacity: 0,
                tid: tid,
            },
        }
    }

    fn new_branch(
        key: K,
        left: *mut BptreeNode<K, V>,
        right: *mut BptreeNode<K, V>,
        tid: u64,
    ) -> Self {
        BptreeNode::Branch {
            inner: BptreeBranch {
                key: [OptionNode::Some(key), OptionNode::None, OptionNode::None, OptionNode::None, OptionNode::None],
                links: [
                    left,
                    right,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                ],
                parent: ptr::null_mut(),
                parent_idx: 0,
                capacity: 1,
                tid: tid,
            },
        }
    }

    // Recurse and search.
    pub fn search(&self, key: &K) -> Option<&V> {
        unimplemented!();
        match self {
            &BptreeNode::Leaf { ref inner } => None,
            &BptreeNode::Branch { ref inner } => None,
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Result<*mut BptreeNode<K, V>, BptreeErr> {
        /* Should we auto split? */
        Ok(ptr::null_mut())
    }

    pub fn update(&mut self, key: K, value: V) {
        /* If not present, insert */
        /* If present, replace */
        unimplemented!()
    }

    // Should this be a reference?
    pub fn remove(&mut self, key: &K) -> Option<(K, V)> {
        /* If present, remove */
        /* Else nothing, no-op */
        unimplemented!();
        None
    }

    /* Return if the node is valid */
    fn verify() -> bool {
        unimplemented!();
        false
    }

    fn map_nodes() -> () {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::{BptreeBranch, BptreeLeaf, BptreeNode};

    #[test]
    fn test_node_leaf_basic() {
        let mut leaf: BptreeNode<u64, u64> = BptreeNode::new_leaf(0);

        // Insert
        assert!(leaf.insert(0, 0).is_ok())

        // Delete
        // Search
    }

    #[test]
    fn test_node_leaf_split() {
    }
}
