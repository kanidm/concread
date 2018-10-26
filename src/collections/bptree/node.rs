use super::map::BptreeErr;
use std::ptr;

const CAPACITY: usize = 5;
const L_CAPACITY: usize = CAPACITY + 1;

pub struct BptreeLeaf<K, V> {
    /* These options get null pointer optimised for us :D */
    key: [Option<K>; CAPACITY],
    value: [Option<V>; CAPACITY],
    parent: *mut BptreeNode<K, V>,
    parent_idx: u16,
    capacity: u16,
    tid: u64,
}

pub struct BptreeBranch<K, V> {
    key: [Option<K>; CAPACITY],
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
                key: [None, None, None, None, None],
                value: [None, None, None, None, None],
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
                key: [Some(key), None, None, None, None],
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
    }

    // Should this be a reference?
    pub fn remove(&mut self, key: &K) -> Option<(K, V)> {
        /* If present, remove */
        /* Else nothing, no-op */
        None
    }

    /* Return if the node is valid */
    fn verify() -> bool {
        false
    }

    fn map_nodes() -> () {}
}

#[cfg(test)]
mod tests {
    use super::{BptreeBranch, BptreeLeaf, BptreeNode};

    #[test]
    fn test_node_basic() {
        
    }
}
