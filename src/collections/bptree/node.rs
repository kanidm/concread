use super::map::BptreeErr;
use std::ptr;

const CAPACITY: usize = 5;
const L_CAPACITY: usize = CAPACITY + 1;

pub enum BptreeNodeInner<K, V> {
    Leaf {
        value: [Option<V>; CAPACITY],
    },
    Branch {
        links: [*mut BptreeNode<K, V>; L_CAPACITY],
    },
}

pub struct BptreeNode<K, V> {
    key: [Option<K>; CAPACITY],
    inner: BptreeNodeInner<K, V>,
    parent: *mut BptreeNode<K, V>,
    parent_idx: u16,
    capacity: u16,
    tid: u64,
}

impl<K, V> BptreeNode<K, V>
where
    K: Clone + PartialEq + Ord,
    V: Clone,
{
    pub fn new_leaf(tid: u64) -> Self {
        BptreeNode {
            key: [None, None, None, None, None],
            inner: BptreeNodeInner::Leaf {
                value: [None, None, None, None, None],
            },
            parent: ptr::null_mut(),
            parent_idx: 0,
            capacity: 0,
            tid: tid,
        }
    }

    fn new_branch(
        key: K,
        left: *mut BptreeNode<K, V>,
        right: *mut BptreeNode<K, V>,
        tid: u64,
    ) -> Self {
        BptreeNode {
            key: [Some(key), None, None, None, None],
            inner: BptreeNodeInner::Branch {
                links: [
                    left,
                    right,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                ],
            },
            parent: ptr::null_mut(),
            parent_idx: 0,
            capacity: 1,
            tid: tid,
        }
    }

    // Recurse and search.
    pub fn search(&self, key: &K) -> Option<&V> {
        None
    }

    pub fn contains(&self, key: &K) -> bool {
        self.search(key).is_some()
    }

    // Is there really a condition where we would actually fail to insert?
    // if K already exists?
    pub fn insert(&mut self, key: K, value: V) -> Result<*mut BptreeNode<K, V>, BptreeErr> {
        /* Should we auto split? */
        match self.key.binary_search(&Some(key)) {
            Ok(idx) => {
                println!("{:?}", idx);
            }
            Err(idx) => {
                println!("{:?}", idx);
            }
        };


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
    use super::BptreeNode;

    #[test]
    fn test_node_leaf() {
        let mut leaf: BptreeNode<u64, u64> = BptreeNode::new_leaf(0);
        // Insert values
        let r1 = leaf.insert(4, 0);
        assert!(r1.is_ok());
        assert!(leaf.contains(&4));

        let r2 = leaf.insert(5, 0);
        assert!(r2.is_ok());
        assert!(leaf.contains(&5));
        // How do I hand the duplicate without update?
        let r3 = leaf.insert(5, 0);
        assert!(r3.is_err());

        let r4 = leaf.insert(3, 0);
        assert!(r4.is_ok());
        assert!(leaf.contains(&3));
        // remove values
        //  from tail
        //  from head
        //  from centre
        //  what happens when low cap and no parent?
        // what happens when full? Do we split the leaf?
        // verify the node
    }
}
