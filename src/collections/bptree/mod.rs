
use crossbeam_epoch as epoch;
use crossbeam_epoch::{Atomic, Owned, Shared, Guard};
use std::ptr;
use std::sync::atomic::Ordering::{Release, Acquire};
use std::sync::{Mutex, MutexGuard};
use std::collections::LinkedList;

const CAPACITY: usize = 5;
const L_CAPACITY: usize = CAPACITY + 1;

struct Bst<K, V> {
    write: Mutex<()>,
    // Shared root txn?
    active: Atomic<BstTxn<K, V>>,
}

// Does bsttxn impl copy?
struct BstTxn<K, V> {
    tid: u64,
    root: *mut BstNode<K, V>,
    length: usize,
    // Contains garbage lists?
    // How can we set the garbage list of the former that we
    // copy from? Unsafe mut on the active? Mutex on the garbage list?
    // Cell of sometype?
    owned: LinkedList<*mut BstNode<K, V>>,
}

struct BstWriteTxn<'a, K: 'a, V: 'a> {
    txn: BstTxn<K, V>,
    caller: &'a Bst<K, V>,
    _mguard: MutexGuard<'a, ()>
}

struct BstReadTxn<K, V> {
    txn: *const BstTxn<K, V>,
    _guard: Guard,
}

struct BstLeaf<K, V> {
    /* These options get null pointer optimised for us :D */
    key: [Option<K>; CAPACITY],
    value: [Option<V>; CAPACITY],
    parent: *mut BstNode<K, V>,
    parent_idx: u16,
    capacity: u16,
    tid: u64,
}

struct BstBranch<K, V> {
    key: [Option<K>; CAPACITY],
    links: [*mut BstNode<K, V>; L_CAPACITY],
    parent: *mut BstNode<K, V>,
    parent_idx: u16,
    capacity: u16,
    tid: u64,
}

// Do I even need an error type?
enum BstErr {
    Unknown,
}

enum BstNode<K, V> {
    Leaf {
        inner: BstLeaf<K, V>
    },
    Branch {
        inner: BstBranch<K, V>
    }
}

impl <K, V> BstNode<K, V> where
    K: Clone + PartialEq,
    V: Clone,
{
    pub fn new_leaf(tid: u64) -> Self {
        BstNode::Leaf {
            inner: BstLeaf {
                key: [None, None, None, None, None],
                value: [None, None, None, None, None],
                parent:  ptr::null_mut(),
                parent_idx: 0,
                capacity: 0,
                tid: tid
            }
        }
    }

    fn new_branch(key: K, left: *mut BstNode<K, V>, right: *mut BstNode<K, V>, tid: u64) -> Self {
        BstNode::Branch {
            inner: BstBranch {
                key: [Some(key), None, None, None, None],
                links: [left, right, ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut()],
                parent: ptr::null_mut(),
                parent_idx: 0,
                capacity: 1,
                tid: tid
            }
        }
    }

    // Recurse and search.
    pub fn search(&self, key: &K) -> Option<&V> {
        match self {
            &BstNode::Leaf { ref inner } => {
                None
            }
            &BstNode::Branch { ref inner } => {
                None
            }
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Result<*mut BstNode<K, V>, BstErr> {
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

    fn map_nodes() -> () {
    }
}

impl<K, V> Bst<K, V> where
    K: Clone + PartialEq,
    V: Clone,
{
    pub fn new() -> Self {
        let new_root = Box::new(
            BstNode::new_leaf(0)
        );
        // Create the root txn as empty tree.
        let btxn = BstTxn {
            tid: 0,
            // root: None,
            root: Box::into_raw(new_root) as *mut _,
            length: 0,
            owned: LinkedList::new(),
        };
        // Now push the new txn to our Bst
        Bst {
            write: Mutex::new(()),
            active: Atomic::new(btxn),
        }
    }

    fn commit(&self, new_txn: BstTxn<K, V>) -> Result<(), BstErr> {
        Ok(())
    }

    pub fn begin_write_txn(&self) -> BstWriteTxn<K, V> {
        let mguard = self.write.lock().unwrap();
        let guard = epoch::pin();

        let cur_shared = self.active.load(Acquire, &guard);

        BstWriteTxn {
            txn: unsafe {
                // This clones the txn, it increments the tid for us!
                cur_shared.deref().clone()
            },
            caller: self,
            _mguard: mguard,
        }
    }

    pub fn begin_read_txn(&self) -> BstReadTxn<K, V> {
        let guard = epoch::pin();

        let cur = {
            let c = self.active.load(Acquire, &guard);
            c.as_raw()
        };

        BstReadTxn {
            txn: cur,
            _guard: guard,
        }
    }

}

impl<K, V> BstTxn<K, V> where
    K: Clone + PartialEq,
    V: Clone,
{
    #[inline(always)]
    fn search(&self, key: &K) -> Option<&V> {
        None
    }

    #[inline(always)]
    fn insert(&mut self, key: K, value: V) -> Result<(), BstErr> {
        /* Recursively insert. */
        /* This is probably an unsafe .... */
        unsafe {
            (*self.root).insert(key, value).and_then(|nr: *mut _ | {
                self.length += 1;
                self.root = nr;
                Ok(())
            })
        }
    }

    #[inline(always)]
    fn clear(&mut self) {
        // Because this changes root from Some to None, it moves ownership
        // of root to this function, and all it's descendants that are dropped.

        /* !!!!!!!!!!!!!!!!!! TAKE OWNERSHIP OF ROOT AND FREE IT !!!!!!!!!!!!!!!!! */

        // With EBR you need to walk the tree and mark everything to be dropped.
        // Perhaps just the root needs EBR marking?
        let new_root = Box::new(
            BstNode::new_leaf(self.tid)
        );
        self.root = Box::into_raw(new_root);
        self.length = 0;
    }

    #[inline(always)]
    fn remove(&mut self, key: &K) -> Option<(K, V)> {
        None
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.length
    }
}

impl<K, V> Clone for BstTxn<K, V> {
    fn clone(&self) -> Self {
        BstTxn {
            tid: self.tid + 1,
            // Copies the root
            root: self.root,
            length: self.length,
            owned: LinkedList::new(),
        }
    }
}

impl<K, V> BstReadTxn<K, V> where
    K: Clone + PartialEq,
    V: Clone,
{
    pub fn search(&self, key: &K) -> Option<&V> {
        unsafe {
            (*self.txn).search(key)
        }
    }

    pub fn len(&self) -> usize {
        unsafe {
            (*self.txn).len()
        }
    }
}

// This is really just a gateway wrapper to the bsttxn fns.
impl<'a, K, V> BstWriteTxn<'a, K, V> where
    K: Clone + PartialEq,
    V: Clone,
{
    pub fn search(&self, key: &K) -> Option<&V> {
        self.txn.search(key)
    }

    pub fn insert(&mut self, key: K, value: V) -> Result<(), BstErr> {
        self.txn.insert(key, value)
    }

    pub fn clear(&mut self) {
        self.txn.clear()
    }

    /// Delete the value
    pub fn remove(&mut self, key: &K) -> Option<(K, V)> {
        self.txn.remove(key)
    }

    pub fn len(&self) -> usize {
        self.txn.len()
    }

    pub fn commit(mut self) -> Result<(), BstErr> {
        self.caller.commit(self.txn)
    }
}



#[cfg(test)]
mod tests {
    use super::Bst;
    #[test]
    fn test_node_basic() {
        // Test that simple operations on nodes work as expected
    }

    #[test]
    fn test_insert_search_single() {
        let mut bst: Bst<i64, i64> = Bst::new();

        // First, take a read_txn and check it's length.
        let rotxn_a = bst.begin_read_txn();
        assert!(rotxn_a.len() == 0);
        assert!(rotxn_a.search(&0) == None);
        assert!(rotxn_a.search(&1) == None);

        {
            let mut wrtxn_a = bst.begin_write_txn();

            wrtxn_a.insert(1, 1);
            assert!(wrtxn_a.len() == 1);
            assert!(wrtxn_a.search(&0) == None);
            assert!(wrtxn_a.search(&1) == Some(&1));

            wrtxn_a.commit();
        }

        // original read should still show 0 len.
        assert!(rotxn_a.len() == 0);
        assert!(rotxn_a.search(&0) == None);
        assert!(rotxn_a.search(&1) == None);
        // New read should show 1 len.
        let rotxn_b = bst.begin_read_txn();
        assert!(rotxn_b.len() == 1);
        assert!(rotxn_b.search(&0) == None);
        assert!(rotxn_b.search(&1) == Some(&1));
        // Read txn goes out of scope here.
    }
}


