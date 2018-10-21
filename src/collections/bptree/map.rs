use crossbeam_epoch as epoch;
use crossbeam_epoch::{Atomic, Guard, Owned, Shared};
use std::collections::LinkedList;
use std::ptr;
use std::sync::atomic::Ordering::{Acquire, Release};
use std::sync::{Mutex, MutexGuard};

use super::node::{BptreeBranch, BptreeLeaf, BptreeNode};

struct BptreeMap<K, V> {
    write: Mutex<()>,
    // Shared root txn?
    active: Atomic<BptreeTxn<K, V>>,
}

// Does bsttxn impl copy?
struct BptreeTxn<K, V> {
    tid: u64,
    root: *mut BptreeNode<K, V>,
    length: usize,
    // Contains garbage lists?
    // How can we set the garbage list of the former that we
    // copy from? Unsafe mut on the active? Mutex on the garbage list?
    // Cell of sometype?
    owned: LinkedList<*mut BptreeNode<K, V>>,
}

struct BptreeWriteTxn<'a, K: 'a, V: 'a> {
    txn: BptreeTxn<K, V>,
    caller: &'a BptreeMap<K, V>,
    _mguard: MutexGuard<'a, ()>,
}

struct BptreeReadTxn<K, V> {
    txn: *const BptreeTxn<K, V>,
    _guard: Guard,
}

// Do I even need an error type?
pub enum BptreeErr {
    Unknown,
}

impl<K, V> BptreeMap<K, V>
where
    K: Clone + PartialEq,
    V: Clone,
{
    pub fn new() -> Self {
        let new_root = Box::new(BptreeNode::new_leaf(0));
        // Create the root txn as empty tree.
        let btxn = BptreeTxn {
            tid: 0,
            // root: None,
            root: Box::into_raw(new_root) as *mut _,
            length: 0,
            owned: LinkedList::new(),
        };
        // Now push the new txn to our Bptree
        BptreeMap {
            write: Mutex::new(()),
            active: Atomic::new(btxn),
        }
    }

    fn commit(&self, new_txn: BptreeTxn<K, V>) -> Result<(), BptreeErr> {
        Ok(())
    }

    pub fn begin_write_txn(&self) -> BptreeWriteTxn<K, V> {
        let mguard = self.write.lock().unwrap();
        let guard = epoch::pin();

        let cur_shared = self.active.load(Acquire, &guard);

        BptreeWriteTxn {
            txn: unsafe {
                // This clones the txn, it increments the tid for us!
                cur_shared.deref().clone()
            },
            caller: self,
            _mguard: mguard,
        }
    }

    pub fn begin_read_txn(&self) -> BptreeReadTxn<K, V> {
        let guard = epoch::pin();

        let cur = {
            let c = self.active.load(Acquire, &guard);
            c.as_raw()
        };

        BptreeReadTxn {
            txn: cur,
            _guard: guard,
        }
    }
}

impl<K, V> BptreeTxn<K, V>
where
    K: Clone + PartialEq,
    V: Clone,
{
    #[inline(always)]
    fn search(&self, key: &K) -> Option<&V> {
        None
    }

    #[inline(always)]
    fn insert(&mut self, key: K, value: V) -> Result<(), BptreeErr> {
        /* Recursively insert. */
        /* This is probably an unsafe .... */
        unsafe {
            (*self.root).insert(key, value).and_then(|nr: *mut _| {
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
        let new_root = Box::new(BptreeNode::new_leaf(self.tid));
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

impl<K, V> Clone for BptreeTxn<K, V> {
    fn clone(&self) -> Self {
        BptreeTxn {
            tid: self.tid + 1,
            // Copies the root
            root: self.root,
            length: self.length,
            owned: LinkedList::new(),
        }
    }
}

impl<K, V> BptreeReadTxn<K, V>
where
    K: Clone + PartialEq,
    V: Clone,
{
    pub fn search(&self, key: &K) -> Option<&V> {
        unsafe { (*self.txn).search(key) }
    }

    pub fn len(&self) -> usize {
        unsafe { (*self.txn).len() }
    }
}

// This is really just a gateway wrapper to the bsttxn fns.
impl<'a, K, V> BptreeWriteTxn<'a, K, V>
where
    K: Clone + PartialEq,
    V: Clone,
{
    pub fn search(&self, key: &K) -> Option<&V> {
        self.txn.search(key)
    }

    pub fn insert(&mut self, key: K, value: V) -> Result<(), BptreeErr> {
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

    pub fn commit(mut self) -> Result<(), BptreeErr> {
        self.caller.commit(self.txn)
    }
}

#[cfg(test)]
mod tests {
    use super::BptreeMap;

    #[test]
    fn test_insert_search_single() {
        let mut bst: BptreeMap<i64, i64> = BptreeMap::new();

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
            // assert!(wrtxn_a.search(&1) == Some(&1));

            wrtxn_a.commit();
        }

        // original read should still show 0 len.
        assert!(rotxn_a.len() == 0);
        assert!(rotxn_a.search(&0) == None);
        assert!(rotxn_a.search(&1) == None);
        // New read should show 1 len.
        let rotxn_b = bst.begin_read_txn();
        // assert!(rotxn_b.len() == 1);
        assert!(rotxn_b.search(&0) == None);
        // assert!(rotxn_b.search(&1) == Some(&1));
        // Read txn goes out of scope here.
    }
}
