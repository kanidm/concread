//! See the documentation for `BptreeMap`

use super::cursor::CursorReadOps;
use super::cursor::{CursorRead, CursorWrite};
use super::node::{ABNode, Node};
use parking_lot::{Mutex, MutexGuard};
use std::fmt::Debug;

/// A concurrently readable map based on a modified B+Tree structure.
///
/// This structure can be used in locations where you would otherwise us
/// `RwLock<BTreeMap>` or `Mutex<BTreeMap>`.
///
/// This is a concurrently readable structure, meaning it has transactional
/// properties. Writers are serialised (one after the other), and readers
/// can exist in parallel with stable views of the structure at a point
/// in time.
///
/// This is achieved through the use of COW or MVCC. As a write occurs
/// subsets of the tree are cloned into the writer thread and then commited
/// later. This may cause memory usage to increase in exchange for a gain
/// in concurrent behaviour.
///
/// Transactions can be rolled-back (aborted) without penalty by dropping
/// the `BptreeMapWriteTxn` without calling `commit()`.
pub struct BptreeMap<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    write: Mutex<()>,
    active: Mutex<(ABNode<K, V>, usize)>,
}

/// An active read transaction over a `BptreeMap`. The data in this tree
/// is guaranteed to not change and will remain consistent for the life
/// of this transaction.
pub struct BptreeMapReadTxn<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    work: CursorRead<K, V>,
}

/// An active write transaction for a `BptreeMap`. The data in this tree
/// may be modified exclusively through this transaction without affecting
/// readers. The write may be rolledback/aborted by dropping this guard
/// without calling `commit()`. Once `commit()` is called, readers will be
/// able to access and percieve changes in new transactions.
pub struct BptreeMapWriteTxn<'a, K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    work: CursorWrite<K, V>,
    caller: &'a BptreeMap<K, V>,
    _guard: MutexGuard<'a, ()>,
}

impl<K: Clone + Ord + Debug, V: Clone> BptreeMap<K, V> {
    /// Construct a new concurrent tree
    pub fn new() -> Self {
        BptreeMap {
            write: Mutex::new(()),
            active: Mutex::new((Node::new_ableaf(0), 0)),
        }
    }

    /// Initiate a read transaction for the tree, concurrent to any
    /// other readers or writers.
    pub fn read(&self) -> BptreeMapReadTxn<K, V> {
        let rguard = self.active.lock();
        BptreeMapReadTxn {
            work: CursorRead::new(rguard.0.clone(), rguard.1),
        }
        // rguard is dropped, the ABNode lives on!
    }

    /// Initiate a write transaction for the tree, exclusive to this
    /// writer, and concurrently to all existing reads.
    pub fn write(&self) -> BptreeMapWriteTxn<K, V> {
        /* Take the exclusive write lock first */
        let mguard = self.write.lock();
        /* Now take a ro-txn to get the data copied */
        let rguard = self.active.lock();
        /*
         * Take a ref to the root, we want to minimise our time in the.
         * active lock. We could do a full clone here but that would trigger
         * node-width worth of atomics, and if the write is dropped without
         * action we've save a lot of cycles.
         */
        let (data, length): (ABNode<K, V>, usize) = rguard.clone();
        /* Setup the cursor that will work on the tree */
        let cursor = CursorWrite::new(data, length);

        /* Now build the write struct */
        BptreeMapWriteTxn {
            work: cursor,
            caller: self,
            _guard: mguard,
        }
        /* rguard dropped here */
    }

    fn commit(&self, newdata: (ABNode<K, V>, usize)) {
        let mut rwguard = self.active.lock();
        *rwguard = newdata;
    }
}

impl<'a, K: Clone + Ord + Debug, V: Clone> BptreeMapWriteTxn<'a, K, V> {
    // == RO methods

    /// Retrieve a value from the tree. If the value exists, a reference is returned
    /// as `Some(&V)`, otherwise if not present `None` is returned.
    pub fn get(&'a self, k: &'a K) -> Option<&'a V> {
        self.work.search(k)
    }

    /// Assert if a key exists in the tree.
    pub fn contains_key(&self, k: &K) -> bool {
        self.work.contains_key(k)
    }

    /// Returns the current number of k:v pairs in the tree
    pub fn len(&self) -> usize {
        self.work.len()
    }

    // is_empty

    // (adv) range

    /// Iterator over `(&K, &V)` of the set
    pub fn iter(&self) -> Iter {
        self.work.kv_iter();
    }

    // (adv) keys

    // (adv) values

    // == RW methods

    /// Reset this tree to an empty state. As this is within the transaction this
    /// change only takes effect once commited.
    pub fn clear(&mut self) {
        self.work.clear()
    }

    // get_mut

    /// Insert or update a value by key. If the value previously existed it is returned
    /// as `Some(V)`. If the value did not previously exist this returns `None`.
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.work.insert(k, v)
    }

    // remove

    // split_off

    // ADVANCED
    // append (join two sets)

    // range_mut

    // entry

    // iter_mut

    /// Compact the tree structure if the density is below threshold, yielding improved search
    /// performance and lowering memory footprint.
    ///
    /// Many tree structures attempt to remain "balanced" consuming excess memory to allow
    /// amortizing cost and distributing values over the structure. Generally this means that
    /// a classic B+Tree has only ~55% to ~66% occupation of it's leaves (varying based on their
    /// width). The branches have a similar layout.
    ///
    /// Given linear (ordered) inserts this structure will have 100% utilisation at the leaves
    /// and between ~66% to ~75% occupation through out the branches. If you built this from a
    /// iterator, this is probably the case you have here!
    ///
    /// However under random insert loads we tend toward ~60% utilisation similar to the classic
    /// B+tree. Given reverse key order inserts we have poor behaviour with about ~%20 occupation.
    ///
    /// Instead of paying a cost in time and memory on every insert to achieve the "constant" %60
    /// loading, we prefer to minimise the work in the tree in favour of compacting the structure
    /// when required. This is especially visible given that most workloads are linear or random
    /// and we save time on these workloads by not continually rebalancing.
    ///
    /// If you call this function, and the current occupation is less than 50% the tree will be
    /// rebalanced. This may briefly consume more ram, but will achieve a near ~100% occupation
    /// of k:v in the tree, with a reduction in leaves and branches - basically it makes search
    /// faster.
    pub fn compact(&mut self) {
        let (l, m) = self.work.tree_density();
        if (m / l) > 1 {
            self.work.compact()
        }
    }

    /// Initiate a compaction of the tree regardless of it's density or loading factors.
    ///
    /// You probably should use `compact()` instead.
    ///
    /// See `compact()` for the logic of why this exists.
    pub fn compact_force(&mut self) {
        self.work.compact()
    }

    /// Commit the changes from this write transaction. Readers after this point
    /// will be able to percieve these changes.
    ///
    /// To abort, just do not call this function.
    pub fn commit(self) {
        self.caller.commit(self.work.finalise())
    }
}

#[cfg(test)]
mod tests {
    use super::BptreeMap;

    #[test]
    fn test_bptree_map_basic_write() {
        let bptree: BptreeMap<usize, usize> = BptreeMap::new();
        {
            let mut bpwrite = bptree.write();
            // We should be able to insert.
            bpwrite.insert(0, 0);
            bpwrite.insert(1, 1);
            assert!(bpwrite.get(&0) == Some(&0));
            assert!(bpwrite.get(&1) == Some(&1));
            bpwrite.commit();
        }
        {
            // Do a clear, but roll it back.
            let mut bpwrite = bptree.write();
            bpwrite.clear();
            // DO NOT commit, this triggers the rollback.
        }
        {
            let bpwrite = bptree.write();
            assert!(bpwrite.get(&0) == Some(&0));
            assert!(bpwrite.get(&1) == Some(&1));
        }
    }
}
