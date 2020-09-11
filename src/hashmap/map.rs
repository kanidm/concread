//! See the documentation for `HashMap` module.

// TODO:
#![allow(clippy::implicit_hasher)]

use ahash::AHasher;
use std::borrow::Borrow;
// use std::collections::hash_map::DefaultHasher;
use super::cursor::CursorReadOps;
use super::cursor::{CursorRead, CursorWrite, SuperBlock};
use super::iter::*;
use super::node::Datum;
use parking_lot::{Mutex, MutexGuard};
use rand::Rng;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::sync::Arc;

// #[cfg(feature = "simd_support")] use packed_simd::*;
// #[cfg(feature = "simd_support")]

macro_rules! hash_key {
    ($k:expr, $key1:expr, $key2:expr) => {{
        // let mut hasher = DefaultHasher::new();
        let mut hasher = AHasher::new_with_keys($key1, $key2);
        $k.hash(&mut hasher);
        hasher.finish()
    }};
}

/// A concurrently readable map based on a modified B+Tree structured with fast
/// parallel hashed key lookup.
///
/// This structure can be used in locations where you would otherwise us
/// `RwLock<HashMap>` or `Mutex<HashMap>`.
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
/// the `HashMapWriteTxn` without calling `commit()`.
pub struct HashMap<K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    write: Mutex<()>,
    active: Mutex<Arc<SuperBlock<K, V>>>,
    key1: u128,
    key2: u128,
}

unsafe impl<K: Hash + Eq + Clone + Debug, V: Clone> Send for HashMap<K, V> {}
unsafe impl<K: Hash + Eq + Clone + Debug, V: Clone> Sync for HashMap<K, V> {}

/// An active read transaction over a `HashMap`. The data in this tree
/// is guaranteed to not change and will remain consistent for the life
/// of this transaction.
pub struct HashMapReadTxn<'a, K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    _caller: &'a HashMap<K, V>,
    pin: Arc<SuperBlock<K, V>>,
    work: CursorRead<K, V>,
    key1: u128,
    key2: u128,
}

/// An active write transaction for a `HashMap`. The data in this tree
/// may be modified exclusively through this transaction without affecting
/// readers. The write may be rolledback/aborted by dropping this guard
/// without calling `commit()`. Once `commit()` is called, readers will be
/// able to access and percieve changes in new transactions.
pub struct HashMapWriteTxn<'a, K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    work: CursorWrite<K, V>,
    caller: &'a HashMap<K, V>,
    _guard: MutexGuard<'a, ()>,
    key1: u128,
    key2: u128,
}

enum SnapshotType<'a, K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    R(&'a CursorRead<K, V>),
    W(&'a CursorWrite<K, V>),
}

/// A point-in-time snapshot of the tree from within a read OR write. This is
/// useful for building other transactional types ontop of this structure, as
/// you need a way to downcast both HashMapReadTxn or HashMapWriteTxn to
/// a singular reader type for a number of get_inner() style patterns.
///
/// This snapshot IS safe within the read thread due to the nature of the
/// implementation borrowing the inner tree to prevent mutations within the
/// same thread while the read snapshot is open.
pub struct HashMapReadSnapshot<'a, K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    work: SnapshotType<'a, K, V>,
    key1: u128,
    key2: u128,
}

impl<K: Hash + Eq + Clone + Debug, V: Clone> Default for HashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone + Debug, V: Clone> HashMap<K, V> {
    /// Construct a new concurrent hashmap
    pub fn new() -> Self {
        HashMap {
            write: Mutex::new(()),
            active: Mutex::new(Arc::new(SuperBlock::default())),
            key1: rand::thread_rng().gen::<u128>(),
            key2: rand::thread_rng().gen::<u128>(),
        }
    }

    /// Initiate a read transaction for the Hashmap, concurrent to any
    /// other readers or writers.
    pub fn read(&self) -> HashMapReadTxn<K, V> {
        let rguard = self.active.lock();
        let pin = rguard.clone();
        let work = CursorRead::new(pin.as_ref());
        HashMapReadTxn {
            _caller: self,
            pin,
            work,
            key1: self.key1,
            key2: self.key2,
        }
    }

    /// Initiate a write transaction for the map, exclusive to this
    /// writer, and concurrently to all existing reads.
    pub fn write(&self) -> HashMapWriteTxn<K, V> {
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
        let sblock: &SuperBlock<K, V> = rguard.as_ref();
        /* Setup the cursor that will work on the tree */
        let cursor = CursorWrite::new(sblock);
        /* Now build the write struct */
        HashMapWriteTxn {
            work: cursor,
            caller: self,
            _guard: mguard,
            key1: self.key1,
            key2: self.key2,
        }
        /* rguard dropped here */
    }

    /// Attempt to create a new write, returns None if another writer
    /// already exists.
    pub fn try_write(&self) -> Option<HashMapWriteTxn<K, V>> {
        self.write.try_lock().map(|mguard| {
            let rguard = self.active.lock();
            let sblock: &SuperBlock<K, V> = rguard.as_ref();
            let cursor = CursorWrite::new(sblock);
            HashMapWriteTxn {
                work: cursor,
                caller: self,
                _guard: mguard,
                key1: self.key1,
                key2: self.key2,
            }
        })
    }

    fn commit(&self, newdata: SuperBlock<K, V>) {
        // println!("commit wr");
        let mut rwguard = self.active.lock();
        // Now we need to setup the sb pointers properly.
        // The current active SHOULD have a NONE last seen as it's the current
        // tree holder.
        newdata.commit_prep(rwguard.as_ref());

        let arc_newdata = Arc::new(newdata);
        // Now pin the older to this new txn.
        {
            let mut pin_guard = rwguard.as_ref().pin_next.lock();
            *pin_guard = Some(arc_newdata.clone());
        }

        // Now push the new SB.
        *rwguard = arc_newdata;
    }
}

impl<K: Hash + Eq + Clone + Debug, V: Clone> FromIterator<(K, V)> for HashMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let hmap = HashMap::new();
        let mut hmap_write = hmap.write();
        hmap_write.extend(iter);
        hmap_write.commit();
        hmap
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> Extend<(K, V)> for HashMapWriteTxn<'a, K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        iter.into_iter().for_each(|(k, v)| {
            let _ = self.insert(k, v);
        });
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> HashMapWriteTxn<'a, K, V> {
    pub(crate) fn get_txid(&self) -> u64 {
        self.work.get_txid()
    }

    pub(crate) fn prehash<'b, Q: ?Sized>(&'a self, k: &'b Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        hash_key!(k, self.key1, self.key2)
    }

    pub(crate) fn get_prehashed<'b, Q: ?Sized>(&'a self, k: &'b Q, k_hash: u64) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.work.search(k_hash, k)
    }

    /// Retrieve a value from the map. If the value exists, a reference is returned
    /// as `Some(&V)`, otherwise if not present `None` is returned.
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k_hash = hash_key!(k, self.key1, self.key2);
        self.get_prehashed(k, k_hash)
    }

    /// Assert if a key exists in the map.
    pub fn contains_key<'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(k).is_some()
    }

    /// returns the current number of k:v pairs in the tree
    pub fn len(&self) -> usize {
        self.work.len()
    }

    /// Determine if the set is currently empty
    pub fn is_empty(&self) -> bool {
        self.work.len() == 0
    }

    /// Iterator over `(&K, &V)` of the set
    pub fn iter(&self) -> Iter<K, V> {
        self.work.kv_iter()
    }

    /// Iterator over &K
    pub fn values(&self) -> ValueIter<K, V> {
        self.work.v_iter()
    }

    /// Iterator over &V
    pub fn keys(&self) -> KeyIter<K, V> {
        self.work.k_iter()
    }

    /// Reset this map to an empty state. As this is within the transaction this
    /// change only takes effect once commited. Once cleared, you can begin adding
    /// new writes and changes, again, that will only be visible once commited.
    pub fn clear(&mut self) {
        self.work.clear();
    }

    /// Insert or update a value by key. If the value previously existed it is returned
    /// as `Some(V)`. If the value did not previously exist this returns `None`.
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        // Hash the key.
        let k_hash = hash_key!(k, self.key1, self.key2);
        self.work.insert(k_hash, k, v)
    }

    /// Remove a key if it exists in the tree. If the value exists, we return it as `Some(V)`,
    /// and if it did not exist, we return `None`
    pub fn remove(&mut self, k: &K) -> Option<V> {
        let k_hash = hash_key!(k, self.key1, self.key2);
        self.work.remove(k_hash, k)
    }

    /// Get a mutable reference to a value in the tree. This is correctly, and
    /// safely cloned before you attempt to mutate the value, isolating it from
    /// other transactions.
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let k_hash = hash_key!(k, self.key1, self.key2);
        self.work.get_mut_ref(k_hash, k)
    }

    /// This is *unsafe* because changing the key CAN and WILL break hashing, which can
    /// have serious consequences. This API only exists to allow arcache to access the inner
    /// content of the slot to simplify it's API. You should basically never touch this
    /// function as it's the HashMap equivalent of a the demon sphere.
    pub(crate) unsafe fn get_slot_mut(&mut self, k_hash: u64) -> Option<&mut [Datum<K, V>]> {
        self.work.get_slot_mut_ref(k_hash)
    }

    /// Create a read-snapshot of the current map. This does NOT guarantee the map may
    /// not be mutated during the read, so you MUST guarantee that no functions of the
    /// write txn are called while this snapshot is active.
    pub fn to_snapshot(&'a self) -> HashMapReadSnapshot<K, V> {
        HashMapReadSnapshot {
            work: SnapshotType::W(&self.work),
            key1: self.key1,
            key2: self.key2,
        }
    }

    /// Commit the changes from this write transaction. Readers after this point
    /// will be able to percieve these changes.
    ///
    /// To abort (unstage changes), just do not call this function.
    pub fn commit(self) {
        self.caller.commit(self.work.finalise())
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> HashMapReadTxn<'a, K, V> {
    pub(crate) fn get_txid(&self) -> u64 {
        self.work.get_txid()
    }

    pub(crate) fn prehash<'b, Q: ?Sized>(&'a self, k: &'b Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        hash_key!(k, self.key1, self.key2)
    }

    pub(crate) fn get_prehashed<'b, Q: ?Sized>(&'a self, k: &'b Q, k_hash: u64) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.work.search(k_hash, k)
    }

    /// Retrieve a value from the tree. If the value exists, a reference is returned
    /// as `Some(&V)`, otherwise if not present `None` is returned.
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k_hash = hash_key!(k, self.key1, self.key2);
        self.get_prehashed(k, k_hash)
    }

    /// Assert if a key exists in the tree.
    pub fn contains_key<'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(k).is_some()
    }

    /// Returns the current number of k:v pairs in the tree
    pub fn len(&self) -> usize {
        self.work.len()
    }

    /// Determine if the set is currently empty
    pub fn is_empty(&self) -> bool {
        self.work.len() == 0
    }

    /// Iterator over `(&K, &V)` of the set
    pub fn iter(&self) -> Iter<K, V> {
        self.work.kv_iter()
    }

    /// Iterator over &K
    pub fn values(&self) -> ValueIter<K, V> {
        self.work.v_iter()
    }

    /// Iterator over &V
    pub fn keys(&self) -> KeyIter<K, V> {
        self.work.k_iter()
    }

    /// Create a read-snapshot of the current tree.
    /// As this is the read variant, it IS safe, and guaranteed the tree will not change.
    pub fn to_snapshot(&'a self) -> HashMapReadSnapshot<'a, K, V> {
        HashMapReadSnapshot {
            work: SnapshotType::R(&self.work),
            key1: self.key1,
            key2: self.key2,
        }
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> HashMapReadSnapshot<'a, K, V> {
    /// Retrieve a value from the tree. If the value exists, a reference is returned
    /// as `Some(&V)`, otherwise if not present `None` is returned.
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k_hash = hash_key!(k, self.key1, self.key2);
        match self.work {
            SnapshotType::R(work) => work.search(k_hash, k),
            SnapshotType::W(work) => work.search(k_hash, k),
        }
    }

    /// Assert if a key exists in the tree.
    pub fn contains_key<'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(k).is_some()
    }

    /// Returns the current number of k:v pairs in the tree
    pub fn len(&self) -> usize {
        match self.work {
            SnapshotType::R(work) => work.len(),
            SnapshotType::W(work) => work.len(),
        }
    }

    /// Determine if the set is currently empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // (adv) range

    /// Iterator over `(&K, &V)` of the set
    pub fn iter(&self) -> Iter<K, V> {
        match self.work {
            SnapshotType::R(work) => work.kv_iter(),
            SnapshotType::W(work) => work.kv_iter(),
        }
    }

    /// Iterator over &K
    pub fn values(&self) -> ValueIter<K, V> {
        match self.work {
            SnapshotType::R(work) => work.v_iter(),
            SnapshotType::W(work) => work.v_iter(),
        }
    }

    /// Iterator over &V
    pub fn keys(&self) -> KeyIter<K, V> {
        match self.work {
            SnapshotType::R(work) => work.k_iter(),
            SnapshotType::W(work) => work.k_iter(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HashMap;

    #[test]
    fn test_hashmap_basic_write() {
        let hmap: HashMap<usize, usize> = HashMap::new();
        let mut hmap_write = hmap.write();

        hmap_write.insert(10, 10);
        hmap_write.insert(15, 15);

        assert!(hmap_write.contains_key(&10));
        assert!(hmap_write.contains_key(&15));
        assert!(!hmap_write.contains_key(&20));

        assert!(hmap_write.get(&10) == Some(&10));
        {
            let v = hmap_write.get_mut(&10).unwrap();
            *v = 11;
        }
        assert!(hmap_write.get(&10) == Some(&11));

        assert!(hmap_write.remove(&10).is_some());
        assert!(!hmap_write.contains_key(&10));
        assert!(hmap_write.contains_key(&15));

        assert!(hmap_write.remove(&30).is_none());

        hmap_write.clear();
        assert!(!hmap_write.contains_key(&10));
        assert!(!hmap_write.contains_key(&15));
        hmap_write.commit();
    }

    #[test]
    fn test_hashmap_basic_read_write() {
        let hmap: HashMap<usize, usize> = HashMap::new();
        let mut hmap_w1 = hmap.write();
        hmap_w1.insert(10, 10);
        hmap_w1.insert(15, 15);
        hmap_w1.commit();

        let hmap_r1 = hmap.read();
        assert!(hmap_r1.contains_key(&10));
        assert!(hmap_r1.contains_key(&15));
        assert!(!hmap_r1.contains_key(&20));

        let mut hmap_w2 = hmap.write();
        hmap_w2.insert(20, 20);
        hmap_w2.commit();

        assert!(hmap_r1.contains_key(&10));
        assert!(hmap_r1.contains_key(&15));
        assert!(!hmap_r1.contains_key(&20));

        let hmap_r2 = hmap.read();
        assert!(hmap_r2.contains_key(&10));
        assert!(hmap_r2.contains_key(&15));
        assert!(hmap_r2.contains_key(&20));
    }

    #[test]
    fn test_hashmap_basic_read_snapshot() {
        let hmap: HashMap<usize, usize> = HashMap::new();
        let mut hmap_w1 = hmap.write();
        hmap_w1.insert(10, 10);
        hmap_w1.insert(15, 15);

        let snap = hmap_w1.to_snapshot();
        assert!(snap.contains_key(&10));
        assert!(snap.contains_key(&15));
        assert!(!snap.contains_key(&20));
    }

    #[test]
    fn test_hashmap_basic_iter() {
        let hmap: HashMap<usize, usize> = HashMap::new();
        let mut hmap_w1 = hmap.write();
        assert!(hmap_w1.iter().count() == 0);

        hmap_w1.insert(10, 10);
        hmap_w1.insert(15, 15);

        assert!(hmap_w1.iter().count() == 2);
    }

    #[test]
    fn test_hashmap_from_iter() {
        let hmap: HashMap<usize, usize> = vec![(10, 10), (15, 15), (20, 20)].into_iter().collect();
        let hmap_r2 = hmap.read();
        assert!(hmap_r2.contains_key(&10));
        assert!(hmap_r2.contains_key(&15));
        assert!(hmap_r2.contains_key(&20));
    }
}
