
//! HashMap - A concurrently readable HashMap
//!
//! This is a specialisation of the `BptreeMap`, allowing a concurrently readable
//! HashMap. Unlike a traditional hashmap it does *not* have `O(1)` lookup, as it
//! internally uses a tree-like structure to store a series of buckets. However
//! if you do not need key-ordering, due to the storage of the hashes as `u64`
//! the operations in the tree to seek the bucket is much faster than the use of
//! the same key in the `BptreeMap`.
//!
//! For more details. see the `BptreeMap`
//!
//! This structure is very different to the `im` crate. The `im` crate is
//! sync + send over individual operations. This means that multiple writes can
//! be interleaved atomicly and safely, and the readers always see the latest
//! data. While this is potentially useful to a set of problems, transactional
//! structures are suited to problems where readers have to maintain consistent
//! data views for a duration of time, cpu cache friendly behaviours and
//! database like transaction properties (ACID).

// TODO:
#![allow(clippy::implicit_hasher)]

use std::borrow::Borrow;
use crate::internals::hashmap::cursor::CursorReadOps;
use crate::internals::hashmap::cursor::{CursorRead, CursorWrite, SuperBlock};
use crate::internals::hashmap::iter::*;
use crate::internals::hashmap::node::Datum;

use crate::internals::lincowcell::LinCowCellCapable;
use crate::internals::lincowcell::{LinCowCell, LinCowCellReadTxn, LinCowCellWriteTxn};

use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;

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
    K: Hash + Eq + Clone + Debug + Sync + Send + 'static,
    V: Clone + Sync + Send + 'static,
{
    inner: LinCowCell<SuperBlock<K, V>, CursorRead<K, V>, CursorWrite<K, V>>,
}

unsafe impl<K: Hash + Eq + Clone + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static>
    Send for HashMap<K, V>
{
}
unsafe impl<K: Hash + Eq + Clone + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static>
    Sync for HashMap<K, V>
{
}

/// An active read transaction over a `HashMap`. The data in this tree
/// is guaranteed to not change and will remain consistent for the life
/// of this transaction.
pub struct HashMapReadTxn<'a, K, V>
where
    K: Hash + Eq + Clone + Debug + Sync + Send + 'static,
    V: Clone + Sync + Send + 'static,
{
    inner: LinCowCellReadTxn<'a, SuperBlock<K, V>, CursorRead<K, V>, CursorWrite<K, V>>,
}

/// An active write transaction for a `HashMap`. The data in this tree
/// may be modified exclusively through this transaction without affecting
/// readers. The write may be rolledback/aborted by dropping this guard
/// without calling `commit()`. Once `commit()` is called, readers will be
/// able to access and percieve changes in new transactions.
pub struct HashMapWriteTxn<'a, K, V>
where
    K: Hash + Eq + Clone + Debug + Sync + Send + 'static,
    V: Clone + Sync + Send + 'static,
{
    inner: LinCowCellWriteTxn<'a, SuperBlock<K, V>, CursorRead<K, V>, CursorWrite<K, V>>,
}

enum SnapshotType<'a, K, V>
where
    K: Hash + Eq + Clone + Debug + Sync + Send + 'static,
    V: Clone + Sync + Send + 'static,
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
    K: Hash + Eq + Clone + Debug + Sync + Send + 'static,
    V: Clone + Sync + Send + 'static,
{
    inner: SnapshotType<'a, K, V>,
}

impl<K: Hash + Eq + Clone + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static> Default
    for HashMap<K, V>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static>
    HashMap<K, V>
{
    /// Construct a new concurrent hashmap
    pub fn new() -> Self {
        // I acknowledge I understand what is required to make this safe.
        HashMap {
            inner: LinCowCell::new(unsafe { SuperBlock::new() }),
        }
    }

    /// Initiate a read transaction for the Hashmap, concurrent to any
    /// other readers or writers.
    pub fn read(&self) -> HashMapReadTxn<K, V> {
        let inner = self.inner.read();
        HashMapReadTxn {
            inner
        }
    }

    /// Initiate a write transaction for the map, exclusive to this
    /// writer, and concurrently to all existing reads.
    pub fn write(&self) -> HashMapWriteTxn<K, V> {
        let inner = self.inner.write();
        HashMapWriteTxn {
            inner
        }
    }

    /// Attempt to create a new write, returns None if another writer
    /// already exists.
    pub fn try_write(&self) -> Option<HashMapWriteTxn<K, V>> {
        self.inner
            .try_write()
            .map(|inner|
            HashMapWriteTxn {
                inner
            }
            )
    }
}

impl<K: Hash + Eq + Clone + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static>
    FromIterator<(K, V)> for HashMap<K, V>
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut new_sblock = unsafe { SuperBlock::new() };
        let prev = new_sblock.create_reader();
        let mut cursor = new_sblock.create_writer();
        cursor.extend(iter);

        let _ = new_sblock.pre_commit(cursor, &prev);

        HashMap {
            inner: LinCowCell::new(new_sblock),
        }
    }
}

impl<
        'a,
        K: Hash + Eq + Clone + Debug + Sync + Send + 'static,
        V: Clone + Sync + Send + 'static,
    > Extend<(K, V)> for HashMapWriteTxn<'a, K, V>
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        iter.into_iter().for_each(|(k, v)| {
            let _ = self.insert(k, v);
        });
    }
}

impl<
        'a,
        K: Hash + Eq + Clone + Debug + Sync + Send + 'static,
        V: Clone + Sync + Send + 'static,
    > HashMapWriteTxn<'a, K, V>
{
    pub(crate) fn get_txid(&self) -> u64 {
        self.inner.as_ref().get_txid()
    }

    pub(crate) fn prehash<'b, Q: ?Sized>(&'a self, k: &'b Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.inner.as_ref().hash_key(k)
    }

    pub(crate) fn get_prehashed<'b, Q: ?Sized>(&'a self, k: &'b Q, k_hash: u64) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.inner.as_ref().search(k_hash, k)
    }

    /// Retrieve a value from the map. If the value exists, a reference is returned
    /// as `Some(&V)`, otherwise if not present `None` is returned.
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k_hash = self.inner.as_ref().hash_key(k);
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
        self.inner.as_ref().len()
    }

    /// Determine if the set is currently empty
    pub fn is_empty(&self) -> bool {
        self.inner.as_ref().len() == 0
    }

    /// Iterator over `(&K, &V)` of the set
    pub fn iter(&self) -> Iter<K, V> {
        self.inner.as_ref().kv_iter()
    }

    /// Iterator over &K
    pub fn values(&self) -> ValueIter<K, V> {
        self.inner.as_ref().v_iter()
    }

    /// Iterator over &V
    pub fn keys(&self) -> KeyIter<K, V> {
        self.inner.as_ref().k_iter()
    }

    /// Reset this map to an empty state. As this is within the transaction this
    /// change only takes effect once commited. Once cleared, you can begin adding
    /// new writes and changes, again, that will only be visible once commited.
    pub fn clear(&mut self) {
        self.inner.as_mut().clear()
    }

    /// Insert or update a value by key. If the value previously existed it is returned
    /// as `Some(V)`. If the value did not previously exist this returns `None`.
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        // Hash the key.
        let k_hash = self.inner.as_ref().hash_key(k);
        self.inner.as_mut().insert(k_hash, k, v)
    }

    /// Remove a key if it exists in the tree. If the value exists, we return it as `Some(V)`,
    /// and if it did not exist, we return `None`
    pub fn remove(&mut self, k: &K) -> Option<V> {
        let k_hash = self.inner.as_ref().hash_key(k);
        self.inner.as_mut().remove(k_hash, k)
    }

    /// Get a mutable reference to a value in the tree. This is correctly, and
    /// safely cloned before you attempt to mutate the value, isolating it from
    /// other transactions.
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let k_hash = self.inner.as_ref().hash_key(k);
        self.inner.as_mut().get_mut_ref(k_hash, k)
    }

    /// This is *unsafe* because changing the key CAN and WILL break hashing, which can
    /// have serious consequences. This API only exists to allow arcache to access the inner
    /// content of the slot to simplify it's API. You should basically never touch this
    /// function as it's the HashMap equivalent of a the demon sphere.
    pub(crate) unsafe fn get_slot_mut(&mut self, k_hash: u64) -> Option<&mut [Datum<K, V>]> {
        self.inner.as_mut().get_slot_mut_ref(k_hash)
    }

    /// Create a read-snapshot of the current map. This does NOT guarantee the map may
    /// not be mutated during the read, so you MUST guarantee that no functions of the
    /// write txn are called while this snapshot is active.
    pub fn to_snapshot(&'a self) -> HashMapReadSnapshot<K, V> {
        HashMapReadSnapshot {
            inner: SnapshotType::W(&self.inner.as_ref()),
        }
    }

    /// Commit the changes from this write transaction. Readers after this point
    /// will be able to percieve these changes.
    ///
    /// To abort (unstage changes), just do not call this function.
    pub fn commit(self) {
        self.inner.commit();
    }
}

impl<
        'a,
        K: Hash + Eq + Clone + Debug + Sync + Send + 'static,
        V: Clone + Sync + Send + 'static,
    > HashMapReadTxn<'a, K, V>
{
    pub(crate) fn get_txid(&self) -> u64 {
        self.inner.as_ref().get_txid()
    }

    pub(crate) fn prehash<'b, Q: ?Sized>(&'a self, k: &'b Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.inner.as_ref().hash_key(k)
    }

    pub(crate) fn get_prehashed<'b, Q: ?Sized>(&'a self, k: &'b Q, k_hash: u64) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.inner.search(k_hash, k)
    }

    /// Retrieve a value from the tree. If the value exists, a reference is returned
    /// as `Some(&V)`, otherwise if not present `None` is returned.
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k_hash = self.inner.as_ref().hash_key(k);
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
        self.inner.as_ref().len()
    }

    /// Determine if the set is currently empty
    pub fn is_empty(&self) -> bool {
        self.inner.as_ref().len() == 0
    }

    /// Iterator over `(&K, &V)` of the set
    pub fn iter(&self) -> Iter<K, V> {
        self.inner.as_ref().kv_iter()
    }

    /// Iterator over &K
    pub fn values(&self) -> ValueIter<K, V> {
        self.inner.as_ref().v_iter()
    }

    /// Iterator over &V
    pub fn keys(&self) -> KeyIter<K, V> {
        self.inner.as_ref().k_iter()
    }

    /// Create a read-snapshot of the current tree.
    /// As this is the read variant, it IS safe, and guaranteed the tree will not change.
    pub fn to_snapshot(&'a self) -> HashMapReadSnapshot<'a, K, V> {
        HashMapReadSnapshot {
            inner: SnapshotType::R(&self.inner.as_ref()),
        }
    }
}

impl<
        'a,
        K: Hash + Eq + Clone + Debug + Sync + Send + 'static,
        V: Clone + Sync + Send + 'static,
    > HashMapReadSnapshot<'a, K, V>
{
    /// Retrieve a value from the tree. If the value exists, a reference is returned
    /// as `Some(&V)`, otherwise if not present `None` is returned.
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        match self.inner {
            SnapshotType::R(inner) => {
                let k_hash = inner.hash_key(k);
                inner.search(k_hash, k)
            }
            SnapshotType::W(inner) => {
                let k_hash = inner.hash_key(k);
                inner.search(k_hash, k)
            }
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
        match self.inner {
            SnapshotType::R(inner) => inner.len(),
            SnapshotType::W(inner) => inner.len(),
        }
    }

    /// Determine if the set is currently empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // (adv) range

    /// Iterator over `(&K, &V)` of the set
    pub fn iter(&self) -> Iter<K, V> {
        match self.inner {
            SnapshotType::R(inner) => inner.kv_iter(),
            SnapshotType::W(inner) => inner.kv_iter(),
        }
    }

    /// Iterator over &K
    pub fn values(&self) -> ValueIter<K, V> {
        match self.inner {
            SnapshotType::R(inner) => inner.v_iter(),
            SnapshotType::W(inner) => inner.v_iter(),
        }
    }

    /// Iterator over &V
    pub fn keys(&self) -> KeyIter<K, V> {
        match self.inner {
            SnapshotType::R(inner) => inner.k_iter(),
            SnapshotType::W(inner) => inner.k_iter(),
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
