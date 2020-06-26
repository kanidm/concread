//! See the documentation for `HashMap` module.

// TODO:
#![allow(clippy::implicit_hasher)]

use ahash::AHasher;
use std::borrow::Borrow;
// use std::collections::hash_map::DefaultHasher;
use rand::Rng;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::mem;

use super::iter::*;
use crate::collections::bptree::{
    BptreeMap, BptreeMapReadSnapshot, BptreeMapReadTxn, BptreeMapWriteTxn,
};

// #[cfg(feature = "simd_support")] use packed_simd::*;
// #[cfg(feature = "simd_support")]

use smallvec::SmallVec;

const DEFAULT_STACK_ALLOC: usize = 1;

pub(crate) type Vinner<K, V> = SmallVec<[(K, V); DEFAULT_STACK_ALLOC]>;

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
    map: BptreeMap<u64, Vinner<K, V>>,
    key1: u64,
    key2: u64,
}

/// An active read transaction over a `HashMap`. The data in this tree
/// is guaranteed to not change and will remain consistent for the life
/// of this transaction.
pub struct HashMapReadTxn<'a, K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    map: BptreeMapReadTxn<'a, u64, Vinner<K, V>>,
    key1: u64,
    key2: u64,
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
    map: BptreeMapWriteTxn<'a, u64, Vinner<K, V>>,
    key1: u64,
    key2: u64,
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
    map: BptreeMapReadSnapshot<'a, u64, Vinner<K, V>>,
    key1: u64,
    key2: u64,
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
            map: BptreeMap::new(),
            key1: rand::thread_rng().gen::<u64>(),
            key2: rand::thread_rng().gen::<u64>(),
        }
    }

    /// Initiate a read transaction for the Hashmap, concurrent to any
    /// other readers or writers.
    pub fn read(&self) -> HashMapReadTxn<K, V> {
        HashMapReadTxn {
            map: self.map.read(),
            key1: self.key1,
            key2: self.key2,
        }
    }

    /// Initiate a write transaction for the map, exclusive to this
    /// writer, and concurrently to all existing reads.
    pub fn write(&self) -> HashMapWriteTxn<K, V> {
        HashMapWriteTxn {
            map: self.map.write(),
            key1: self.key1,
            key2: self.key2,
        }
    }

    /// Attempt to create a new write, returns None if another writer
    /// already exists.
    pub fn try_write(&self) -> Option<HashMapWriteTxn<K, V>> {
        self.map.try_write().map(|lmap| HashMapWriteTxn {
            map: lmap,
            key1: self.key1,
            key2: self.key2,
        })
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
        self.map.get_txid()
    }

    /// Retrieve a value from the map. If the value exists, a reference is returned
    /// as `Some(&V)`, otherwise if not present `None` is returned.
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k_hash = hash_key!(k, self.key1, self.key2);
        self.map.get(&k_hash).and_then(|va| {
            va.iter()
                .filter(|(ki, _vi)| k.eq(ki.borrow()))
                .take(1)
                .map(|(_kr, vr)| vr)
                .next()
        })
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
        // TODO: We need to count this ourselves!
        self.map.len()
    }

    /// Determine if the set is currently empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterator over `(&K, &V)` of the set
    pub fn iter(&self) -> Iter<K, V> {
        Iter::new(self.map.iter())
    }

    /// Iterator over &K
    pub fn values(&self) -> ValueIter<K, V> {
        ValueIter::new(self.map.iter())
    }

    /// Iterator over &V
    pub fn keys(&self) -> KeyIter<K, V> {
        KeyIter::new(self.map.iter())
    }

    /// Reset this map to an empty state. As this is within the transaction this
    /// change only takes effect once commited. Once cleared, you can begin adding
    /// new writes and changes, again, that will only be visible once commited.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Insert or update a value by key. If the value previously existed it is returned
    /// as `Some(V)`. If the value did not previously exist this returns `None`.
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        // Hash the key.
        let k_hash = hash_key!(k, self.key1, self.key2);
        // Does it exist?
        match self.map.get_mut(&k_hash) {
            Some(va) => {
                // Does our k exist in va?
                for (ki, vi) in va.as_mut_slice().iter_mut() {
                    if *ki == k {
                        // swap v and vi
                        let mut ov = v;
                        mem::swap(&mut ov, vi);
                        // Return the previous value.
                        return Some(ov);
                    }
                }
                // If we get here, it wasn't present.
                va.push((k, v));
                None
            }
            None => {
                let mut va = SmallVec::new();
                va.push((k, v));
                self.map.insert(k_hash, va);
                None
            }
        }
    }

    /// Remove a key if it exists in the tree. If the value exists, we return it as `Some(V)`,
    /// and if it did not exist, we return `None`
    pub fn remove(&mut self, k: &K) -> Option<V> {
        let k_hash = hash_key!(k, self.key1, self.key2);
        match self.map.get_mut(&k_hash) {
            Some(va) => {
                let mut idx = 0;
                for (ki, _vi) in va.iter() {
                    if k.eq(ki.borrow()) {
                        break;
                    }
                    idx += 1;
                }
                if idx > va.len() {
                    None
                } else {
                    let (_ki, vi) = va.remove(idx);
                    Some(vi)
                }
            }
            None => None,
        }
    }

    /// Get a mutable reference to a value in the tree. This is correctly, and
    /// safely cloned before you attempt to mutate the value, isolating it from
    /// other transactions.
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let k_hash = hash_key!(k, self.key1, self.key2);
        self.map.get_mut(&k_hash).and_then(|va| {
            va.iter_mut()
                .filter(|(ki, _vi)| k.eq(ki.borrow()))
                .take(1)
                .map(|(_kr, vr)| vr)
                .next()
        })
    }

    /// Create a read-snapshot of the current map. This does NOT guarantee the map may
    /// not be mutated during the read, so you MUST guarantee that no functions of the
    /// write txn are called while this snapshot is active.
    pub fn to_snapshot(&'a self) -> HashMapReadSnapshot<K, V> {
        HashMapReadSnapshot {
            map: self.map.to_snapshot(),
            key1: self.key1,
            key2: self.key2,
        }
    }

    /// Commit the changes from this write transaction. Readers after this point
    /// will be able to percieve these changes.
    ///
    /// To abort (unstage changes), just do not call this function.
    pub fn commit(self) {
        self.map.commit()
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> HashMapReadTxn<'a, K, V> {
    pub(crate) fn get_txid(&self) -> u64 {
        self.map.get_txid()
    }

    /// Retrieve a value from the tree. If the value exists, a reference is returned
    /// as `Some(&V)`, otherwise if not present `None` is returned.
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k_hash = hash_key!(k, self.key1, self.key2);
        self.map.get(&k_hash).and_then(|va| {
            va.iter()
                .filter(|(ki, _vi)| k.eq(ki.borrow()))
                .take(1)
                .map(|(_kr, vr)| {
                    // This is some lifetime stripping to deal with the fact that
                    // this ref IS valid, but it's bound to k_hash, not to &self
                    // so we ... cheat.
                    vr as *const V
                })
                // ThIs Is ThE GuD RuSt
                .map(|v| unsafe { &*v as &'a V })
                .next()
        })
    }

    /// Assert if a key exists in the tree.
    pub fn contains_key<'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(k).is_some()
    }

    /*
    /// Returns the current number of k:v pairs in the tree
    pub fn len(&self) -> usize {
        unimplemented!();
    }

    /// Determine if the set is currently empty
    pub fn is_empty(&self) -> bool {
        unimplemented!();
    }
    */

    /// Iterator over `(&K, &V)` of the set
    pub fn iter(&self) -> Iter<K, V> {
        Iter::new(self.map.iter())
    }

    /// Iterator over &K
    pub fn values(&self) -> ValueIter<K, V> {
        ValueIter::new(self.map.iter())
    }

    /// Iterator over &V
    pub fn keys(&self) -> KeyIter<K, V> {
        KeyIter::new(self.map.iter())
    }

    /// Create a read-snapshot of the current tree.
    /// As this is the read variant, it IS safe, and guaranteed the tree will not change.
    pub fn to_snapshot(&'a self) -> HashMapReadSnapshot<'a, K, V> {
        HashMapReadSnapshot {
            map: self.map.to_snapshot(),
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
        self.map.get(&k_hash).and_then(|va| {
            va.iter()
                .filter(|(ki, _vi)| k.eq(ki.borrow()))
                .take(1)
                .map(|(_kr, vr)| vr as *const V)
                .map(|v| unsafe { &*v as &'a V })
                .next()
        })
    }

    /// Assert if a key exists in the tree.
    pub fn contains_key<'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(k).is_some()
    }

    /*
    /// Returns the current number of k:v pairs in the tree
    pub fn len(&self) -> usize {
        unimplemented!();
    }

    /// Determine if the set is currently empty
    pub fn is_empty(&self) -> bool {
        unimplemented!();
    }
    */

    /// Iterator over `(&K, &V)` of the set
    pub fn iter(&self) -> Iter<K, V> {
        Iter::new(self.map.iter())
    }

    /// Iterator over &K
    pub fn values(&self) -> ValueIter<K, V> {
        ValueIter::new(self.map.iter())
    }

    /// Iterator over &V
    pub fn keys(&self) -> KeyIter<K, V> {
        KeyIter::new(self.map.iter())
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
