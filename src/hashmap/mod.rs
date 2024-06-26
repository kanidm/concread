//! HashMap - A concurrently readable HashMap
//!
//! This is a specialisation of the `BptreeMap`, allowing a concurrently readable
//! HashMap. Unlike a traditional hashmap it does *not* have `O(1)` lookup, as it
//! internally uses a tree-like structure to store a series of buckets. However
//! if you do not need key-ordering, due to the storage of the hashes as `u64`
//! the operations in the tree to seek the bucket is much faster than the use of
//! the same key in the `BptreeMap`.
//!
//! For more details. see the [BptreeMap](crate::bptree::BptreeMap)
//!
//! This structure is very different to the `im` crate. The `im` crate is
//! sync + send over individual operations. This means that multiple writes can
//! be interleaved atomically and safely, and the readers always see the latest
//! data. While this is potentially useful to a set of problems, transactional
//! structures are suited to problems where readers have to maintain consistent
//! data views for a duration of time, cpu cache friendly behaviours and
//! database like transaction properties (ACID).

#![allow(clippy::implicit_hasher)]

#[cfg(feature = "asynch")]
pub mod asynch;

#[cfg(feature = "serde")]
use serde::{
    de::{Deserialize, Deserializer},
    ser::{Serialize, SerializeMap, Serializer},
};

#[cfg(feature = "serde")]
use crate::utils::MapCollector;

#[cfg(all(feature = "arcache", feature = "arcache-is-hashmap"))]
use crate::internals::hashmap::cursor::Datum;

use crate::internals::lincowcell::{LinCowCell, LinCowCellReadTxn, LinCowCellWriteTxn};

include!("impl.rs");

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
        HashMapReadTxn { inner }
    }

    /// Initiate a write transaction for the map, exclusive to this
    /// writer, and concurrently to all existing reads.
    pub fn write(&self) -> HashMapWriteTxn<K, V> {
        let inner = self.inner.write();
        HashMapWriteTxn { inner }
    }

    /// Attempt to create a new write, returns None if another writer
    /// already exists.
    pub fn try_write(&self) -> Option<HashMapWriteTxn<K, V>> {
        self.inner
            .try_write()
            .map(|inner| HashMapWriteTxn { inner })
    }
}

impl<K: Hash + Eq + Clone + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static>
    HashMapWriteTxn<'_, K, V>
{
    #[cfg(all(feature = "arcache", feature = "arcache-is-hashmap"))]
    pub(crate) fn get_txid(&self) -> u64 {
        self.inner.as_ref().get_txid()
    }

    #[cfg(all(feature = "arcache", feature = "arcache-is-hashmap"))]
    pub(crate) fn prehash<Q>(&self, k: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.as_ref().hash_key(k)
    }

    /// This is *unsafe* because changing the key CAN and WILL break hashing, which can
    /// have serious consequences. This API only exists to allow arcache to access the inner
    /// content of the slot to simplify its API. You should basically never touch this
    /// function as it's the HashMap equivalent of the demon sphere.
    #[cfg(all(feature = "arcache", feature = "arcache-is-hashmap"))]
    pub(crate) unsafe fn get_slot_mut(&mut self, k_hash: u64) -> Option<&mut [Datum<K, V>]> {
        self.inner.as_mut().get_slot_mut_ref(k_hash)
    }

    /// Commit the changes from this write transaction. Readers after this point
    /// will be able to perceive these changes.
    ///
    /// To abort (unstage changes), just do not call this function.
    pub fn commit(self) {
        self.inner.commit();
    }
}

impl<K: Hash + Eq + Clone + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static>
    HashMapReadTxn<'_, K, V>
{
    #[cfg(all(feature = "arcache", feature = "arcache-is-hashmap"))]
    pub(crate) fn get_txid(&self) -> u64 {
        self.inner.as_ref().get_txid()
    }

    #[cfg(all(feature = "arcache", feature = "arcache-is-hashmap"))]
    pub(crate) fn prehash<Q>(&self, k: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.as_ref().hash_key(k)
    }
}

#[cfg(feature = "serde")]
impl<K, V> Serialize for HashMapReadTxn<'_, K, V>
where
    K: Serialize + Hash + Eq + Clone + Debug + Sync + Send + 'static,
    V: Serialize + Clone + Sync + Send + 'static,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_map(Some(self.len()))?;

        for (key, val) in self.iter() {
            state.serialize_entry(key, val)?;
        }

        state.end()
    }
}

#[cfg(feature = "serde")]
impl<K, V> Serialize for HashMap<K, V>
where
    K: Serialize + Hash + Eq + Clone + Debug + Sync + Send + 'static,
    V: Serialize + Clone + Sync + Send + 'static,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.read().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, K, V> Deserialize<'de> for HashMap<K, V>
where
    K: Deserialize<'de> + Hash + Eq + Clone + Debug + Sync + Send + 'static,
    V: Deserialize<'de> + Clone + Sync + Send + 'static,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MapCollector::new())
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
        hmap_write.extend(vec![(15, 15)]);

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
        assert!(!hmap_write.is_empty());

        assert_eq!(hmap_write.keys().count(), 1);

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
        assert!(!hmap_w2.is_empty());
        assert_eq!(hmap_w2.keys().count(), 3);
        assert_eq!(hmap_w2.len(), 3);
        hmap_w2.commit();

        assert!(hmap_r1.contains_key(&10));
        assert!(hmap_r1.contains_key(&15));
        assert!(!hmap_r1.contains_key(&20));

        let hmap_r2 = hmap.read();
        assert!(hmap_r2.contains_key(&10));
        assert!(hmap_r2.contains_key(&15));
        assert!(hmap_r2.contains_key(&20));
        assert!(!hmap_r2.is_empty());
        assert_eq!(hmap_r2.len(), 3);
        assert_eq!(hmap_r2.keys().count(), 3);
    }

    #[test]
    fn test_hashmap_basic_read_snapshot() {
        let hmap: HashMap<usize, usize> = HashMap::default();
        let mut hmap_w1 = hmap.write();
        hmap_w1.insert(10, 10);
        hmap_w1.insert(15, 15);

        let snap = hmap_w1.to_snapshot();
        assert!(snap.contains_key(&10));
        assert!(snap.contains_key(&15));
        assert!(!snap.contains_key(&20));
        hmap_w1.commit();

        let hmap_read = hmap.read();
        let snap = hmap_read.to_snapshot();
        assert!(snap.contains_key(&10));
        assert!(snap.contains_key(&15));
        assert!(!snap.contains_key(&20));
        assert_eq!(snap.len(), 2);
        assert!(!snap.is_empty());
        assert!(snap.iter().find(|(_k, v)| **v == 10).is_some());
        assert_eq!(snap.values().count(), 2);
        assert_eq!(snap.keys().count(), 2);
    }

    #[test]
    fn test_hashmap_basic_iter() {
        let hmap: HashMap<usize, usize> = HashMap::new();
        let mut hmap_w1 = hmap.write();
        assert!(hmap_w1.iter().count() == 0);

        hmap_w1.insert(10, 10);
        hmap_w1.insert(15, 15);

        assert!(hmap_w1.iter().count() == 2);

        let hmap_r1 = hmap.read();
        assert!(hmap_r1.iter().count() == 0);
    }

    #[test]
    fn test_hashmap_from_iter() {
        let hmap: HashMap<usize, usize> = vec![(10, 10), (15, 15), (20, 20)].into_iter().collect();
        let hmap_r2 = hmap.read();
        assert!(hmap_r2.contains_key(&10));
        assert!(hmap_r2.contains_key(&15));
        assert!(hmap_r2.contains_key(&20));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_hashmap_serialize_deserialize() {
        let hmap: HashMap<usize, usize> = vec![(10, 11), (15, 16), (20, 21)].into_iter().collect();

        let value = serde_json::to_value(&hmap).unwrap();
        assert_eq!(value, serde_json::json!({ "10": 11, "15": 16, "20": 21 }));

        let hmap: HashMap<usize, usize> = serde_json::from_value(value).unwrap();
        let mut vec: Vec<(usize, usize)> = hmap.read().iter().map(|(k, v)| (*k, *v)).collect();
        vec.sort_unstable();
        assert_eq!(vec, [(10, 11), (15, 16), (20, 21)]);
    }

    #[test]
    fn test_hashmap_keys() {
        let hmap: HashMap<usize, usize> = vec![(10, 10), (15, 15), (20, 20)].into_iter().collect();
        let hmap_read = hmap.read();
        assert!(hmap_read.keys().find(|&&x| x == 10).is_some());
        let hmap_write = hmap.write();
        assert!(hmap_write.keys().find(|&&x| x == 10).is_some());
    }
    #[test]
    fn test_hashmap_values() {
        let hmap: HashMap<usize, usize> = vec![(10, 11), (15, 15), (20, 20)].into_iter().collect();
        let hmap_read = hmap.read();
        assert!(hmap_read.values().find(|&&x| x == 11).is_some());
        let hmap_write = hmap.write();
        assert!(hmap_write.values().find(|&&x| x == 11).is_some());
    }

    #[test]
    fn test_write_snapshot_bits() {
        let hmap: HashMap<usize, usize> = vec![(10, 11), (15, 15), (20, 20)].into_iter().collect();
        let hmap_write = hmap.write();
        let hmap_write_snapshot = hmap_write.to_snapshot();
        assert!(!hmap_write_snapshot.is_empty());
        assert_eq!(hmap_write_snapshot.len(), 3);
        assert!(hmap_write_snapshot.contains_key(&10));
        assert!(hmap_write_snapshot.values().find(|&&x| x == 11).is_some());
        assert!(hmap_write_snapshot.values().find(|&&x| x == 10).is_none());
        assert!(hmap_write_snapshot.keys().find(|&&x| x == 10).is_some());
        assert!(hmap_write_snapshot.keys().find(|&&x| x == 11).is_none());
        assert!(hmap_write_snapshot.keys().find(|&&x| x == 10).is_some());
        assert!(hmap_write_snapshot.iter().count() == 3);
    }
}
