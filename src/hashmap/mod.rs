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
//! be interleaved atomicly and safely, and the readers always see the latest
//! data. While this is potentially useful to a set of problems, transactional
//! structures are suited to problems where readers have to maintain consistent
//! data views for a duration of time, cpu cache friendly behaviours and
//! database like transaction properties (ACID).

#![allow(clippy::implicit_hasher)]

#[cfg(feature = "asynch")]
pub mod asynch;

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

impl<
        'a,
        K: Hash + Eq + Clone + Debug + Sync + Send + 'static,
        V: Clone + Sync + Send + 'static,
    > HashMapWriteTxn<'a, K, V>
{
    /*
    pub(crate) fn get_txid(&self) -> u64 {
        self.inner.as_ref().get_txid()
    }
    */

    /*
    pub(crate) fn prehash<'b, Q: ?Sized>(&'a self, k: &'b Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.inner.as_ref().hash_key(k)
    }
    */

    /*
    /// This is *unsafe* because changing the key CAN and WILL break hashing, which can
    /// have serious consequences. This API only exists to allow arcache to access the inner
    /// content of the slot to simplify its API. You should basically never touch this
    /// function as its the HashMap equivalent of the demon sphere.
    pub(crate) unsafe fn get_slot_mut(&mut self, k_hash: u64) -> Option<&mut [Datum<K, V>]> {
        self.inner.as_mut().get_slot_mut_ref(k_hash)
    }
    */

    /// Commit the changes from this write transaction. Readers after this point
    /// will be able to percieve these changes.
    ///
    /// To abort (unstage changes), just do not call this function.
    pub fn commit(self) {
        self.inner.commit();
    }
}

/*
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
}
*/

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
