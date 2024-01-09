//! HashTrie - A concurrently readable HashTrie
//!
//! A HashTrie is similar to the Tree based `HashMap`, however instead of
//! storing hashes in a tree arrangement, we use Trie behaviours to slice a hash
//! into array indexes for accessing the elements. This reduces memory consumed
//! as we do not need to store hashes of values in branches, but it can cause
//! memory to increase as we don't have node-split behaviour like a BTree
//! that backs the HashMap.
//!
//! Generally, this structure is faster than the HashMap, but at the expense
//! that it may consume more memory for it's internal storage. However, even at
//! large sizes such as ~16,000,000 entries, this will only consume ~16KB for
//! branches. The majority of your space will be taken by your keys and values.
//!
//! If in doubt, use `HashMap` 😁
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

#[cfg(feature = "arcache")]
use crate::internals::hashtrie::cursor::Datum;

use crate::internals::lincowcell::{LinCowCell, LinCowCellReadTxn, LinCowCellWriteTxn};

include!("impl.rs");

impl<K: Hash + Eq + Clone + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static>
    HashTrie<K, V>
{
    /// Construct a new concurrent hashtrie
    pub fn new() -> Self {
        // I acknowledge I understand what is required to make this safe.
        HashTrie {
            inner: LinCowCell::new(unsafe { SuperBlock::new() }),
        }
    }

    /// Initiate a read transaction for the Hashmap, concurrent to any
    /// other readers or writers.
    pub fn read(&self) -> HashTrieReadTxn<K, V> {
        let inner = self.inner.read();
        HashTrieReadTxn { inner }
    }

    /// Initiate a write transaction for the map, exclusive to this
    /// writer, and concurrently to all existing reads.
    pub fn write(&self) -> HashTrieWriteTxn<K, V> {
        let inner = self.inner.write();
        HashTrieWriteTxn { inner }
    }

    /// Attempt to create a new write, returns None if another writer
    /// already exists.
    pub fn try_write(&self) -> Option<HashTrieWriteTxn<K, V>> {
        self.inner
            .try_write()
            .map(|inner| HashTrieWriteTxn { inner })
    }
}

impl<K: Hash + Eq + Clone + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static>
    HashTrieWriteTxn<'_, K, V>
{
    #[cfg(feature = "arcache")]
    pub(crate) fn get_txid(&self) -> u64 {
        self.inner.as_ref().get_txid()
    }

    #[cfg(feature = "arcache")]
    pub(crate) fn prehash<Q: ?Sized>(&self, k: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.inner.as_ref().hash_key(k)
    }

    /// This is *unsafe* because changing the key CAN and WILL break hashing, which can
    /// have serious consequences. This API only exists to allow arcache to access the inner
    /// content of the slot to simplify its API. You should basically never touch this
    /// function as it's the HashTrie equivalent of a the demon sphere.
    #[cfg(feature = "arcache")]
    pub(crate) unsafe fn get_slot_mut(&mut self, k_hash: u64) -> Option<&mut [Datum<K, V>]> {
        self.inner.as_mut().get_slot_mut_ref(k_hash)
    }

    /// Commit the changes from this write transaction. Readers after this point
    /// will be able to percieve these changes.
    ///
    /// To abort (unstage changes), just do not call this function.
    pub fn commit(self) {
        self.inner.commit();
    }
}

impl<K: Hash + Eq + Clone + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static>
    HashTrieReadTxn<'_, K, V>
{
    #[cfg(feature = "arcache")]
    pub(crate) fn get_txid(&self) -> u64 {
        self.inner.as_ref().get_txid()
    }

    #[cfg(feature = "arcache")]
    pub(crate) fn prehash<Q: ?Sized>(&self, k: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.inner.as_ref().hash_key(k)
    }
}

#[cfg(test)]
mod tests {
    use super::HashTrie;

    #[test]
    fn test_hashtrie_basic_write() {
        let hmap: HashTrie<usize, usize> = HashTrie::new();
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
    fn test_hashtrie_basic_read_write() {
        let hmap: HashTrie<usize, usize> = HashTrie::new();
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
    fn test_hashtrie_basic_read_snapshot() {
        let hmap: HashTrie<usize, usize> = HashTrie::new();
        let mut hmap_w1 = hmap.write();
        hmap_w1.insert(10, 10);
        hmap_w1.insert(15, 15);

        let snap = hmap_w1.to_snapshot();
        assert!(snap.contains_key(&10));
        assert!(snap.contains_key(&15));
        assert!(!snap.contains_key(&20));
    }

    #[test]
    fn test_hashtrie_basic_iter() {
        let hmap: HashTrie<usize, usize> = HashTrie::new();
        let mut hmap_w1 = hmap.write();
        assert!(hmap_w1.iter().count() == 0);

        hmap_w1.insert(10, 10);
        hmap_w1.insert(15, 15);

        assert!(hmap_w1.iter().count() == 2);
    }

    #[test]
    fn test_hashtrie_from_iter() {
        let hmap: HashTrie<usize, usize> = vec![(10, 10), (15, 15), (20, 20)].into_iter().collect();
        let hmap_r2 = hmap.read();
        assert!(hmap_r2.contains_key(&10));
        assert!(hmap_r2.contains_key(&15));
        assert!(hmap_r2.contains_key(&20));
    }

    #[test]
    fn test_hashtrie_double_free() {
        let hmap: HashTrie<usize, usize> = HashTrie::new();
        let mut tx = hmap.write();
        for _i in 0..2 {
            tx.insert(13, 34);
            tx.remove(&13);
        }
    }
}
