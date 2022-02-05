//! HashTrie - A async locked concurrently readable HashTrie
//!
//! For more, see [`HashTrie`]

#![allow(clippy::implicit_hasher)]

use crate::internals::lincowcell_async::{LinCowCell, LinCowCellReadTxn, LinCowCellWriteTxn};

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
    pub async fn read<'x>(&'x self) -> HashTrieReadTxn<'x, K, V> {
        let inner = self.inner.read().await;
        HashTrieReadTxn { inner }
    }

    /// Initiate a write transaction for the map, exclusive to this
    /// writer, and concurrently to all existing reads.
    pub async fn write<'x>(&'x self) -> HashTrieWriteTxn<'x, K, V> {
        let inner = self.inner.write().await;
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

impl<
        'a,
        K: Hash + Eq + Clone + Debug + Sync + Send + 'static,
        V: Clone + Sync + Send + 'static,
    > HashTrieWriteTxn<'a, K, V>
{
    /// Commit the changes from this write transaction. Readers after this point
    /// will be able to percieve these changes.
    ///
    /// To abort (unstage changes), just do not call this function.
    pub async fn commit(self) {
        self.inner.commit().await;
    }
}

#[cfg(test)]
mod tests {
    use super::HashTrie;

    #[tokio::test]
    async fn test_hashtrie_basic_write() {
        let hmap: HashTrie<usize, usize> = HashTrie::new();
        let mut hmap_write = hmap.write().await;

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
        hmap_write.commit().await;
    }

    #[tokio::test]
    async fn test_hashtrie_basic_read_write() {
        let hmap: HashTrie<usize, usize> = HashTrie::new();
        let mut hmap_w1 = hmap.write().await;
        hmap_w1.insert(10, 10);
        hmap_w1.insert(15, 15);
        hmap_w1.commit().await;

        let hmap_r1 = hmap.read().await;
        assert!(hmap_r1.contains_key(&10));
        assert!(hmap_r1.contains_key(&15));
        assert!(!hmap_r1.contains_key(&20));

        let mut hmap_w2 = hmap.write().await;
        hmap_w2.insert(20, 20);
        hmap_w2.commit().await;

        assert!(hmap_r1.contains_key(&10));
        assert!(hmap_r1.contains_key(&15));
        assert!(!hmap_r1.contains_key(&20));

        let hmap_r2 = hmap.read().await;
        assert!(hmap_r2.contains_key(&10));
        assert!(hmap_r2.contains_key(&15));
        assert!(hmap_r2.contains_key(&20));
    }

    #[tokio::test]
    async fn test_hashtrie_basic_read_snapshot() {
        let hmap: HashTrie<usize, usize> = HashTrie::new();
        let mut hmap_w1 = hmap.write().await;
        hmap_w1.insert(10, 10);
        hmap_w1.insert(15, 15);

        let snap = hmap_w1.to_snapshot();
        assert!(snap.contains_key(&10));
        assert!(snap.contains_key(&15));
        assert!(!snap.contains_key(&20));
    }

    #[tokio::test]
    async fn test_hashtrie_basic_iter() {
        let hmap: HashTrie<usize, usize> = HashTrie::new();
        let mut hmap_w1 = hmap.write().await;
        assert!(hmap_w1.iter().count() == 0);

        hmap_w1.insert(10, 10);
        hmap_w1.insert(15, 15);

        assert!(hmap_w1.iter().count() == 2);
    }

    #[tokio::test]
    async fn test_hashtrie_from_iter() {
        let hmap: HashTrie<usize, usize> = vec![(10, 10), (15, 15), (20, 20)].into_iter().collect();
        let hmap_r2 = hmap.read().await;
        assert!(hmap_r2.contains_key(&10));
        assert!(hmap_r2.contains_key(&15));
        assert!(hmap_r2.contains_key(&20));
    }
}
