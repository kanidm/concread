use super::ll::{LLNode, LL};

use crossbeam::channel::{Receiver, Sender};
use std::collections::HashMap as Map;

use std::fmt::Debug;
use std::hash::Hash;
use std::time::Instant;

// const READ_THREAD_MIN: usize = 8;
pub(crate) const READ_THREAD_RATIO: usize = 16;

/// Statistics related to the Arc
#[derive(Clone, Debug, PartialEq)]
pub struct CacheStats {
    /// The number of hits during all read operations.
    pub reader_hits: usize,
    /// The number of inclusions through read operations.
    pub reader_includes: usize,
    /// The number of hits during all write operations.
    pub write_hits: usize,
    /// The number of inclusions or changes through write operations.
    pub write_inc_or_mod: usize,
    /// The maximum number of items in the shared cache.
    pub shared_max: usize,
    /// The number of items in the frequent set
    pub freq: usize,
    /// The number of items in the recent set
    pub recent: usize,
    /// The number of items evicted from the frequent set
    pub freq_evicts: usize,
    /// The number of items evicted from the recent set
    pub recent_evicts: usize,
    /// The current cache weight between recent and frequent.
    pub p_weight: usize,
    /// The number of keys seen through the cache's lifetime.
    pub all_seen_keys: usize,
}

pub(crate) enum ThreadCacheItem<V> {
    Present(V, bool),
    Removed(bool),
}

pub(crate) enum CacheEvent<K, V> {
    Hit(Instant, K),
    Include(Instant, K, V, u64),
}

#[derive(Hash, Ord, PartialOrd, Eq, PartialEq, Clone, Debug)]
pub(crate) struct CacheItemInner<K>
where
    K: Hash + Eq + Ord + Clone + Debug,
{
    pub k: K,
    pub txid: u64,
}

#[derive(Clone, Debug)]
pub(crate) enum CacheItem<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug,
{
    Freq(*mut LLNode<CacheItemInner<K>>, V),
    Rec(*mut LLNode<CacheItemInner<K>>, V),
    GhostFreq(*mut LLNode<CacheItemInner<K>>),
    GhostRec(*mut LLNode<CacheItemInner<K>>),
    Haunted(*mut LLNode<CacheItemInner<K>>),
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CacheState {
    Freq,
    Rec,
    GhostFreq,
    GhostRec,
    Haunted,
    None,
}

#[cfg(test)]
#[derive(Debug, PartialEq)]
pub(crate) struct CStat {
    pub max: usize,
    pub cache: usize,
    pub tlocal: usize,
    pub freq: usize,
    pub rec: usize,
    pub ghost_freq: usize,
    pub ghost_rec: usize,
    pub haunted: usize,
    pub p: usize,
}

pub(crate) struct ArcInner<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug,
    V: Clone + Debug,
{
    // Weight of items between the two caches.
    pub p: usize,
    pub freq: LL<CacheItemInner<K>>,
    pub rec: LL<CacheItemInner<K>>,
    pub ghost_freq: LL<CacheItemInner<K>>,
    pub ghost_rec: LL<CacheItemInner<K>>,
    pub haunted: LL<CacheItemInner<K>>,
    pub rx: Receiver<CacheEvent<K, V>>,
    pub min_txid: u64,
}

pub(crate) struct ArcShared<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug,
    V: Clone + Debug,
{
    // Max number of elements to cache.
    pub max: usize,
    // Max number of elements for a reader per thread.
    pub read_max: usize,
    // channels for readers.
    // tx (cloneable)
    pub tx: Sender<CacheEvent<K, V>>,
}

pub(crate) struct ReadCache<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug,
    V: Clone + Debug,
{
    // cache of our missed items to send forward.
    // On drop we drain this to the channel
    pub set: Map<K, *mut LLNode<(K, V)>>,
    pub read_size: usize,
    pub tlru: LL<(K, V)>,
}

impl<K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> CacheItem<K, V> {
    pub(crate) fn to_vref(&self) -> Option<&V> {
        match &self {
            CacheItem::Freq(_, v) | CacheItem::Rec(_, v) => Some(&v),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn to_state(&self) -> CacheState {
        match &self {
            CacheItem::Freq(_, _v) => CacheState::Freq,
            CacheItem::Rec(_, _v) => CacheState::Rec,
            CacheItem::GhostFreq(_) => CacheState::GhostFreq,
            CacheItem::GhostRec(_) => CacheState::GhostRec,
            CacheItem::Haunted(_) => CacheState::Haunted,
        }
    }
}

