//! ARCache - A concurrently readable adaptive replacement cache.
//!
//! An ARCache is used in place of a `RwLock<LruCache>` or `Mutex<LruCache>`.
//! This structure is transactional, meaning that readers have guaranteed
//! point-in-time views of the cache and their items, while allowing writers
//! to proceed with inclusions and cache state management in parallel.
//!
//! This means that unlike a `RwLock` which can have many readers OR one writer
//! this cache is capable of many readers, over multiple data generations AND
//! writers that are serialised. This formally means that this is an ACID
//! compliant Cache.

mod ll;

use self::ll::{LLNode, LLWeight, LL};
// use self::traits::ArcWeight;
use crate::cowcell::{CowCell, CowCellReadTxn};
use crate::hashmap::*;
use crossbeam::channel::{unbounded, Receiver, Sender};
use crossbeam::queue::ArrayQueue;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap as Map;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use std::borrow::Borrow;
use std::cell::UnsafeCell;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use std::ops::Deref;
use std::ops::DerefMut;
use std::time::Instant;

// const READ_THREAD_MIN: usize = 8;
const READ_THREAD_RATIO: usize = 16;

/// Statistics related to the Arc
#[derive(Clone, Debug, PartialEq)]
pub struct CacheStats {
    /// The number of hits during all read operations on the primary cache.
    pub reader_hits: usize,
    /// The number of hits during all read operations on the thread local caches.
    pub reader_tlocal_hits: usize,
    /// The number of inclusions to the thread local storage.
    pub reader_tlocal_includes: usize,
    /// The number of inclusions through read operations.
    pub reader_includes: usize,
    /// The number of hits during all write operations.
    pub write_hits: usize,
    /// The number of inclusions or changes through write operations.
    pub write_includes: usize,
    /// The number of modifications to the cache content during a write.
    pub write_modifies: usize,
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

struct ReaderStatEvent {
    t: Instant,
    hits: usize,
    tlocal_hits: usize,
    tlocal_includes: usize,
    reader_includes: usize,
}

enum ThreadCacheItem<V> {
    Present(V, bool, usize),
    Removed(bool),
}

struct CacheHitEvent {
    t: Instant,
    k_hash: u64,
}

struct CacheIncludeEvent<K, V> {
    t: Instant,
    k: K,
    v: V,
    txid: u64,
    size: usize,
}

#[derive(Hash, Ord, PartialOrd, Eq, PartialEq, Clone, Debug)]
struct CacheItemInner<K>
where
    K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
{
    k: K,
    txid: u64,
    size: usize,
}

impl<K> LLWeight for CacheItemInner<K>
where
    K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
{
    #[inline]
    fn ll_weight(&self) -> usize {
        self.size
    }
}

#[derive(Clone, Debug)]
enum CacheItem<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
{
    Freq(*mut LLNode<CacheItemInner<K>>, V),
    Rec(*mut LLNode<CacheItemInner<K>>, V),
    GhostFreq(*mut LLNode<CacheItemInner<K>>),
    GhostRec(*mut LLNode<CacheItemInner<K>>),
    Haunted(*mut LLNode<CacheItemInner<K>>),
}

unsafe impl<
        K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
        V: Clone + Debug + Sync + Send + 'static,
    > Send for CacheItem<K, V>
{
}
unsafe impl<
        K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
        V: Clone + Debug + Sync + Send + 'static,
    > Sync for CacheItem<K, V>
{
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
    max: usize,
    cache: usize,
    tlocal: usize,
    freq: usize,
    rec: usize,
    ghost_freq: usize,
    ghost_rec: usize,
    haunted: usize,
    p: usize,
}

struct ArcInner<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
    V: Clone + Debug + Sync + Send + 'static,
{
    /// Weight of items between the two caches.
    p: usize,
    freq: LL<CacheItemInner<K>>,
    rec: LL<CacheItemInner<K>>,
    ghost_freq: LL<CacheItemInner<K>>,
    ghost_rec: LL<CacheItemInner<K>>,
    haunted: LL<CacheItemInner<K>>,
    //
    stat_rx: Receiver<ReaderStatEvent>,
    // rx: Receiver<CacheEvent<K, V>>,
    hit_queue: Arc<ArrayQueue<CacheHitEvent>>,
    inc_queue: Arc<ArrayQueue<CacheIncludeEvent<K, V>>>,
    min_txid: u64,
}

struct ArcShared<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
    V: Clone + Debug + Sync + Send + 'static,
{
    // Max number of elements to cache.
    max: usize,
    // Max number of elements for a reader per thread.
    read_max: usize,
    // channels for readers.
    stat_tx: Sender<ReaderStatEvent>,
    hit_queue: Arc<ArrayQueue<CacheHitEvent>>,
    inc_queue: Arc<ArrayQueue<CacheIncludeEvent<K, V>>>,
    /// The number of items that are present in the cache before we start to process
    /// the arc sets/lists.
    watermark: usize,
}

/// A configurable builder to create new concurrent Adaptive Replacement Caches.
#[derive(Default)]
pub struct ARCacheBuilder {
    stats: Option<CacheStats>,
    max: Option<usize>,
    read_max: Option<usize>,
    watermark: Option<usize>,
}

/// A concurrently readable adaptive replacement cache. Operations are performed on the
/// cache via read and write operations.
pub struct ARCache<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
    V: Clone + Debug + Sync + Send + 'static,
{
    // Use a unified tree, allows simpler movement of items between the
    // cache types.
    cache: HashMap<K, CacheItem<K, V>>,
    // This is normally only ever taken in "read" mode, so it's effectively
    // an uncontended barrier.
    shared: RwLock<ArcShared<K, V>>,
    // These are only taken during a quiesce
    inner: Mutex<ArcInner<K, V>>,
    stats: CowCell<CacheStats>,
    above_watermark: AtomicBool,
}

unsafe impl<
        K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
        V: Clone + Debug + Sync + Send + 'static,
    > Send for ARCache<K, V>
{
}
unsafe impl<
        K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
        V: Clone + Debug + Sync + Send + 'static,
    > Sync for ARCache<K, V>
{
}

#[derive(Debug, Clone)]
struct ReadCacheItem<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
    V: Clone + Debug + Sync + Send + 'static,
{
    k: K,
    v: V,
    size: usize,
}

impl<K, V> LLWeight for ReadCacheItem<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
    V: Clone + Debug + Sync + Send + 'static,
{
    #[inline]
    fn ll_weight(&self) -> usize {
        self.size
    }
}

struct ReadCache<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
    V: Clone + Debug + Sync + Send + 'static,
{
    // cache of our missed items to send forward.
    // On drop we drain this to the channel
    set: Map<K, *mut LLNode<ReadCacheItem<K, V>>>,
    read_size: usize,
    tlru: LL<ReadCacheItem<K, V>>,
}

/// An active read transaction over the cache. The data is this cache is guaranteed to be
/// valid at the point in time the read is created. You may include items during a cache
/// miss via the "insert" function.
pub struct ARCacheReadTxn<'a, K, V>
where
    K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
    V: Clone + Debug + Sync + Send + 'static,
{
    caller: &'a ARCache<K, V>,
    // ro_txn to cache
    cache: HashMapReadTxn<'a, K, CacheItem<K, V>>,
    tlocal: Option<ReadCache<K, V>>,
    // channel to send stat information
    stat_tx: Sender<ReaderStatEvent>,
    // tx channel to send forward events.
    // tx: Sender<CacheEvent<K, V>>,
    hit_queue: Arc<ArrayQueue<CacheHitEvent>>,
    inc_queue: Arc<ArrayQueue<CacheIncludeEvent<K, V>>>,
    above_watermark: bool,
    hits: usize,
    tlocal_hits: usize,
    tlocal_includes: usize,
    reader_includes: usize,
}

unsafe impl<
        K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
        V: Clone + Debug + Sync + Send + 'static,
    > Send for ARCacheReadTxn<'_, K, V>
{
}
unsafe impl<
        K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
        V: Clone + Debug + Sync + Send + 'static,
    > Sync for ARCacheReadTxn<'_, K, V>
{
}

/// An active write transaction over the cache. The data in this cache is isolated
/// from readers, and may be rolled-back if an error occurs. Changes only become
/// globally visible once you call "commit". Items may be added to the cache on
/// a miss via "insert", and you can explicitly remove items by calling "remove".
pub struct ARCacheWriteTxn<'a, K, V>
where
    K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
    V: Clone + Debug + Sync + Send + 'static,
{
    caller: &'a ARCache<K, V>,
    // wr_txn to cache
    cache: HashMapWriteTxn<'a, K, CacheItem<K, V>>,
    // Cache of missed items (w_ dirty/clean)
    // On COMMIT we drain this to the main cache
    tlocal: Map<K, ThreadCacheItem<V>>,
    hit: UnsafeCell<Vec<u64>>,
    clear: UnsafeCell<bool>,
    above_watermark: bool,
}

/*
pub struct ArcReadSnapshot<K, V> {
    // How to communicate back to the caller the loads we did?
    tlocal: &mut Map<K, ThreadCacheItem<K, V>>,
}
*/

impl<
        K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
        V: Clone + Debug + Sync + Send + 'static,
    > CacheItem<K, V>
{
    fn to_vref(&self) -> Option<&V> {
        match &self {
            CacheItem::Freq(_, v) | CacheItem::Rec(_, v) => Some(v),
            _ => None,
        }
    }

    #[cfg(test)]
    fn to_state(&self) -> CacheState {
        match &self {
            CacheItem::Freq(_, _v) => CacheState::Freq,
            CacheItem::Rec(_, _v) => CacheState::Rec,
            CacheItem::GhostFreq(_) => CacheState::GhostFreq,
            CacheItem::GhostRec(_) => CacheState::GhostRec,
            CacheItem::Haunted(_) => CacheState::Haunted,
        }
    }
}

macro_rules! drain_ll_to_ghost {
    (
        $cache:expr,
        $ll:expr,
        $gf:expr,
        $gr:expr,
        $txid:expr
    ) => {{
        while $ll.len() > 0 {
            let n = $ll.pop();
            debug_assert!(!n.is_null());
            unsafe {
                // Set the item's evict txid.
                (*n).as_mut().txid = $txid;
            }
            let mut r = $cache.get_mut(unsafe { &(*n).as_mut().k });
            match r {
                Some(ref mut ci) => {
                    let mut next_state = match &ci {
                        CacheItem::Freq(n, _) => {
                            $gf.append_n(*n);
                            CacheItem::GhostFreq(*n)
                        }
                        CacheItem::Rec(n, _) => {
                            $gr.append_n(*n);
                            CacheItem::GhostRec(*n)
                        }
                        _ => {
                            // Impossible state!
                            unreachable!();
                        }
                    };
                    // Now change the state.
                    mem::swap(*ci, &mut next_state);
                }
                None => {
                    // Impossible state!
                    unreachable!();
                }
            }
        } // end while
    }};
}

macro_rules! evict_to_len {
    (
        $cache:expr,
        $ll:expr,
        $to_ll:expr,
        $size:expr,
        $txid:expr
    ) => {{
        debug_assert!($ll.len() >= $size);

        while $ll.len() > $size {
            let n = $ll.pop();
            debug_assert!(!n.is_null());
            let mut r = $cache.get_mut(unsafe { &(*n).as_mut().k });
            unsafe {
                // Set the item's evict txid.
                (*n).as_mut().txid = $txid;
            }
            match r {
                Some(ref mut ci) => {
                    let mut next_state = match &ci {
                        CacheItem::Freq(llp, _v) => {
                            debug_assert!(*llp == n);
                            // No need to extract, already popped!
                            // $ll.extract(*llp);
                            $to_ll.append_n(*llp);
                            CacheItem::GhostFreq(*llp)
                        }
                        CacheItem::Rec(llp, _v) => {
                            debug_assert!(*llp == n);
                            // No need to extract, already popped!
                            // $ll.extract(*llp);
                            $to_ll.append_n(*llp);
                            CacheItem::GhostRec(*llp)
                        }
                        _ => {
                            // Impossible state!
                            unreachable!();
                        }
                    };
                    // Now change the state.
                    mem::swap(*ci, &mut next_state);
                }
                None => {
                    // Impossible state!
                    unreachable!();
                }
            }
        }
    }};
}

macro_rules! evict_to_haunted_len {
    (
        $cache:expr,
        $ll:expr,
        $to_ll:expr,
        $size:expr,
        $txid:expr
    ) => {{
        debug_assert!($ll.len() >= $size);

        while $ll.len() > $size {
            let n = $ll.pop();
            debug_assert!(!n.is_null());
            $to_ll.append_n(n);
            let mut r = $cache.get_mut(unsafe { &(*n).as_mut().k });
            unsafe {
                // Set the item's evict txid.
                (*n).as_mut().txid = $txid;
            }
            match r {
                Some(ref mut ci) => {
                    // Now change the state.
                    let mut next_state = CacheItem::Haunted(n);
                    mem::swap(*ci, &mut next_state);
                }
                None => {
                    // Impossible state!
                    unreachable!();
                }
            };
        }
    }};
}

impl ARCacheBuilder {
    /// Create a new ARCache builder that you can configure before creation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure a new ARCache, that derives it's size based on your expected workload.
    ///
    /// The values are total number of items you want to have in memory, the number
    /// of read threads you expect concurrently, the expected average number of cache
    /// misses per read operation, and the expected average number of writes or write
    /// cache misses per operation. The following formula is assumed:
    ///
    /// `max + (threads * (max/16))`
    /// `    + (threads * avg number of misses per op)`
    /// `    + avg number of writes per transaction`
    ///
    /// The cache may still exceed your provided total, and inaccurate tuning numbers
    /// will yield a situation where you may use too-little ram, or too much. This could
    /// be to your read misses exceeding your expected amount causing the queues to have
    /// more items in them at a time, or your writes are larger than expected.
    ///
    /// If you set ex_ro_miss to zero, no read thread local cache will be configured, but
    /// space will still be reserved for channel communication.
    #[must_use]
    pub fn set_expected_workload(
        self,
        total: usize,
        threads: usize,
        ex_ro_miss: usize,
        ex_rw_miss: usize,
        read_cache: bool,
    ) -> Self {
        let total = isize::try_from(total).unwrap();
        let threads = isize::try_from(threads).unwrap();
        let ro_miss = isize::try_from(ex_ro_miss).unwrap();
        let wr_miss = isize::try_from(ex_rw_miss).unwrap();
        let ratio = isize::try_from(READ_THREAD_RATIO).unwrap();
        // I'd like to thank wolfram alpha ... for this magic.
        let max = -((ratio * ((ro_miss * threads) + wr_miss - total)) / (ratio + threads));
        let read_max = if read_cache { max / ratio } else { 0 };

        let max = usize::try_from(max).unwrap();
        let read_max = usize::try_from(read_max).unwrap();

        ARCacheBuilder {
            stats: self.stats,
            max: Some(max),
            read_max: Some(read_max),
            watermark: self.watermark,
        }
    }

    /// Configure a new ARCache, with a capacity of `max` main cache items and `read_max`
    /// Note that due to the way the cache operates, the number of items can and
    /// will exceed `max` on a regular basis, so you should consider using `set_expected_workload`
    /// and specifying your expected workload parameters to have a better derived
    /// cache size.
    #[must_use]
    pub fn set_size(self, max: usize, read_max: usize) -> Self {
        ARCacheBuilder {
            stats: self.stats,
            max: Some(max),
            read_max: Some(read_max),
            watermark: self.watermark,
        }
    }

    /// See [new_size] for more information. This allows manual configuration of the data
    /// tracking watermark. To disable this, set to 0. If watermark is greater than
    /// max, it will be clamped to max.
    #[must_use]
    pub fn set_watermark(self, watermark: usize) -> Self {
        ARCacheBuilder {
            stats: self.stats,
            max: self.max,
            read_max: self.read_max,
            watermark: Some(watermark),
        }
    }

    /// Import read/write hit stats from a previous execution of this cache.
    #[must_use]
    pub fn set_stats(self, stats: CacheStats) -> Self {
        ARCacheBuilder {
            stats: Some(stats),
            max: self.max,
            read_max: self.read_max,
            watermark: self.watermark,
        }
    }

    /// Consume this builder, returning a cache if successful. If configured parameters are
    /// missing or incorrect, a None will be returned.
    pub fn build<K, V>(self) -> Option<ARCache<K, V>>
    where
        K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
        V: Clone + Debug + Sync + Send + 'static,
    {
        let ARCacheBuilder {
            stats,
            max,
            read_max,
            watermark,
        } = self;

        let (max, read_max) = max.zip(read_max)?;

        let mut stats = stats.unwrap_or(CacheStats {
            reader_hits: 0,
            reader_tlocal_hits: 0,
            reader_tlocal_includes: 0,
            reader_includes: 0,
            write_hits: 0,
            write_includes: 0,
            write_modifies: 0,
            shared_max: 0,
            freq: 0,
            recent: 0,
            freq_evicts: 0,
            recent_evicts: 0,
            p_weight: 0,
            all_seen_keys: 0,
        });

        // Reset some values to 0 because else it doesn't really max sense ...
        stats.shared_max = 0;
        stats.freq = 0;
        stats.recent = 0;
        stats.all_seen_keys = 0;
        stats.p_weight = 0;

        // Ensure that p isn't too large. Could happen if the cache size was reduced from
        // a previous stats invocation.
        //
        // NOTE: I decided not to port this over, because it may cause issues in early cache start
        // up when the lists are empty.
        // stats.p_weight = stats.p_weight.clamp(0, max);

        let watermark = watermark.unwrap_or(if max < 128 { 0 } else { (max / 20) * 16 });
        let watermark = watermark.clamp(0, max);

        let (stat_tx, stat_rx) = unbounded();

        // The hit queue is reasonably cheap, so we can let this grow a bit.
        /*
        let chan_size = max / 20;
        let chan_size = if chan_size < 16 { 16 } else { chan_size };
        let chan_size = chan_size.clamp(0, 128);
        */
        let chan_size = 64;
        let hit_queue = Arc::new(ArrayQueue::new(chan_size));

        // this can oversize and take a lot of time to drain and manage, so we keep this bounded.
        // let chan_size = chan_size.clamp(0, 64);
        let chan_size = 32;
        let inc_queue = Arc::new(ArrayQueue::new(chan_size));

        let shared = RwLock::new(ArcShared {
            max,
            read_max,
            stat_tx,
            hit_queue: hit_queue.clone(),
            inc_queue: inc_queue.clone(),
            watermark,
        });
        let inner = Mutex::new(ArcInner {
            // We use p from the former stats.
            p: 0,
            freq: LL::new(),
            rec: LL::new(),
            ghost_freq: LL::new(),
            ghost_rec: LL::new(),
            haunted: LL::new(),
            stat_rx,
            hit_queue,
            inc_queue,
            min_txid: 0,
        });

        Some(ARCache {
            cache: HashMap::new(),
            shared,
            inner,
            stats: CowCell::new(stats),
            above_watermark: AtomicBool::new(true),
        })
    }
}

impl<
        K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
        V: Clone + Debug + Sync + Send + 'static,
    > ARCache<K, V>
{
    /// Use ARCacheBuilder instead
    #[deprecated(since = "0.2.20", note = "please use`ARCacheBuilder` instead")]
    pub fn new(
        total: usize,
        threads: usize,
        ex_ro_miss: usize,
        ex_rw_miss: usize,
        read_cache: bool,
    ) -> Self {
        ARCacheBuilder::default()
            .set_expected_workload(total, threads, ex_ro_miss, ex_rw_miss, read_cache)
            .build()
            .expect("Invaled cache parameters!")
    }

    /// Use ARCacheBuilder instead
    #[deprecated(since = "0.2.20", note = "please use`ARCacheBuilder` instead")]
    pub fn new_size(max: usize, read_max: usize) -> Self {
        ARCacheBuilder::default()
            .set_size(max, read_max)
            .build()
            .expect("Invaled cache parameters!")
    }

    /// Use ARCacheBuilder instead
    #[deprecated(since = "0.2.20", note = "please use`ARCacheBuilder` instead")]
    pub fn new_size_watermark(max: usize, read_max: usize, watermark: usize) -> Self {
        ARCacheBuilder::default()
            .set_size(max, read_max)
            .set_watermark(watermark)
            .build()
            .expect("Invaled cache parameters!")
    }

    /// Begin a read operation on the cache. This reader has a thread-local cache for items
    /// that are localled included via `insert`, and can communicate back to the main cache
    /// to safely include items.
    pub fn read(&self) -> ARCacheReadTxn<K, V> {
        let rshared = self.shared.read();
        let tlocal = if rshared.read_max > 0 {
            Some(ReadCache {
                set: Map::new(),
                read_size: rshared.read_max,
                tlru: LL::new(),
            })
        } else {
            None
        };
        let above_watermark = self.above_watermark.load(Ordering::Relaxed);
        ARCacheReadTxn {
            caller: self,
            cache: self.cache.read(),
            tlocal,
            stat_tx: rshared.stat_tx.clone(),
            hit_queue: rshared.hit_queue.clone(),
            inc_queue: rshared.inc_queue.clone(),
            above_watermark,
            hits: 0,
            tlocal_hits: 0,
            tlocal_includes: 0,
            reader_includes: 0,
        }
    }

    /// Begin a write operation on the cache. This writer has a thread-local store
    /// for all items that have been included or dirtied in the transactions, items
    /// may be removed from this cache (ie deleted, invalidated).
    pub fn write(&self) -> ARCacheWriteTxn<K, V> {
        let above_watermark = self.above_watermark.load(Ordering::Relaxed);
        ARCacheWriteTxn {
            caller: self,
            cache: self.cache.write(),
            tlocal: Map::new(),
            hit: UnsafeCell::new(Vec::new()),
            clear: UnsafeCell::new(false),
            above_watermark,
        }
    }

    /// View the statistics for this cache. These values are a snapshot of a point in
    /// time and may not be accurate at "this exact moment".
    pub fn view_stats(&self) -> CowCellReadTxn<CacheStats> {
        self.stats.read()
    }

    fn try_write(&self) -> Option<ARCacheWriteTxn<K, V>> {
        self.cache.try_write().map(|cache| {
            let above_watermark = self.above_watermark.load(Ordering::Relaxed);
            ARCacheWriteTxn {
                caller: self,
                cache,
                tlocal: Map::new(),
                hit: UnsafeCell::new(Vec::new()),
                clear: UnsafeCell::new(false),
                above_watermark,
            }
        })
    }

    fn try_quiesce(&self) {
        if let Some(wr_txn) = self.try_write() {
            wr_txn.commit()
        };
    }

    fn calc_p_freq(ghost_rec_len: usize, ghost_freq_len: usize, p: &mut usize) {
        let delta = if ghost_rec_len > ghost_freq_len {
            ghost_rec_len / ghost_freq_len
        } else {
            1
        };
        if delta < *p {
            *p -= delta
        } else {
            *p = 0
        }
    }

    fn calc_p_rec(cap: usize, ghost_rec_len: usize, ghost_freq_len: usize, p: &mut usize) {
        let delta = if ghost_freq_len > ghost_rec_len {
            ghost_freq_len / ghost_rec_len
        } else {
            1
        };
        if delta <= cap - *p {
            *p += delta
        } else {
            *p = cap
        }
    }

    fn drain_tlocal_inc<'a>(
        &'a self,
        cache: &mut HashMapWriteTxn<'a, K, CacheItem<K, V>>,
        inner: &mut ArcInner<K, V>,
        shared: &ArcShared<K, V>,
        stats: &mut CacheStats,
        tlocal: Map<K, ThreadCacheItem<V>>,
        commit_txid: u64,
    ) {
        // drain tlocal into the main cache.
        tlocal.into_iter().for_each(|(k, tcio)| {
            let r = cache.get_mut(&k);
            match (r, tcio) {
                (None, ThreadCacheItem::Present(tci, clean, size)) => {
                    assert!(clean);
                    let llp = inner.rec.append_k(CacheItemInner {
                        k: k.clone(),
                        txid: commit_txid,
                        size,
                    });
                    stats.write_includes += 1;
                    cache.insert(k, CacheItem::Rec(llp, tci));
                }
                (None, ThreadCacheItem::Removed(clean)) => {
                    assert!(clean);
                    // Mark this as haunted
                    let llp = inner.haunted.append_k(CacheItemInner {
                        k: k.clone(),
                        txid: commit_txid,
                        size: 1,
                    });
                    cache.insert(k, CacheItem::Haunted(llp));
                }
                (Some(ref mut ci), ThreadCacheItem::Removed(clean)) => {
                    assert!(clean);
                    // From whatever set we were in, pop and move to haunted.
                    let mut next_state = match ci {
                        CacheItem::Freq(llp, _v) => {
                            // println!("tlocal {:?} Freq -> Freq", k);
                            inner.freq.extract(*llp);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            inner.haunted.append_n(*llp);
                            CacheItem::Haunted(*llp)
                        }
                        CacheItem::Rec(llp, _v) => {
                            // println!("tlocal {:?} Rec -> Freq", k);
                            // Remove the node and put it into freq.
                            inner.rec.extract(*llp);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            inner.haunted.append_n(*llp);
                            CacheItem::Haunted(*llp)
                        }
                        CacheItem::GhostFreq(llp) => {
                            // println!("tlocal {:?} GhostFreq -> Freq", k);
                            inner.ghost_freq.extract(*llp);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            inner.haunted.append_n(*llp);
                            CacheItem::Haunted(*llp)
                        }
                        CacheItem::GhostRec(llp) => {
                            // println!("tlocal {:?} GhostRec -> Rec", k);
                            inner.ghost_rec.extract(*llp);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            inner.haunted.append_n(*llp);
                            CacheItem::Haunted(*llp)
                        }
                        CacheItem::Haunted(llp) => {
                            // println!("tlocal {:?} Haunted -> Rec", k);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            CacheItem::Haunted(*llp)
                        }
                    };
                    // Now change the state.
                    mem::swap(*ci, &mut next_state);
                }
                // TODO: https://github.com/rust-lang/rust/issues/68354 will stabilise
                // in 1.44 so we can prevent a need for a clone.
                (Some(ref mut ci), ThreadCacheItem::Present(ref tci, clean, size)) => {
                    assert!(clean);
                    //   * as we include each item, what state was it in before?
                    // It's in the cache - what action must we take?
                    let mut next_state = match ci {
                        CacheItem::Freq(llp, _v) => {
                            stats.write_modifies += 1;
                            // println!("tlocal {:?} Freq -> Freq", k);
                            inner.freq.extract(*llp);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            unsafe { (**llp).as_mut().size = size };
                            // Move the list item to it's head.
                            inner.freq.append_n(*llp);
                            // Update v.
                            CacheItem::Freq(*llp, (*tci).clone())
                        }
                        CacheItem::Rec(llp, _v) => {
                            stats.write_modifies += 1;
                            // println!("tlocal {:?} Rec -> Freq", k);
                            // Remove the node and put it into freq.
                            inner.rec.extract(*llp);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            unsafe { (**llp).as_mut().size = size };
                            inner.freq.append_n(*llp);
                            CacheItem::Freq(*llp, (*tci).clone())
                        }
                        CacheItem::GhostFreq(llp) => {
                            stats.write_includes += 1;
                            // println!("tlocal {:?} GhostFreq -> Freq", k);
                            // Ajdust p
                            Self::calc_p_freq(
                                inner.ghost_rec.len(),
                                inner.ghost_freq.len(),
                                &mut inner.p,
                            );
                            inner.ghost_freq.extract(*llp);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            unsafe { (**llp).as_mut().size = size };
                            inner.freq.append_n(*llp);
                            CacheItem::Freq(*llp, (*tci).clone())
                        }
                        CacheItem::GhostRec(llp) => {
                            stats.write_includes += 1;
                            // println!("tlocal {:?} GhostRec -> Rec", k);
                            // Ajdust p
                            Self::calc_p_rec(
                                shared.max,
                                inner.ghost_rec.len(),
                                inner.ghost_freq.len(),
                                &mut inner.p,
                            );
                            inner.ghost_rec.extract(*llp);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            unsafe { (**llp).as_mut().size = size };
                            inner.rec.append_n(*llp);
                            CacheItem::Rec(*llp, (*tci).clone())
                        }
                        CacheItem::Haunted(llp) => {
                            stats.write_includes += 1;
                            // println!("tlocal {:?} Haunted -> Rec", k);
                            inner.haunted.extract(*llp);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            unsafe { (**llp).as_mut().size = size };
                            inner.rec.append_n(*llp);
                            CacheItem::Rec(*llp, (*tci).clone())
                        }
                    };
                    // Now change the state.
                    mem::swap(*ci, &mut next_state);
                }
            }
        });
    }

    fn drain_hit_rx<'a>(
        &'a self,
        cache: &mut HashMapWriteTxn<'a, K, CacheItem<K, V>>,
        inner: &mut ArcInner<K, V>,
        commit_ts: Instant,
    ) {
        // * for each item
        // while let Ok(ce) = inner.rx.try_recv() {
        while let Some(ce) = inner.hit_queue.pop() {
            let CacheHitEvent { t, k_hash } = ce;
            if let Some(ref mut ci_slots) = unsafe { cache.get_slot_mut(k_hash) } {
                for ref mut ci in ci_slots.iter_mut() {
                    let mut next_state = match &ci.v {
                        CacheItem::Freq(llp, v) => {
                            // println!("rxhit {:?} Freq -> Freq", k);
                            inner.freq.touch(*llp);
                            CacheItem::Freq(*llp, v.clone())
                        }
                        CacheItem::Rec(llp, v) => {
                            // println!("rxhit {:?} Rec -> Freq", k);
                            inner.rec.extract(*llp);
                            inner.freq.append_n(*llp);
                            CacheItem::Freq(*llp, v.clone())
                        }
                        // While we can't add this from nothing, we can
                        // at least keep it in the ghost sets.
                        CacheItem::GhostFreq(llp) => {
                            // println!("rxhit {:?} GhostFreq -> GhostFreq", k);
                            inner.ghost_freq.touch(*llp);
                            CacheItem::GhostFreq(*llp)
                        }
                        CacheItem::GhostRec(llp) => {
                            // println!("rxhit {:?} GhostRec -> GhostRec", k);
                            inner.ghost_rec.touch(*llp);
                            CacheItem::GhostRec(*llp)
                        }
                        CacheItem::Haunted(llp) => {
                            // println!("rxhit {:?} Haunted -> Haunted", k);
                            // We can't do anything about this ...
                            CacheItem::Haunted(*llp)
                        }
                    };
                    mem::swap(&mut (*ci).v, &mut next_state);
                } // for each item in the bucket.
            }
            // Do nothing, it must have been evicted.

            // Stop processing the queue, we are up to "now".
            if t >= commit_ts {
                break;
            }
        }
    }

    fn drain_inc_rx<'a>(
        &'a self,
        cache: &mut HashMapWriteTxn<'a, K, CacheItem<K, V>>,
        inner: &mut ArcInner<K, V>,
        shared: &ArcShared<K, V>,
        commit_ts: Instant,
    ) {
        while let Some(ce) = inner.inc_queue.pop() {
            // Update if it was inc
            let CacheIncludeEvent {
                t,
                k,
                v: iv,
                txid,
                size,
            } = ce;
            let mut r = cache.get_mut(&k);
            match r {
                Some(ref mut ci) => {
                    let mut next_state = match &ci {
                        CacheItem::Freq(llp, _v) => {
                            if unsafe { (**llp).as_ref().txid >= txid } || inner.min_txid > txid {
                                // println!("rxinc {:?} Freq -> Freq (touch only)", k);
                                // Our cache already has a newer value, keep it.
                                inner.freq.touch(*llp);
                                None
                            } else {
                                // println!("rxinc {:?} Freq -> Freq (update)", k);
                                // The value is newer, update.
                                inner.freq.extract(*llp);
                                unsafe { (**llp).as_mut().txid = txid };
                                unsafe { (**llp).as_mut().size = size };
                                inner.freq.append_n(*llp);
                                Some(CacheItem::Freq(*llp, iv))
                            }
                        }
                        CacheItem::Rec(llp, v) => {
                            inner.rec.extract(*llp);
                            if unsafe { (**llp).as_ref().txid >= txid } || inner.min_txid > txid {
                                // println!("rxinc {:?} Rec -> Freq (touch only)", k);
                                inner.freq.append_n(*llp);
                                Some(CacheItem::Freq(*llp, v.clone()))
                            } else {
                                // println!("rxinc {:?} Rec -> Freq (update)", k);
                                unsafe { (**llp).as_mut().txid = txid };
                                unsafe { (**llp).as_mut().size = size };
                                inner.freq.append_n(*llp);
                                Some(CacheItem::Freq(*llp, iv))
                            }
                        }
                        CacheItem::GhostFreq(llp) => {
                            // Adjust p
                            Self::calc_p_freq(
                                inner.ghost_rec.len(),
                                inner.ghost_freq.len(),
                                &mut inner.p,
                            );
                            if unsafe { (**llp).as_ref().txid > txid } || inner.min_txid > txid {
                                // println!("rxinc {:?} GhostFreq -> GhostFreq", k);
                                // The cache version is newer, this is just a hit.
                                inner.ghost_freq.touch(*llp);
                                None
                            } else {
                                // This item is newer, so we can include it.
                                // println!("rxinc {:?} GhostFreq -> Rec", k);
                                inner.ghost_freq.extract(*llp);
                                unsafe { (**llp).as_mut().txid = txid };
                                unsafe { (**llp).as_mut().size = size };
                                inner.freq.append_n(*llp);
                                Some(CacheItem::Freq(*llp, iv))
                            }
                        }
                        CacheItem::GhostRec(llp) => {
                            // Adjust p
                            Self::calc_p_rec(
                                shared.max,
                                inner.ghost_rec.len(),
                                inner.ghost_freq.len(),
                                &mut inner.p,
                            );
                            if unsafe { (**llp).as_ref().txid > txid } || inner.min_txid > txid {
                                // println!("rxinc {:?} GhostRec -> GhostRec", k);
                                inner.ghost_rec.touch(*llp);
                                None
                            } else {
                                // println!("rxinc {:?} GhostRec -> Rec", k);
                                inner.ghost_rec.extract(*llp);
                                unsafe { (**llp).as_mut().txid = txid };
                                unsafe { (**llp).as_mut().size = size };
                                inner.rec.append_n(*llp);
                                Some(CacheItem::Rec(*llp, iv))
                            }
                        }
                        CacheItem::Haunted(llp) => {
                            if unsafe { (**llp).as_ref().txid > txid } || inner.min_txid > txid {
                                // println!("rxinc {:?} Haunted -> Haunted", k);
                                None
                            } else {
                                // println!("rxinc {:?} Haunted -> Rec", k);
                                inner.haunted.extract(*llp);
                                unsafe { (**llp).as_mut().txid = txid };
                                unsafe { (**llp).as_mut().size = size };
                                inner.rec.append_n(*llp);
                                Some(CacheItem::Rec(*llp, iv))
                            }
                        }
                    };
                    if let Some(ref mut next_state) = next_state {
                        mem::swap(*ci, next_state);
                    }
                }
                None => {
                    // It's not present - include it!
                    // println!("rxinc {:?} None -> Rec", k);
                    if txid >= inner.min_txid {
                        let llp = inner.rec.append_k(CacheItemInner {
                            k: k.clone(),
                            txid,
                            size,
                        });
                        cache.insert(k, CacheItem::Rec(llp, iv));
                    }
                }
            };

            // Stop processing the queue, we are up to "now".
            if t >= commit_ts {
                break;
            }
        }
    }

    fn drain_tlocal_hits<'a>(
        &'a self,
        cache: &mut HashMapWriteTxn<'a, K, CacheItem<K, V>>,
        inner: &mut ArcInner<K, V>,
        // shared: &ArcShared<K, V>,
        commit_txid: u64,
        hit: Vec<u64>,
    ) {
        // Stats updated by caller
        hit.into_iter().for_each(|k_hash| {
            // * everything hit must be in main cache now, so bring these
            //   all to the relevant item heads.
            // * Why do this last? Because the write is the "latest" we want all the fresh
            //   written items in the cache over the "read" hits, it gives us some aprox
            //   of time ordering, but not perfect.

            // Find the item in the cache.
            // * based on it's type, promote it in the correct list, or move it.
            // How does this prevent incorrect promotion from rec to freq? txid?
            // println!("Checking Hit ... {:?}", k);
            let mut r = unsafe { cache.get_slot_mut(k_hash) };
            match r {
                Some(ref mut ci_slots) => {
                    for ref mut ci in ci_slots.iter_mut() {
                        // This differs from above - we skip if we don't touch anything
                        // that was added in this txn. This is to prevent double touching
                        // anything that was included in a write.
                        let mut next_state = match &ci.v {
                            CacheItem::Freq(llp, v) => {
                                if unsafe { (**llp).as_ref().txid != commit_txid } {
                                    // println!("hit {:?} Freq -> Freq", k);
                                    inner.freq.touch(*llp);
                                    Some(CacheItem::Freq(*llp, v.clone()))
                                } else {
                                    None
                                }
                            }
                            CacheItem::Rec(llp, v) => {
                                if unsafe { (**llp).as_ref().txid != commit_txid } {
                                    // println!("hit {:?} Rec -> Freq", k);
                                    inner.rec.extract(*llp);
                                    inner.freq.append_n(*llp);
                                    Some(CacheItem::Freq(*llp, v.clone()))
                                } else {
                                    None
                                }
                            }
                            _ => {
                                // Ignore hits on items that may have been cleared.
                                None
                            }
                        };
                        // Now change the state.
                        if let Some(ref mut next_state) = next_state {
                            mem::swap(&mut (*ci).v, next_state);
                        }
                    } // for each ci in slots
                }
                None => {
                    // Impossible state!
                    unreachable!();
                }
            }
        });
    }

    fn drain_stat_rx<'a>(
        &'a self,
        inner: &mut ArcInner<K, V>,
        stats: &mut CacheStats,
        commit_ts: Instant,
    ) {
        while let Ok(ce) = inner.stat_rx.try_recv() {
            let ReaderStatEvent {
                t,
                hits,
                tlocal_hits,
                tlocal_includes,
                reader_includes,
            } = ce;

            stats.reader_tlocal_hits += tlocal_hits;
            stats.reader_tlocal_includes += tlocal_includes;
            stats.reader_hits += hits;
            stats.reader_includes += reader_includes;

            // Stop processing the queue, we are up to "now".
            if t >= commit_ts {
                break;
            }
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn evict<'a>(
        &'a self,
        cache: &mut HashMapWriteTxn<'a, K, CacheItem<K, V>>,
        inner: &mut ArcInner<K, V>,
        shared: &ArcShared<K, V>,
        stats: &mut CacheStats,
        commit_txid: u64,
    ) {
        debug_assert!(inner.p <= shared.max);
        // Convince the compiler copying is okay.
        let p = inner.p;
        stats.p_weight = p;

        if inner.rec.len() + inner.freq.len() > shared.max {
            // println!("Checking cache evict");
            /*
            println!(
                "from -> rec {:?}, freq {:?}",
                inner.rec.len(),
                inner.freq.len()
            );
            */
            let delta = (inner.rec.len() + inner.freq.len()) - shared.max;
            // We have overflowed by delta. As we are not "evicting as we go" we have to work out
            // what we should have evicted up to now.
            //
            // keep removing from rec until == p OR delta == 0, and if delta remains, then remove from freq.

            let rec_to_len = if inner.p == 0 {
                // println!("p == 0 => {:?}", inner.rec.len());
                debug_assert!(delta <= inner.rec.len());
                // We are fully weight to freq, so only remove in rec.
                inner.rec.len() - delta
            } else if inner.rec.len() > inner.p {
                // There is a partial weighting, how much do we need to move?
                let rec_delta = inner.rec.len() - inner.p;
                if rec_delta > delta {
                    /*
                    println!(
                        "p ({:?}) <= rec ({:?}), rec_delta ({:?}) > delta ({:?})",
                        inner.p,
                        inner.rec.len(),
                        rec_delta,
                        delta
                    );
                    */
                    // We will have removed enough through delta alone in rec.
                    inner.rec.len() - delta
                } else {
                    /*
                    println!(
                        "p ({:?}) <= rec ({:?}), rec_delta ({:?}) <= delta ({:?})",
                        inner.p,
                        inner.rec.len(),
                        rec_delta,
                        delta
                    );
                    */
                    // Remove the full delta, and excess will be removed from freq.
                    inner.rec.len() - rec_delta
                }
            } else {
                // rec is already below p, therefore we must need to remove in freq, and
                // we need to consider how much is in rec.
                // println!("p ({:?}) > rec ({:?})", inner.p, inner.rec.len());
                inner.rec.len()
            };

            // Now we can get the expected sizes;
            debug_assert!(shared.max >= rec_to_len);
            let freq_to_len = shared.max - rec_to_len;
            // println!("move to -> rec {:?}, freq {:?}", rec_to_len, freq_to_len);
            debug_assert!(freq_to_len + rec_to_len <= shared.max);

            stats.freq_evicts += inner.freq.len() - freq_to_len;
            stats.recent_evicts += inner.rec.len() - rec_to_len;

            evict_to_len!(
                cache,
                inner.rec,
                &mut inner.ghost_rec,
                rec_to_len,
                commit_txid
            );
            evict_to_len!(
                cache,
                inner.freq,
                &mut inner.ghost_freq,
                freq_to_len,
                commit_txid
            );

            // Finally, do an evict of the ghost sets if they are too long - these are weighted
            // inverse to the above sets. Note the freq to len in ghost rec, and rec to len in
            // ghost freq!
            if inner.ghost_rec.len() > (shared.max - p) {
                evict_to_haunted_len!(
                    cache,
                    inner.ghost_rec,
                    &mut inner.haunted,
                    freq_to_len,
                    commit_txid
                );
            }

            if inner.ghost_freq.len() > p {
                evict_to_haunted_len!(
                    cache,
                    inner.ghost_freq,
                    &mut inner.haunted,
                    rec_to_len,
                    commit_txid
                );
            }
        }
    }

    #[allow(clippy::unnecessary_mut_passed)]
    fn commit<'a>(
        &'a self,
        mut cache: HashMapWriteTxn<'a, K, CacheItem<K, V>>,
        tlocal: Map<K, ThreadCacheItem<V>>,
        hit: Vec<u64>,
        clear: bool,
        init_above_watermark: bool,
    ) {
        // What is the time?
        let commit_ts = Instant::now();
        let commit_txid = cache.get_txid();
        // Copy p + init cache sizes for adjustment.
        let mut inner = self.inner.lock();
        let shared = self.shared.read();
        let mut stat_guard = self.stats.write();
        let stats = stat_guard.get_mut();

        // Did we request to be cleared? If so, we move everything to a ghost set
        // that was live.
        //
        // we also set the min_txid watermark which prevents any inclusion of
        // any item that existed before this point in time.
        if clear {
            // Set the watermark of this txn.
            inner.min_txid = commit_txid;

            // Indicate that we evicted all to ghost/freq
            stats.freq_evicts += inner.freq.len();
            stats.recent_evicts += inner.rec.len();

            // Move everything active into ghost sets.
            drain_ll_to_ghost!(
                &mut cache,
                inner.freq,
                inner.ghost_freq,
                inner.ghost_rec,
                commit_txid
            );
            drain_ll_to_ghost!(
                &mut cache,
                inner.rec,
                inner.ghost_freq,
                inner.ghost_rec,
                commit_txid
            );
        }

        // Why is it okay to drain the rx/tlocal and create the cache in a temporary
        // oversize? Because these values in the queue/tlocal are already in memory
        // and we are moving them to the cache, we are not actually using any more
        // memory (well, not significantly more). By adding everything, then evicting
        // we also get better and more accurate hit patterns over the cache based on what
        // was used. This gives us an advantage over other cache types - we can see
        // patterns based on temporal usage that other caches can't, at the expense that
        // it may take some moments for that cache pattern to sync to the main thread.

        // stats.write_inc_or_mod += tlocal.len();

        self.drain_tlocal_inc(
            &mut cache,
            inner.deref_mut(),
            shared.deref(),
            stats,
            tlocal,
            commit_txid,
        );

        // drain rx until empty or time >= time.
        self.drain_inc_rx(&mut cache, inner.deref_mut(), shared.deref(), commit_ts);

        self.drain_hit_rx(&mut cache, inner.deref_mut(), commit_ts);

        // drain the tlocal hits into the main cache.
        stats.write_hits += hit.len();
        self.drain_tlocal_hits(&mut cache, inner.deref_mut(), commit_txid, hit);

        // now clean the space for each of the primary caches, evicting into the ghost sets.
        // * It's possible that both caches are now over-sized if rx was empty
        //   but wr inc many items.
        // * p has possibly changed width, causing a balance shift
        // * and ghost items have been included changing ghost list sizes.
        // so we need to do a clean up/balance of all the list lengths.
        self.evict(
            &mut cache,
            inner.deref_mut(),
            shared.deref(),
            stats,
            commit_txid,
        );

        self.drain_stat_rx(inner.deref_mut(), stats, commit_ts);

        stats.shared_max = shared.max;
        stats.freq = inner.freq.len();
        stats.recent = inner.rec.len();
        stats.all_seen_keys = cache.len();

        // Indicate if we are at/above watermark, so that read/writers begin to indicate their
        // hit events so we can start to setup/order our arc sets correctly.
        //
        // If we drop below this again, they'll go back to just insert/remove content only mode.
        if init_above_watermark {
            if (inner.freq.len() + inner.rec.len()) < shared.watermark {
                self.above_watermark.store(false, Ordering::Relaxed);
            }
        } else if (inner.freq.len() + inner.rec.len()) >= shared.watermark {
            self.above_watermark.store(true, Ordering::Relaxed);
        }

        // Commit the stats
        stat_guard.commit();
        // commit on the wr txn.
        cache.commit();
        // done!
        // eprintln!("quiesce took - {:?}", commit_ts.elapsed());
    }
}

impl<
        'a,
        K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
        V: Clone + Debug + Sync + Send + 'static,
    > ARCacheWriteTxn<'a, K, V>
{
    /// Commit the changes of this writer, making them globally visible. This causes
    /// all items written to this thread's local store to become visible in the main
    /// cache.
    ///
    /// To rollback (abort) and operation, just do not call commit (consider std::mem::drop
    /// on the write transaction)
    pub fn commit(self) {
        self.caller.commit(
            self.cache,
            self.tlocal,
            self.hit.into_inner(),
            self.clear.into_inner(),
            self.above_watermark,
        )
    }

    /// Clear all items of the cache. This operation does not take effect until you commit.
    /// After calling "clear", you may then include new items which will be stored thread
    /// locally until you commit.
    pub fn clear(&mut self) {
        // Mark that we have been requested to clear the cache.
        unsafe {
            let clear_ptr = self.clear.get();
            *clear_ptr = true;
        }
        // Dump the hit log.
        unsafe {
            let hit_ptr = self.hit.get();
            (*hit_ptr).clear();
        }
        // Dump the thread local state.
        self.tlocal.clear();
        // From this point any get will miss on the main cache.
        // Inserts are accepted.
    }

    /// Attempt to retieve a k-v pair from the cache. If it is present in the main cache OR
    /// the thread local cache, a `Some` is returned, else you will recieve a `None`. On a
    /// `None`, you must then consult the external data source that this structure is acting
    /// as a cache for.
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord,
    {
        let k_hash: u64 = self.cache.prehash(k);

        let r: Option<&V> = if let Some(tci) = self.tlocal.get(k) {
            match tci {
                ThreadCacheItem::Present(v, _clean, _size) => {
                    let v = v as *const _;
                    unsafe { Some(&(*v)) }
                }
                ThreadCacheItem::Removed(_clean) => {
                    return None;
                }
            }
        } else {
            // If we have been requested to clear, the main cache is "empty"
            // but we can't do that until a commit, so we just flag it and avoid.
            let is_cleared = unsafe {
                let clear_ptr = self.clear.get();
                *clear_ptr
            };
            if !is_cleared {
                if let Some(v) = self.cache.get_prehashed(k, k_hash) {
                    (*v).to_vref()
                } else {
                    None
                }
            } else {
                None
            }
        };
        // How do we track this was a hit?
        // Remember, we don't track misses - they are *implied* by the fact they'll trigger
        // an inclusion from the external system. Subsequent, any further re-hit on an
        // included value WILL be tracked, allowing arc to adjust appropriately.
        if self.above_watermark && r.is_some() {
            unsafe {
                let hit_ptr = self.hit.get();
                (*hit_ptr).push(k_hash);
            }
        }
        r
    }

    /// Determine if this cache contains the following key.
    pub fn contains_key<'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord,
    {
        self.get(k).is_some()
    }

    /// Add a value to the cache. This may be because you have had a cache miss and
    /// now wish to include in the thread local storage, or because you have written
    /// a new value and want it to be submitted for caching. This item is marked as
    /// clean, IE you have synced it to whatever associated store exists.
    pub fn insert(&mut self, k: K, v: V) {
        self.tlocal.insert(k, ThreadCacheItem::Present(v, true, 1));
    }

    /// Insert an item to the cache, with an associated weight/size factor. See also [insert]
    pub fn insert_sized(&mut self, k: K, v: V, size: usize) {
        self.tlocal
            .insert(k, ThreadCacheItem::Present(v, true, size));
    }

    /// Remove this value from the thread local cache IE mask from from being
    /// returned until this thread performs an insert. This item is marked as clean
    /// IE you have synced it to whatever associated store exists.
    pub fn remove(&mut self, k: K) {
        self.tlocal.insert(k, ThreadCacheItem::Removed(true));
    }

    /// Add a value to the cache. This may be because you have had a cache miss and
    /// now wish to include in the thread local storage, or because you have written
    /// a new value and want it to be submitted for caching. This item is marked as
    /// dirty, because you have *not* synced it. You MUST call iter_mut_mark_clean before calling
    /// `commit` on this transaction, or a panic will occur.
    pub fn insert_dirty(&mut self, k: K, v: V) {
        self.tlocal.insert(k, ThreadCacheItem::Present(v, false, 1));
    }

    /// Insert a dirty item to the cache, with an associated weight/size factor. See also [insert_dirty]
    pub fn insert_dirty_sized(&mut self, k: K, v: V, size: usize) {
        self.tlocal
            .insert(k, ThreadCacheItem::Present(v, false, size));
    }

    /// Remove this value from the thread local cache IE mask from from being
    /// returned until this thread performs an insert. This item is marked as
    /// dirty, because you have *not* synced it. You MUST call iter_mut_mark_clean before calling
    /// `commit` on this transaction, or a panic will occur.
    pub fn remove_dirty(&mut self, k: K) {
        self.tlocal.insert(k, ThreadCacheItem::Removed(false));
    }

    /// Determines if dirty elements exist in this cache or not.
    pub fn is_dirty(&self) -> bool {
        self.iter_dirty().take(1).next().is_some()
    }

    /// Yields an iterator over all values that are currently dirty. As the iterator
    /// progresses, items will NOT be marked clean. This allows you to examine
    /// any currently dirty items in the cache.
    pub fn iter_dirty(&self) -> impl Iterator<Item = (&K, Option<&V>)> {
        self.tlocal
            .iter()
            .filter(|(_k, v)| match v {
                ThreadCacheItem::Present(_v, c, _size) => !c,
                ThreadCacheItem::Removed(c) => !c,
            })
            .map(|(k, v)| {
                // Get the data.
                let data = match v {
                    ThreadCacheItem::Present(v, _c, _size) => Some(v),
                    ThreadCacheItem::Removed(_c) => None,
                };
                (k, data)
            })
    }

    /// Yields a mutable iterator over all values that are currently dirty. As the iterator
    /// progresses, items will NOT be marked clean. This allows you to modify and
    /// change any currently dirty items as required.
    pub fn iter_mut_dirty(&mut self) -> impl Iterator<Item = (&K, Option<&mut V>)> {
        self.tlocal
            .iter_mut()
            .filter(|(_k, v)| match v {
                ThreadCacheItem::Present(_v, c, _size) => !c,
                ThreadCacheItem::Removed(c) => !c,
            })
            .map(|(k, v)| {
                // Get the data.
                let data = match v {
                    ThreadCacheItem::Present(v, _c, _size) => Some(v),
                    ThreadCacheItem::Removed(_c) => None,
                };
                (k, data)
            })
    }

    /// Yields an iterator over all values that are currently dirty. As the iterator
    /// progresses, items will be marked clean. This is where you should sync dirty
    /// cache content to your associated store. The iterator is K, Option<V>, where
    /// the Option<V> indicates if the item has been remove (None) or is updated (Some).
    pub fn iter_mut_mark_clean(&mut self) -> impl Iterator<Item = (&K, Option<&mut V>)> {
        self.tlocal
            .iter_mut()
            .filter(|(_k, v)| match v {
                ThreadCacheItem::Present(_v, c, _size) => !c,
                ThreadCacheItem::Removed(c) => !c,
            })
            .map(|(k, v)| {
                // Mark it clean.
                match v {
                    ThreadCacheItem::Present(_v, c, _size) => *c = true,
                    ThreadCacheItem::Removed(c) => *c = true,
                }
                // Get the data.
                let data = match v {
                    ThreadCacheItem::Present(v, _c, _size) => Some(v),
                    ThreadCacheItem::Removed(_c) => None,
                };
                (k, data)
            })
    }

    #[cfg(test)]
    pub(crate) fn iter_rec(&self) -> impl Iterator<Item = &K> {
        self.cache.values().filter_map(|ci| match &ci {
            CacheItem::Rec(lln, _) => unsafe {
                let cii = &*((**lln).k.as_ptr());
                Some(&cii.k)
            },
            _ => None,
        })
    }

    #[cfg(test)]
    pub(crate) fn iter_ghost_rec(&self) -> impl Iterator<Item = &K> {
        self.cache.values().filter_map(|ci| match &ci {
            CacheItem::GhostRec(lln) => unsafe {
                let cii = &*((**lln).k.as_ptr());
                Some(&cii.k)
            },
            _ => None,
        })
    }

    #[cfg(test)]
    pub(crate) fn iter_ghost_freq(&self) -> impl Iterator<Item = &K> {
        self.cache.values().filter_map(|ci| match &ci {
            CacheItem::GhostFreq(lln) => unsafe {
                let cii = &*((**lln).k.as_ptr());
                Some(&cii.k)
            },
            _ => None,
        })
    }

    #[cfg(test)]
    pub(crate) fn peek_hit(&self) -> &[u64] {
        let hit_ptr = self.hit.get();
        unsafe { &(*hit_ptr) }
    }

    #[cfg(test)]
    pub(crate) fn peek_cache<'b, Q: ?Sized>(&'a self, k: &'b Q) -> CacheState
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord,
    {
        if let Some(v) = self.cache.get(k) {
            (*v).to_state()
        } else {
            CacheState::None
        }
    }

    #[cfg(test)]
    pub(crate) fn peek_stat(&self) -> CStat {
        let inner = self.caller.inner.lock();
        let shared = self.caller.shared.read();
        CStat {
            max: shared.max,
            cache: self.cache.len(),
            tlocal: self.tlocal.len(),
            freq: inner.freq.len(),
            rec: inner.rec.len(),
            ghost_freq: inner.ghost_freq.len(),
            ghost_rec: inner.ghost_rec.len(),
            haunted: inner.haunted.len(),
            p: inner.p,
        }
    }

    // get_mut
    //  If it's in tlocal, return that as get_mut
    // if it's in the cache, clone to tlocal, then get_mut to tlock
    // if not, return none.

    // to_snapshot
}

impl<
        'a,
        K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
        V: Clone + Debug + Sync + Send + 'static,
    > ARCacheReadTxn<'a, K, V>
{
    /// Attempt to retieve a k-v pair from the cache. If it is present in the main cache OR
    /// the thread local cache, a `Some` is returned, else you will recieve a `None`. On a
    /// `None`, you must then consult the external data source that this structure is acting
    /// as a cache for.
    pub fn get<'b, Q: ?Sized>(&'b mut self, k: &'b Q) -> Option<&'b V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord,
    {
        let k_hash: u64 = self.cache.prehash(k);

        let r: Option<&V> = self
            .tlocal
            .as_ref()
            .and_then(|cache| {
                cache.set.get(k).map(|v| unsafe {
                    // Indicate a hit on the tlocal cache.
                    #[allow(mutable_transmutes)]
                    let tlocal_hits = std::mem::transmute::<&usize, &mut usize>(&self.tlocal_hits);
                    *tlocal_hits += 1;

                    if self.above_watermark {
                        let _ = self.hit_queue.push(CacheHitEvent {
                            t: Instant::now(),
                            k_hash,
                        });
                    }
                    let v = &(**v).as_ref().v as *const _;
                    // This discards the lifetime and repins it to &'b.
                    &(*v)
                })
            })
            .or_else(|| {
                self.cache.get_prehashed(k, k_hash).and_then(|v| {
                    (*v).to_vref().map(|vin| unsafe {
                        // Indicate a hit on the main cache.
                        #[allow(mutable_transmutes)]
                        let hits = std::mem::transmute::<&usize, &mut usize>(&self.hits);
                        *hits += 1;

                        if self.above_watermark {
                            let _ = self.hit_queue.push(CacheHitEvent {
                                t: Instant::now(),
                                k_hash,
                            });
                        }

                        let vin = vin as *const _;
                        &(*vin)
                    })
                })
            });

        r
    }

    /// Determine if this cache contains the following key.
    pub fn contains_key<'b, Q: ?Sized>(&mut self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord,
    {
        self.get(k).is_some()
    }

    /// Insert an item to the cache, with an associated weight/size factor. See also [insert]
    pub fn insert_sized(&mut self, k: K, v: V, size: usize) {
        let mut v = v;
        // Send a copy forward through time and space.
        // let _ = self.tx.try_send(
        self.reader_includes += 1;
        let _ = self.inc_queue.push(CacheIncludeEvent {
            t: Instant::now(),
            k: k.clone(),
            v: v.clone(),
            txid: self.cache.get_txid(),
            size,
        });

        // We have a cache, so lets update it.
        if let Some(ref mut cache) = self.tlocal {
            self.tlocal_includes += 1;
            let n = if cache.tlru.len() >= cache.read_size {
                let n = cache.tlru.pop();
                // swap the old_key/old_val out
                let mut k_clone = k.clone();
                unsafe {
                    mem::swap(&mut k_clone, &mut (*n).as_mut().k);
                    mem::swap(&mut v, &mut (*n).as_mut().v);
                }
                // remove old K from the tree:
                cache.set.remove(&k_clone);
                n
            } else {
                // Just add it!
                cache.tlru.append_k(ReadCacheItem {
                    k: k.clone(),
                    v,
                    size,
                })
            };
            let r = cache.set.insert(k, n);
            // There should never be a previous value.
            assert!(r.is_none());
        }
    }

    /// Add a value to the cache. This may be because you have had a cache miss and
    /// now wish to include in the thread local storage.
    ///
    /// Note that is invalid to insert an item who's key already exists in this thread local cache,
    /// and this is asserted IE will panic if you attempt this. It is also invalid for you to insert
    /// a value that does not match the source-of-truth state, IE inserting a different
    /// value than another thread may percieve. This is a *read* thread, so you should only be adding
    /// values that are relevant to this read transaction and this point in time. If you do not
    /// heed this warning, you may alter the fabric of time and space and have some interesting
    /// distortions in your data over time.
    pub fn insert(&mut self, k: K, v: V) {
        self.insert_sized(k, v, 1)
    }
}

impl<
        'a,
        K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
        V: Clone + Debug + Sync + Send + 'static,
    > Drop for ARCacheReadTxn<'a, K, V>
{
    fn drop(&mut self) {
        let _ = self.stat_tx.send(ReaderStatEvent {
            t: Instant::now(),
            hits: self.hits,
            tlocal_hits: self.tlocal_hits,
            tlocal_includes: self.tlocal_includes,
            reader_includes: self.reader_includes,
        });
        self.caller.try_quiesce();
    }
}

#[cfg(test)]
mod tests {
    use super::ARCache as Arc;
    use super::ARCacheBuilder;
    use super::CStat;
    use super::CacheState;

    #[test]
    fn test_cache_arc_basic() {
        let arc: Arc<usize, usize> = ARCacheBuilder::default()
            .set_size(4, 4)
            .build()
            .expect("Invaled cache parameters!");
        let mut wr_txn = arc.write();

        assert!(wr_txn.get(&1) == None);
        assert!(wr_txn.peek_hit().len() == 0);
        wr_txn.insert(1, 1);
        assert!(wr_txn.get(&1) == Some(&1));
        assert!(wr_txn.peek_hit().len() == 1);

        wr_txn.commit();

        // Now we start the second txn, and see if it's in there.
        let wr_txn = arc.write();
        assert!(wr_txn.peek_cache(&1) == CacheState::Rec);
        assert!(wr_txn.get(&1) == Some(&1));
        assert!(wr_txn.peek_hit().len() == 1);
        wr_txn.commit();
        // And now check it's moved to Freq due to the extra
        let wr_txn = arc.write();
        assert!(wr_txn.peek_cache(&1) == CacheState::Freq);
        println!("{:?}", wr_txn.peek_stat());
    }

    #[test]
    fn test_cache_evict() {
        println!("== 1");
        let arc: Arc<usize, usize> = ARCacheBuilder::default()
            .set_size(4, 4)
            .build()
            .expect("Invaled cache parameters!");
        let mut wr_txn = arc.write();
        assert!(
            CStat {
                max: 4,
                cache: 0,
                tlocal: 0,
                freq: 0,
                rec: 0,
                ghost_freq: 0,
                ghost_rec: 0,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );

        // In the first txn we insert 4 items.
        wr_txn.insert(1, 1);
        wr_txn.insert(2, 2);
        wr_txn.insert(3, 3);
        wr_txn.insert(4, 4);

        assert!(
            CStat {
                max: 4,
                cache: 0,
                tlocal: 4,
                freq: 0,
                rec: 0,
                ghost_freq: 0,
                ghost_rec: 0,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );
        wr_txn.commit();

        // Now we start the second txn, and check the stats.
        println!("== 2");
        let wr_txn = arc.write();
        assert!(
            CStat {
                max: 4,
                cache: 4,
                tlocal: 0,
                freq: 0,
                rec: 4,
                ghost_freq: 0,
                ghost_rec: 0,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );

        // Now touch two items, this promote to the freq set.
        // Remember, a double hit doesn't weight any more than 1 hit.
        assert!(wr_txn.get(&1) == Some(&1));
        assert!(wr_txn.get(&1) == Some(&1));
        assert!(wr_txn.get(&2) == Some(&2));

        wr_txn.commit();

        // Now we start the third txn, and check the stats.
        println!("== 3");
        let mut wr_txn = arc.write();
        assert!(
            CStat {
                max: 4,
                cache: 4,
                tlocal: 0,
                freq: 2,
                rec: 2,
                ghost_freq: 0,
                ghost_rec: 0,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );
        // Add one more item, this will trigger an evict.
        wr_txn.insert(5, 5);
        wr_txn.commit();

        // Now we start the fourth txn, and check the stats.
        println!("== 4");
        let wr_txn = arc.write();
        println!("stat -> {:?}", wr_txn.peek_stat());
        assert!(
            CStat {
                max: 4,
                cache: 5,
                tlocal: 0,
                freq: 2,
                rec: 2,
                ghost_freq: 0,
                ghost_rec: 1,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );
        // And assert what's in the sets to be sure of what went where.
        // 🚨 Can no longer peek these with hashmap backing as the keys may
        // be evicted out-of-order, but the stats are correct!

        // Now touch the two recent items to bring them also to freq

        let rec_set: Vec<usize> = wr_txn.iter_rec().take(2).copied().collect();
        assert!(wr_txn.get(&rec_set[0]) == Some(&rec_set[0]));
        assert!(wr_txn.get(&rec_set[1]) == Some(&rec_set[1]));

        wr_txn.commit();

        // Now we start the fifth txn, and check the stats.
        println!("== 5");
        let mut wr_txn = arc.write();
        println!("stat -> {:?}", wr_txn.peek_stat());
        assert!(
            CStat {
                max: 4,
                cache: 5,
                tlocal: 0,
                freq: 4,
                rec: 0,
                ghost_freq: 0,
                ghost_rec: 1,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );
        // And assert what's in the sets to be sure of what went where.
        // 🚨 Can no longer peek these with hashmap backing as the keys may
        // be evicted out-of-order, but the stats are correct!

        // Now touch the one item that's in ghost rec - this will trigger
        // an evict from ghost freq
        let grec: usize = wr_txn.iter_ghost_rec().take(1).copied().next().unwrap();
        wr_txn.insert(grec, grec);
        assert!(wr_txn.get(&grec) == Some(&grec));
        // When we add 3, we are basically issuing a demand that the rec set should be
        // allowed to grow as we had a potential cache miss here.
        wr_txn.commit();

        // Now we start the sixth txn, and check the stats.
        println!("== 6");
        let mut wr_txn = arc.write();
        println!("stat -> {:?}", wr_txn.peek_stat());
        assert!(
            CStat {
                max: 4,
                cache: 5,
                tlocal: 0,
                freq: 3,
                rec: 1,
                ghost_freq: 1,
                ghost_rec: 0,
                haunted: 0,
                p: 1
            } == wr_txn.peek_stat()
        );
        // And assert what's in the sets to be sure of what went where.
        // 🚨 Can no longer peek these with hashmap backing as the keys may
        // be evicted out-of-order, but the stats are correct!
        assert!(wr_txn.peek_cache(&grec) == CacheState::Rec);

        // Right, seventh txn - we show how a cache scan doesn't cause p shifting or evict.
        // tl;dr - attempt to include a bunch in a scan, and it will be ignored as p is low,
        // and any miss on rec won't shift p unless it's in the ghost rec.
        wr_txn.insert(10, 10);
        wr_txn.insert(11, 11);
        wr_txn.insert(12, 12);
        wr_txn.commit();

        println!("== 7");
        let mut wr_txn = arc.write();
        println!("stat -> {:?}", wr_txn.peek_stat());
        assert!(
            CStat {
                max: 4,
                cache: 8,
                tlocal: 0,
                freq: 3,
                rec: 1,
                ghost_freq: 1,
                ghost_rec: 3,
                haunted: 0,
                p: 1
            } == wr_txn.peek_stat()
        );
        // 🚨 Can no longer peek these with hashmap backing as the keys may
        // be evicted out-of-order, but the stats are correct!

        // Eight txn - now that we had a demand for items before, we re-demand them - this will trigger
        // a shift in p, causing some more to be in the rec cache.
        let grec_set: Vec<usize> = wr_txn.iter_ghost_rec().take(3).copied().collect();

        grec_set.iter().for_each(|i| wr_txn.insert(*i, *i));
        wr_txn.commit();

        println!("== 8");
        let mut wr_txn = arc.write();
        println!("stat -> {:?}", wr_txn.peek_stat());
        assert!(
            CStat {
                max: 4,
                cache: 8,
                tlocal: 0,
                freq: 0,
                rec: 4,
                ghost_freq: 4,
                ghost_rec: 0,
                haunted: 0,
                p: 4
            } == wr_txn.peek_stat()
        );

        grec_set
            .iter()
            .for_each(|i| assert!(wr_txn.peek_cache(i) == CacheState::Rec));

        // Now lets go back the other way - we want freq items to come back.
        let gfreq_set: Vec<usize> = wr_txn.iter_ghost_freq().take(4).copied().collect();

        gfreq_set.iter().for_each(|i| wr_txn.insert(*i, *i));
        wr_txn.commit();

        println!("== 9");
        let wr_txn = arc.write();
        println!("stat -> {:?}", wr_txn.peek_stat());
        assert!(
            CStat {
                max: 4,
                cache: 8,
                tlocal: 0,
                freq: 4,
                rec: 0,
                ghost_freq: 0,
                ghost_rec: 4,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );
        // 🚨 Can no longer peek these with hashmap backing as the keys may
        // be evicted out-of-order, but the stats are correct!
        gfreq_set
            .iter()
            .for_each(|i| assert!(wr_txn.peek_cache(i) == CacheState::Freq));

        // And done!
        wr_txn.commit();
        // See what stats did
        let stats = arc.view_stats();
        println!("{:?}", *stats);
    }

    #[test]
    fn test_cache_concurrent_basic() {
        // Now we want to check some basic interactions of read and write together.

        // Setup the cache.
        let arc: Arc<usize, usize> = ARCacheBuilder::default()
            .set_size(4, 4)
            .build()
            .expect("Invaled cache parameters!");
        // start a rd
        {
            let mut rd_txn = arc.read();
            // add items to the rd
            rd_txn.insert(1, 1);
            rd_txn.insert(2, 2);
            rd_txn.insert(3, 3);
            rd_txn.insert(4, 4);
            // Should be in the tlocal
            // assert!(rd_txn.get(&1).is_some());
            // assert!(rd_txn.get(&2).is_some());
            // assert!(rd_txn.get(&3).is_some());
            // assert!(rd_txn.get(&4).is_some());
            // end the rd
        }
        arc.try_quiesce();
        // What state is the cache now in?
        println!("== 2");
        let wr_txn = arc.write();
        println!("{:?}", wr_txn.peek_stat());
        assert!(
            CStat {
                max: 4,
                cache: 4,
                tlocal: 0,
                freq: 0,
                rec: 4,
                ghost_freq: 0,
                ghost_rec: 0,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );
        assert!(wr_txn.peek_cache(&1) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&2) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&3) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&4) == CacheState::Rec);
        // Magic! Without a single write op we included items!
        // Lets have the read touch two items, and then add two new.
        // This will trigger evict on 1/2
        {
            let mut rd_txn = arc.read();
            // add items to the rd
            assert!(rd_txn.get(&3) == Some(&3));
            assert!(rd_txn.get(&4) == Some(&4));
            rd_txn.insert(5, 5);
            rd_txn.insert(6, 6);
            // end the rd
        }
        // Now commit and check the state.
        wr_txn.commit();
        println!("== 3");
        let wr_txn = arc.write();
        assert!(
            CStat {
                max: 4,
                cache: 6,
                tlocal: 0,
                freq: 2,
                rec: 2,
                ghost_freq: 0,
                ghost_rec: 2,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );
        assert!(wr_txn.peek_cache(&1) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&2) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&3) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&4) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&5) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&6) == CacheState::Rec);

        // Now trigger hits on 1/2 which will cause a shift in P.
        {
            let mut rd_txn = arc.read();
            // add items to the rd
            rd_txn.insert(1, 1);
            rd_txn.insert(2, 2);
            // end the rd
        }

        wr_txn.commit();
        println!("== 4");
        let wr_txn = arc.write();
        assert!(
            CStat {
                max: 4,
                cache: 6,
                tlocal: 0,
                freq: 2,
                rec: 2,
                ghost_freq: 0,
                ghost_rec: 2,
                haunted: 0,
                p: 2
            } == wr_txn.peek_stat()
        );
        assert!(wr_txn.peek_cache(&1) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&2) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&3) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&4) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&5) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&6) == CacheState::GhostRec);
        // See what stats did
        let stats = arc.view_stats();
        println!("stats 1: {:?}", *stats);
        assert!(stats.reader_hits == 2);
        assert!(stats.reader_includes == 8);
        assert!(stats.reader_tlocal_includes == 8);
        assert!(stats.reader_tlocal_hits == 0);
    }

    // Test edge cases that are horrifying and could destroy peoples lives
    // and sanity.
    #[test]
    fn test_cache_concurrent_cursed_1() {
        // Case 1 - It's possible for a read transaction to last for a long time,
        // and then have a cache include, which may cause an attempt to include
        // an outdated value into the cache. To handle this the haunted set exists
        // so that all keys and their eviction ids are always tracked for all of time
        // to ensure that we never incorrectly include a value that may have been updated
        // more recently.
        let arc: Arc<usize, usize> = ARCacheBuilder::default()
            .set_size(4, 4)
            .build()
            .expect("Invaled cache parameters!");

        // Start a wr
        let mut wr_txn = arc.write();
        // Start a rd
        let mut rd_txn = arc.read();
        // Add the value 1,1 via the wr.
        wr_txn.insert(1, 1);

        // assert 1 is not in rd.
        assert!(rd_txn.get(&1) == None);

        // Commit wr
        wr_txn.commit();
        // Even after the commit, it's not in rd.
        assert!(rd_txn.get(&1) == None);
        // begin wr
        let mut wr_txn = arc.write();
        // We now need to flood the cache, to cause ghost rec eviction.
        wr_txn.insert(10, 1);
        wr_txn.insert(11, 1);
        wr_txn.insert(12, 1);
        wr_txn.insert(13, 1);
        wr_txn.insert(14, 1);
        wr_txn.insert(15, 1);
        wr_txn.insert(16, 1);
        wr_txn.insert(17, 1);
        // commit wr
        wr_txn.commit();

        // begin wr
        let wr_txn = arc.write();
        // assert that 1 is haunted.
        assert!(wr_txn.peek_cache(&1) == CacheState::Haunted);
        // assert 1 is not in rd.
        assert!(rd_txn.get(&1) == None);
        // now that 1 is hanuted, in rd attempt to insert 1, X
        rd_txn.insert(1, 100);
        // commit wr
        wr_txn.commit();

        // start wr
        let wr_txn = arc.write();
        // assert that 1 is still haunted.
        assert!(wr_txn.peek_cache(&1) == CacheState::Haunted);
        // assert that 1, x is in rd.
        assert!(rd_txn.get(&1) == Some(&100));
        // done!
    }

    #[test]
    fn test_cache_clear() {
        let arc: Arc<usize, usize> = ARCacheBuilder::default()
            .set_size(4, 4)
            .build()
            .expect("Invaled cache parameters!");

        // Start a wr
        let mut wr_txn = arc.write();
        // Add a bunch of values, and touch some twice.
        wr_txn.insert(10, 10);
        wr_txn.insert(11, 11);
        wr_txn.insert(12, 12);
        wr_txn.insert(13, 13);
        wr_txn.insert(14, 14);
        wr_txn.insert(15, 15);
        wr_txn.insert(16, 16);
        wr_txn.insert(17, 17);
        wr_txn.commit();
        // Begin a new write.
        let wr_txn = arc.write();

        // Touch two values that are in the rec set.
        let rec_set: Vec<usize> = wr_txn.iter_rec().take(2).copied().collect();
        println!("{:?}", rec_set);
        assert!(wr_txn.get(&rec_set[0]) == Some(&rec_set[0]));
        assert!(wr_txn.get(&rec_set[1]) == Some(&rec_set[1]));

        // commit wr
        wr_txn.commit();
        // Begin a new write.
        let mut wr_txn = arc.write();
        println!("stat -> {:?}", wr_txn.peek_stat());
        assert!(
            CStat {
                max: 4,
                cache: 8,
                tlocal: 0,
                freq: 2,
                rec: 2,
                ghost_freq: 0,
                ghost_rec: 4,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );

        // Clear
        wr_txn.clear();
        // Now commit
        wr_txn.commit();
        // Now check their states.
        let wr_txn = arc.write();
        // See what stats did
        println!("stat -> {:?}", wr_txn.peek_stat());
        // stat -> CStat { max: 4, cache: 8, tlocal: 0, freq: 0, rec: 0, ghost_freq: 2, ghost_rec: 6, haunted: 0, p: 0 }
        assert!(
            CStat {
                max: 4,
                cache: 8,
                tlocal: 0,
                freq: 0,
                rec: 0,
                ghost_freq: 2,
                ghost_rec: 6,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );
        let stats = arc.view_stats();
        println!("{:?}", *stats);
    }

    #[test]
    fn test_cache_clear_rollback() {
        let arc: Arc<usize, usize> = ARCacheBuilder::default()
            .set_size(4, 4)
            .build()
            .expect("Invaled cache parameters!");

        // Start a wr
        let mut wr_txn = arc.write();
        // Add a bunch of values, and touch some twice.
        wr_txn.insert(10, 10);
        wr_txn.insert(11, 11);
        wr_txn.insert(12, 12);
        wr_txn.insert(13, 13);
        wr_txn.insert(14, 14);
        wr_txn.insert(15, 15);
        wr_txn.insert(16, 16);
        wr_txn.insert(17, 17);
        wr_txn.commit();
        // Begin a new write.
        let wr_txn = arc.write();
        let rec_set: Vec<usize> = wr_txn.iter_rec().take(2).copied().collect();
        println!("{:?}", rec_set);
        let r = wr_txn.get(&rec_set[0]);
        println!("{:?}", r);
        assert!(r == Some(&rec_set[0]));
        assert!(wr_txn.get(&rec_set[1]) == Some(&rec_set[1]));

        // commit wr
        wr_txn.commit();
        // Begin a new write.
        let mut wr_txn = arc.write();
        println!("stat -> {:?}", wr_txn.peek_stat());
        assert!(
            CStat {
                max: 4,
                cache: 8,
                tlocal: 0,
                freq: 2,
                rec: 2,
                ghost_freq: 0,
                ghost_rec: 4,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );

        // Clear
        wr_txn.clear();
        // Now abort the clear - should do nothing!
        drop(wr_txn);
        // Check the states, should not have changed
        let wr_txn = arc.write();
        println!("stat -> {:?}", wr_txn.peek_stat());
        assert!(
            CStat {
                max: 4,
                cache: 8,
                tlocal: 0,
                freq: 2,
                rec: 2,
                ghost_freq: 0,
                ghost_rec: 4,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );
    }

    #[test]
    fn test_cache_clear_cursed() {
        let arc: Arc<usize, usize> = ARCacheBuilder::default()
            .set_size(4, 4)
            .build()
            .expect("Invaled cache parameters!");
        // Setup for the test
        // --
        let mut wr_txn = arc.write();
        wr_txn.insert(10, 1);
        wr_txn.commit();
        // --
        let wr_txn = arc.write();
        assert!(wr_txn.peek_cache(&10) == CacheState::Rec);
        wr_txn.commit();
        // --
        // Okay, now the test starts. First, we begin a read
        let mut rd_txn = arc.read();
        // Then while that read exists, we open a write, and conduct
        // a cache clear.
        let mut wr_txn = arc.write();
        wr_txn.clear();
        // Commit the clear write.
        wr_txn.commit();

        // Now on the read, we perform a touch of an item, and we include
        // something that was not yet in the cache.
        assert!(rd_txn.get(&10) == Some(&1));
        rd_txn.insert(11, 1);
        // Complete the read
        std::mem::drop(rd_txn);
        // Perform a cache quiesce
        arc.try_quiesce();
        // --

        // Assert that the items that we provided were NOT included, and are
        // in the correct states.
        let wr_txn = arc.write();
        assert!(wr_txn.peek_cache(&10) == CacheState::GhostRec);
        println!("--> {:?}", wr_txn.peek_cache(&11));
        assert!(wr_txn.peek_cache(&11) == CacheState::None);
    }

    #[test]
    fn test_cache_dirty_write() {
        let arc: Arc<usize, usize> = ARCacheBuilder::default()
            .set_size(4, 4)
            .build()
            .expect("Invaled cache parameters!");
        let mut wr_txn = arc.write();
        wr_txn.insert_dirty(10, 1);
        wr_txn.iter_mut_mark_clean().for_each(|(_k, _v)| {});
        wr_txn.commit();
    }

    #[test]
    fn test_cache_read_no_tlocal() {
        // Check a cache with no read local thread capacity
        // Setup the cache.
        let arc: Arc<usize, usize> = ARCacheBuilder::default()
            .set_size(4, 0)
            .build()
            .expect("Invaled cache parameters!");
        // start a rd
        {
            let mut rd_txn = arc.read();
            // add items to the rd
            rd_txn.insert(1, 1);
            rd_txn.insert(2, 2);
            rd_txn.insert(3, 3);
            rd_txn.insert(4, 4);
            // end the rd
            // Everything should be missing frm the tlocal.
            assert!(rd_txn.get(&1).is_none());
            assert!(rd_txn.get(&2).is_none());
            assert!(rd_txn.get(&3).is_none());
            assert!(rd_txn.get(&4).is_none());
        }
        arc.try_quiesce();
        // What state is the cache now in?
        println!("== 2");
        let wr_txn = arc.write();
        assert!(
            CStat {
                max: 4,
                cache: 4,
                tlocal: 0,
                freq: 0,
                rec: 4,
                ghost_freq: 0,
                ghost_rec: 0,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );
        assert!(wr_txn.peek_cache(&1) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&2) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&3) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&4) == CacheState::Rec);
        let stats = arc.view_stats();
        println!("stats 1: {:?}", *stats);
        assert!(stats.reader_includes == 4);
        assert!(stats.reader_tlocal_includes == 0);
        assert!(stats.reader_tlocal_hits == 0);
    }

    #[derive(Clone, Debug)]
    struct Weighted {
        _i: u64,
    }

    #[test]
    fn test_cache_weighted() {
        let arc: Arc<usize, Weighted> = ARCacheBuilder::default()
            .set_size(4, 0)
            .build()
            .expect("Invaled cache parameters!");
        let mut wr_txn = arc.write();

        assert!(
            CStat {
                max: 4,
                cache: 0,
                tlocal: 0,
                freq: 0,
                rec: 0,
                ghost_freq: 0,
                ghost_rec: 0,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );

        // In the first txn we insert 2 weight 2 items.
        wr_txn.insert_sized(1, Weighted { _i: 1 }, 2);
        wr_txn.insert_sized(2, Weighted { _i: 2 }, 2);

        assert!(
            CStat {
                max: 4,
                cache: 0,
                tlocal: 2,
                freq: 0,
                rec: 0,
                ghost_freq: 0,
                ghost_rec: 0,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );
        wr_txn.commit();

        // Now once commited, the proper sizes kick in.

        let wr_txn = arc.write();
        // eprintln!("{:?}", wr_txn.peek_stat());
        assert!(
            CStat {
                max: 4,
                cache: 2,
                tlocal: 0,
                freq: 0,
                rec: 4,
                ghost_freq: 0,
                ghost_rec: 0,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );
        wr_txn.commit();

        // Check the numbers move properly.
        let wr_txn = arc.write();
        wr_txn.get(&1);
        wr_txn.commit();

        let mut wr_txn = arc.write();
        assert!(
            CStat {
                max: 4,
                cache: 2,
                tlocal: 0,
                freq: 2,
                rec: 2,
                ghost_freq: 0,
                ghost_rec: 0,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );

        wr_txn.insert_sized(3, Weighted { _i: 3 }, 2);
        wr_txn.insert_sized(4, Weighted { _i: 4 }, 2);
        wr_txn.commit();

        // Check the evicts
        let wr_txn = arc.write();
        assert!(
            CStat {
                max: 4,
                cache: 4,
                tlocal: 0,
                freq: 2,
                rec: 2,
                ghost_freq: 0,
                ghost_rec: 4,
                haunted: 0,
                p: 0
            } == wr_txn.peek_stat()
        );
        wr_txn.commit();

        let stats = arc.view_stats();
        println!("{:?}", *stats);
    }

    #[test]
    fn test_cache_stats_reload() {
        // Make a cache
        let arc: Arc<usize, usize> = ARCacheBuilder::default()
            .set_size(4, 0)
            .build()
            .expect("Invaled cache parameters!");
        let mut wr_txn = arc.write();
        wr_txn.insert(1, 1);
        wr_txn.commit();

        let stats = arc.view_stats();
        println!("stats 1: {:?}", *stats);

        let arc2: Arc<usize, usize> = ARCacheBuilder::default()
            .set_size(4, 0)
            .set_stats((*stats).clone())
            .build()
            .expect("Failed to build");

        let stats2 = arc2.view_stats();
        println!("stats 2: {:?}", *stats2);
        assert!(stats.write_includes == stats2.write_includes);
        assert!(stats.write_modifies == stats2.write_modifies);
    }
}
