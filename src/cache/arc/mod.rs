//! Arc - A concurrently readable adaptive replacement cache.
//!
//! An Arc is used in place of a `RwLock<LruCache>` or `Mutex<LruCache>`.
//! This structure is transactional, meaning that readers have guaranteed
//! point-in-time views of the cache and their items, while allowing writers
//! to proceed with inclusions and cache state management in parallel.
//!
//! This means that unlike a `RwLock` which can have many readers OR one writer
//! this cache is capable of many readers, over multiple data generations AND
//! writers that are serialised. This formally means that this is an ACID
//! compliant Cache.

mod ll;

use self::ll::{LLNode, LL};
use crate::collections::bptree::*;
use crate::cowcell::{CowCell, CowCellReadTxn};
use crossbeam::channel::{unbounded, Receiver, Sender};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap as Map;

use std::borrow::Borrow;
use std::cell::UnsafeCell;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use std::time::Instant;

const READ_THREAD_MIN: usize = 8;
const READ_THREAD_RATIO: usize = 16;

/// Statistics related to the Arc
#[derive(Clone, Debug, PartialEq)]
pub struct CacheStats {
    pub reader_hits: usize,
    pub reader_includes: usize,
    pub write_hits: usize,
    pub write_inc_or_mod: usize,
    pub shared_max: usize,
    pub freq: usize,
    pub recent: usize,
    pub freq_evicts: usize,
    pub recent_evicts: usize,
    pub p_weight: usize,
    pub all_seen_keys: usize,
}

enum ThreadCacheItem<V> {
    Present(V, bool),
    Removed(bool),
}

enum CacheEvent<K, V> {
    Hit(Instant, K),
    Include(Instant, K, V, usize),
}

#[derive(Hash, Ord, PartialOrd, Eq, PartialEq, Clone, Debug)]
struct CacheItemInner<K>
where
    K: Hash + Eq + Ord + Clone + Debug,
{
    k: K,
    txid: usize,
}

#[derive(Clone, Debug)]
enum CacheItem<K, V>
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
    K: Hash + Eq + Ord + Clone + Debug,
    V: Clone + Debug,
{
    // Weight of items between the two caches.
    p: usize,
    freq: LL<CacheItemInner<K>>,
    rec: LL<CacheItemInner<K>>,
    ghost_freq: LL<CacheItemInner<K>>,
    ghost_rec: LL<CacheItemInner<K>>,
    haunted: LL<CacheItemInner<K>>,
    rx: Receiver<CacheEvent<K, V>>,
    min_txid: usize,
}

struct ArcShared<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug,
    V: Clone + Debug,
{
    // Max number of elements to cache.
    max: usize,
    // Max number of elements for a reader per thread.
    read_max: usize,
    // channels for readers.
    // tx (cloneable)
    tx: Sender<CacheEvent<K, V>>,
}

/// A concurrently readable adaptive replacement cache. Operations are performed on the
/// cache via read and write operations.
pub struct Arc<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug,
    V: Clone + Debug,
{
    // Use a unified tree, allows simpler movement of items between the
    // cache types.
    cache: BptreeMap<K, CacheItem<K, V>>,
    // This is normally only ever taken in "read" mode, so it's effectively
    // an uncontended barrier.
    shared: RwLock<ArcShared<K, V>>,
    // These are only taken during a quiesce
    inner: Mutex<ArcInner<K, V>>,
    stats: CowCell<CacheStats>,
}

unsafe impl<K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> Send for Arc<K, V> {}
unsafe impl<K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> Sync for Arc<K, V> {}

struct ReadCache<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug,
    V: Clone + Debug,
{
    // cache of our missed items to send forward.
    // On drop we drain this to the channel
    set: Map<K, *mut LLNode<(K, V)>>,
    read_size: usize,
    tlru: LL<(K, V)>,
}

/// An active read transaction over the cache. The data is this cache is guaranteed to be
/// valid at the point in time the read is created. You may include items during a cache
/// miss via the "insert" function.
pub struct ArcReadTxn<'a, K, V>
where
    K: Hash + Eq + Ord + Clone + Debug,
    V: Clone + Debug,
{
    caller: &'a Arc<K, V>,
    // ro_txn to cache
    cache: BptreeMapReadTxn<K, CacheItem<K, V>>,
    tlocal: Option<ReadCache<K, V>>,
    // tx channel to send forward events.
    tx: Sender<CacheEvent<K, V>>,
    ts: Instant,
}

/// An active write transaction over the cache. The data in this cache is isolated
/// from readers, and may be rolled-back if an error occurs. Changes only become
/// globally visible once you call "commit". Items may be added to the cache on
/// a miss via "insert", and you can explicitly remove items by calling "remove".
pub struct ArcWriteTxn<'a, K, V>
where
    K: Hash + Eq + Ord + Clone + Debug,
    V: Clone + Debug,
{
    caller: &'a Arc<K, V>,
    // wr_txn to cache
    cache: BptreeMapWriteTxn<'a, K, CacheItem<K, V>>,
    // Cache of missed items (w_ dirty/clean)
    // On COMMIT we drain this to the main cache
    tlocal: Map<K, ThreadCacheItem<V>>,
    hit: UnsafeCell<Vec<K>>,
    clear: UnsafeCell<bool>,
}

/*
pub struct ArcReadSnapshot<K, V> {
    // How to communicate back to the caller the loads we did?
    tlocal: &mut Map<K, ThreadCacheItem<K, V>>,
}
*/

impl<K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> CacheItem<K, V> {
    fn to_vref(&self) -> Option<&V> {
        match &self {
            CacheItem::Freq(_, v) | CacheItem::Rec(_, v) => Some(&v),
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

impl<K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> Arc<K, V> {
    /// Create a new Arc, that derives it's size based on your expected workload.
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
    pub fn new(
        total: usize,
        threads: usize,
        ex_ro_miss: usize,
        ex_rw_miss: usize,
        read_cache: bool,
    ) -> Self {
        let y = isize::try_from(total).unwrap();
        let t = isize::try_from(threads).unwrap();
        let m = isize::try_from(ex_ro_miss).unwrap();
        let w = isize::try_from(ex_rw_miss).unwrap();
        let r = isize::try_from(READ_THREAD_RATIO).unwrap();
        // I'd like to thank wolfram alpha ... for this magic.
        let max = -((r * ((m * t) + w - y)) / (r + t));
        let read_max = if read_cache { max / r } else { 0 };

        let max = usize::try_from(max).unwrap();
        let read_max = usize::try_from(read_max).unwrap();

        Self::new_size(max, read_max)
    }

    /// Create a new Arc, with a capacity of `max` main cache items and `read_max`
    /// Note that due to the way the cache operates, the number of items can and
    /// will exceed `max` on a regular basis, so you should consider using `new`
    /// and specifying your expected workload parameters to have a better derived
    /// cache size.
    pub fn new_size(max: usize, read_max: usize) -> Self {
        assert!(max > 0);
        let (tx, rx) = unbounded();
        let shared = RwLock::new(ArcShared { max, read_max, tx });
        let inner = Mutex::new(ArcInner {
            p: 0,
            freq: LL::new(),
            rec: LL::new(),
            ghost_freq: LL::new(),
            ghost_rec: LL::new(),
            haunted: LL::new(),
            rx,
            min_txid: 0,
        });
        let stats = CowCell::new(CacheStats {
            reader_hits: 0,
            reader_includes: 0,
            write_hits: 0,
            write_inc_or_mod: 0,
            shared_max: 0,
            freq: 0,
            recent: 0,
            freq_evicts: 0,
            recent_evicts: 0,
            p_weight: 0,
            all_seen_keys: 0,
        });
        Arc {
            cache: BptreeMap::new(),
            shared,
            inner,
            stats,
        }
    }

    /// Begin a read operation on the cache. This reader has a thread-local cache for items
    /// that are localled included via `insert`, and can communicate back to the main cache
    /// to safely include items.
    pub fn read(&self) -> ArcReadTxn<K, V> {
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
        ArcReadTxn {
            caller: &self,
            cache: self.cache.read(),
            tlocal,
            tx: rshared.tx.clone(),
            ts: Instant::now(),
        }
    }

    /// Begin a write operation on the cache. This writer has a thread-local store
    /// for all items that have been included or dirtied in the transactions, items
    /// may be removed from this cache (ie deleted, invalidated).
    pub fn write(&self) -> ArcWriteTxn<K, V> {
        ArcWriteTxn {
            caller: &self,
            cache: self.cache.write(),
            tlocal: Map::new(),
            hit: UnsafeCell::new(Vec::new()),
            clear: UnsafeCell::new(false),
        }
    }

    pub fn view_stats(&self) -> CowCellReadTxn<CacheStats> {
        self.stats.read()
    }

    fn try_write(&self) -> Option<ArcWriteTxn<K, V>> {
        self.cache.try_write().map(|cache| ArcWriteTxn {
            caller: &self,
            cache: cache,
            tlocal: Map::new(),
            hit: UnsafeCell::new(Vec::new()),
            clear: UnsafeCell::new(false),
        })
    }

    fn try_quiesce(&self) {
        match self.try_write() {
            Some(wr_txn) => wr_txn.commit(),
            None => {}
        }
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

    fn commit<'a>(
        &'a self,
        mut cache: BptreeMapWriteTxn<'a, K, CacheItem<K, V>>,
        tlocal: Map<K, ThreadCacheItem<V>>,
        hit: Vec<K>,
        clear: bool,
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

        stats.write_inc_or_mod += tlocal.len();

        // drain tlocal into the main cache.
        tlocal.into_iter().for_each(|(k, tcio)| {
            let r = cache.get_mut(&k);
            match (r, tcio) {
                (None, ThreadCacheItem::Present(tci, clean)) => {
                    assert!(clean);
                    let llp = inner.rec.append_k(CacheItemInner {
                        k: k.clone(),
                        txid: commit_txid,
                    });
                    cache.insert(k, CacheItem::Rec(llp, tci));
                }
                (None, ThreadCacheItem::Removed(clean)) => {
                    assert!(clean);
                    // Mark this as haunted
                    let llp = inner.haunted.append_k(CacheItemInner {
                        k: k.clone(),
                        txid: commit_txid,
                    });
                    cache.insert(k, CacheItem::Haunted(llp));
                }
                (Some(ref mut ci), ThreadCacheItem::Removed(clean)) => {
                    assert!(clean);
                    // From whatever set we were in, pop and move to haunted.
                    let mut next_state = match ci {
                        CacheItem::Freq(llp, _v) => {
                            // println!("tlocal {:?} Freq -> Freq", k);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            inner.freq.extract(*llp);
                            inner.haunted.append_n(*llp);
                            CacheItem::Haunted(*llp)
                        }
                        CacheItem::Rec(llp, _v) => {
                            // println!("tlocal {:?} Rec -> Freq", k);
                            // Remove the node and put it into freq.
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            inner.rec.extract(*llp);
                            inner.haunted.append_n(*llp);
                            CacheItem::Haunted(*llp)
                        }
                        CacheItem::GhostFreq(llp) => {
                            // println!("tlocal {:?} GhostFreq -> Freq", k);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            inner.ghost_freq.extract(*llp);
                            inner.haunted.append_n(*llp);
                            CacheItem::Haunted(*llp)
                        }
                        CacheItem::GhostRec(llp) => {
                            // println!("tlocal {:?} GhostRec -> Rec", k);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            inner.ghost_rec.extract(*llp);
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
                (Some(ref mut ci), ThreadCacheItem::Present(ref tci, clean)) => {
                    assert!(clean);
                    //   * as we include each item, what state was it in before?
                    // It's in the cache - what action must we take?
                    let mut next_state = match ci {
                        CacheItem::Freq(llp, _v) => {
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            // println!("tlocal {:?} Freq -> Freq", k);
                            // Move the list item to it's head.
                            inner.freq.touch(*llp);
                            // Update v.
                            CacheItem::Freq(*llp, (*tci).clone())
                        }
                        CacheItem::Rec(llp, _v) => {
                            // println!("tlocal {:?} Rec -> Freq", k);
                            // Remove the node and put it into freq.
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            inner.rec.extract(*llp);
                            inner.freq.append_n(*llp);
                            CacheItem::Freq(*llp, (*tci).clone())
                        }
                        CacheItem::GhostFreq(llp) => {
                            // println!("tlocal {:?} GhostFreq -> Freq", k);
                            // Ajdust p
                            Self::calc_p_freq(
                                inner.ghost_rec.len(),
                                inner.ghost_freq.len(),
                                &mut inner.p,
                            );
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            inner.ghost_freq.extract(*llp);
                            inner.freq.append_n(*llp);
                            CacheItem::Freq(*llp, (*tci).clone())
                        }
                        CacheItem::GhostRec(llp) => {
                            // println!("tlocal {:?} GhostRec -> Rec", k);
                            // Ajdust p
                            Self::calc_p_rec(
                                shared.max,
                                inner.ghost_rec.len(),
                                inner.ghost_freq.len(),
                                &mut inner.p,
                            );
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            inner.ghost_rec.extract(*llp);
                            inner.rec.append_n(*llp);
                            CacheItem::Rec(*llp, (*tci).clone())
                        }
                        CacheItem::Haunted(llp) => {
                            // println!("tlocal {:?} Haunted -> Rec", k);
                            unsafe { (**llp).as_mut().txid = commit_txid };
                            inner.haunted.extract(*llp);
                            inner.rec.append_n(*llp);
                            CacheItem::Rec(*llp, (*tci).clone())
                        }
                    };
                    // Now change the state.
                    mem::swap(*ci, &mut next_state);
                }
            }
        });

        // drain rx until empty or time >= time.
        // * for each item
        while let Ok(ce) = inner.rx.try_recv() {
            let t = match ce {
                // Update if it was hit.
                CacheEvent::Hit(t, k) => {
                    stats.reader_hits += 1;
                    let mut r = cache.get_mut(&k);
                    match r {
                        Some(ref mut ci) => {
                            let mut next_state = match &ci {
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
                            mem::swap(*ci, &mut next_state);
                        }
                        None => {
                            // Do nothing, it must have been evicted.
                        }
                    };
                    t
                }
                // Update if it was inc
                CacheEvent::Include(t, k, iv, txid) => {
                    stats.reader_includes += 1;
                    let mut r = cache.get_mut(&k);
                    match r {
                        Some(ref mut ci) => {
                            let mut next_state = match &ci {
                                CacheItem::Freq(llp, _v) => {
                                    inner.freq.touch(*llp);
                                    if unsafe { (**llp).as_ref().txid >= txid }
                                        || inner.min_txid > txid
                                    {
                                        // println!("rxinc {:?} Freq -> Freq (touch only)", k);
                                        // Our cache already has a newer value, keep it.
                                        None
                                    } else {
                                        // println!("rxinc {:?} Freq -> Freq (update)", k);
                                        // The value is newer, update.
                                        unsafe { (**llp).as_mut().txid = txid };
                                        Some(CacheItem::Freq(*llp, iv))
                                    }
                                }
                                CacheItem::Rec(llp, v) => {
                                    inner.rec.extract(*llp);
                                    inner.freq.append_n(*llp);
                                    if unsafe { (**llp).as_ref().txid >= txid }
                                        || inner.min_txid > txid
                                    {
                                        // println!("rxinc {:?} Rec -> Freq (touch only)", k);
                                        Some(CacheItem::Freq(*llp, v.clone()))
                                    } else {
                                        // println!("rxinc {:?} Rec -> Freq (update)", k);
                                        unsafe { (**llp).as_mut().txid = txid };
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
                                    inner.ghost_freq.extract(*llp);
                                    if unsafe { (**llp).as_ref().txid > txid }
                                        || inner.min_txid > txid
                                    {
                                        // println!("rxinc {:?} GhostFreq -> GhostFreq", k);
                                        // The cache version is newer, this is just a hit.
                                        inner.ghost_freq.append_n(*llp);
                                        None
                                    } else {
                                        // This item is newer, so we can include it.
                                        // println!("rxinc {:?} GhostFreq -> Rec", k);
                                        inner.freq.append_n(*llp);
                                        unsafe { (**llp).as_mut().txid = txid };
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
                                    if unsafe { (**llp).as_ref().txid > txid }
                                        || inner.min_txid > txid
                                    {
                                        // println!("rxinc {:?} GhostRec -> GhostRec", k);
                                        inner.ghost_rec.touch(*llp);
                                        None
                                    } else {
                                        // println!("rxinc {:?} GhostRec -> Rec", k);
                                        inner.ghost_rec.extract(*llp);
                                        inner.rec.append_n(*llp);
                                        unsafe { (**llp).as_mut().txid = txid };
                                        Some(CacheItem::Rec(*llp, iv))
                                    }
                                }
                                CacheItem::Haunted(llp) => {
                                    if unsafe { (**llp).as_ref().txid > txid }
                                        || inner.min_txid > txid
                                    {
                                        // println!("rxinc {:?} Haunted -> Haunted", k);
                                        None
                                    } else {
                                        // println!("rxinc {:?} Haunted -> Rec", k);
                                        inner.haunted.extract(*llp);
                                        inner.rec.append_n(*llp);
                                        unsafe { (**llp).as_mut().txid = txid };
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
                                let llp = inner.rec.append_k(CacheItemInner { k: k.clone(), txid });
                                cache.insert(k, CacheItem::Rec(llp, iv));
                            }
                        }
                    };
                    t
                }
            };
            // Stop processing the queue, we are up to "now".
            if t >= commit_ts {
                break;
            }
        }

        stats.write_hits += hit.len();
        // drain the tlocal hits into the main cache.
        hit.into_iter().for_each(|k| {
            // * everything hit must be in main cache now, so bring these
            //   all to the relevant item heads.
            // * Why do this last? Because the write is the "latest" we want all the fresh
            //   written items in the cache over the "read" hits, it gives us some aprox
            //   of time ordering, but not perfect.

            // Find the item in the cache.
            // * based on it's type, promote it in the correct list, or move it.
            // How does this prevent incorrect promotion from rec to freq? txid?
            // println!("Checking Hit ... {:?}", k);
            let mut r = cache.get_mut(&k);
            match r {
                Some(ref mut ci) => {
                    // This differs from above - we skip if we don't touch anything
                    // that was added in this txn. This is to prevent double touching
                    // anything that was included in a write.
                    let mut next_state = match &ci {
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
                        mem::swap(*ci, next_state);
                    }
                }
                None => {
                    // Impossible state!
                    unreachable!();
                }
            }
        });

        // now clean the space for each of the primary caches, evicting into the ghost sets.
        // * It's possible that both caches are now over-sized if rx was empty
        //   but wr inc many items.
        // * p has possibly changed width, causing a balance shift
        // * and ghost items have been included changing ghost list sizes.
        // so we need to do a clean up/balance of all the list lengths.
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
                &mut cache,
                &mut inner.rec,
                &mut inner.ghost_rec,
                rec_to_len,
                commit_txid
            );
            evict_to_len!(
                &mut cache,
                &mut inner.freq,
                &mut inner.ghost_freq,
                freq_to_len,
                commit_txid
            );

            // Finally, do an evict of the ghost sets if they are too long - these are weighted
            // inverse to the above sets. Note the freq to len in ghost rec, and rec to len in
            // ghost freq!
            if inner.ghost_rec.len() > (shared.max - p) {
                evict_to_haunted_len!(
                    &mut cache,
                    &mut inner.ghost_rec,
                    &mut inner.haunted,
                    freq_to_len,
                    commit_txid
                );
            }

            if inner.ghost_freq.len() > p {
                evict_to_haunted_len!(
                    &mut cache,
                    &mut inner.ghost_freq,
                    &mut inner.haunted,
                    rec_to_len,
                    commit_txid
                );
            }
        }

        stats.shared_max = shared.max;
        stats.freq = inner.freq.len();
        stats.recent = inner.rec.len();
        stats.all_seen_keys = cache.len();

        // Commit the stats
        stat_guard.commit();
        // commit on the wr txn.
        cache.commit();
        // done!
    }
}

impl<'a, K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> ArcWriteTxn<'a, K, V> {
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
        K: Borrow<Q> + From<Q>,
        Q: Hash + Eq + Ord + Clone,
    {
        let r: Option<&V> = if let Some(tci) = self.tlocal.get(k) {
            match tci {
                ThreadCacheItem::Present(v, _clean) => {
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
                if let Some(v) = self.cache.get(k) {
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
        if r.is_some() {
            let hk: K = k.to_owned().into();
            unsafe {
                let hit_ptr = self.hit.get();
                (*hit_ptr).push(hk);
            }
        }
        r
    }

    /// Determine if this cache contains the following key.
    pub fn contains_key<'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q> + From<Q>,
        Q: Hash + Eq + Ord + Clone,
    {
        self.get(k).is_some()
    }

    /// Add a value to the cache. This may be because you have had a cache miss and
    /// now wish to include in the thread local storage, or because you have written
    /// a new value and want it to be submitted for caching. This item is marked as
    /// clean, IE you have synced it to whatever associated store exists.
    pub fn insert(&mut self, k: K, v: V) {
        self.tlocal.insert(k, ThreadCacheItem::Present(v, true));
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
        self.tlocal.insert(k, ThreadCacheItem::Present(v, false));
    }

    /// Remove this value from the thread local cache IE mask from from being
    /// returned until this thread performs an insert. This item is marked as
    /// dirty, because you have *not* synced it. You MUST call iter_mut_mark_clean before calling
    /// `commit` on this transaction, or a panic will occur.
    pub fn remove_dirty(&mut self, k: K) {
        self.tlocal.insert(k, ThreadCacheItem::Removed(false));
    }

    /// Yields an iterator over all values that are currently dirty. As the iterator
    /// progresses, items will be marked clean. This is where you should sync dirty
    /// cache content to your associated store. The iterator is K, Option<V>, where
    /// the Option<V> indicates if the item has been remove (None) or is updated (Some).
    pub fn iter_mut_mark_clean<'b>(
        &'b mut self,
    ) -> impl Iterator<Item = (&'b K, Option<&'b mut V>)> {
        self.tlocal
            .iter_mut()
            .filter(|(k, v)| match v {
                ThreadCacheItem::Present(_v, c) => !c,
                ThreadCacheItem::Removed(c) => !c,
            })
            .map(|(k, v)| {
                // Mark it clean.
                match v {
                    ThreadCacheItem::Present(_v, c) => *c = true,
                    ThreadCacheItem::Removed(c) => *c = true,
                }
                // Get the data.
                let data = match v {
                    ThreadCacheItem::Present(v, c) => Some(v),
                    ThreadCacheItem::Removed(c) => None,
                };
                (k, data)
            })
    }

    #[cfg(test)]
    pub(crate) fn peek_hit(&self) -> &[K] {
        let hit_ptr = self.hit.get();
        unsafe { &(*hit_ptr) }
    }

    #[cfg(test)]
    pub(crate) fn peek_cache<'b, Q: ?Sized>(&'a self, k: &'b Q) -> CacheState
    where
        K: Borrow<Q> + From<Q>,
        Q: Hash + Eq + Ord + Clone,
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

impl<'a, K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> ArcReadTxn<'a, K, V> {
    /// Attempt to retieve a k-v pair from the cache. If it is present in the main cache OR
    /// the thread local cache, a `Some` is returned, else you will recieve a `None`. On a
    /// `None`, you must then consult the external data source that this structure is acting
    /// as a cache for.
    pub fn get<'b, Q: ?Sized>(&'b self, k: &'b Q) -> Option<&'b V>
    where
        K: Borrow<Q> + From<Q>,
        Q: Hash + Eq + Ord + Clone,
    {
        let r: Option<&V> = self
            .tlocal
            .as_ref()
            .and_then(|cache| {
                cache.set.get(k).and_then(|v| unsafe {
                    let v = &(**v).as_ref().1 as *const _;
                    // This discards the lifetime and repins it to &'b.
                    Some(&(*v))
                })
            })
            .or_else(|| {
                self.cache.get(k).and_then(|v| unsafe {
                    (*v).to_vref().map(|vin| unsafe {
                        let vin = vin as *const _;
                        &(*vin)
                    })
                })
            });

        if r.is_some() {
            let hk: K = k.to_owned().into();
            self.tx
                .send(CacheEvent::Hit(self.ts.clone(), hk))
                .expect("Invalid tx state");
        }
        r
    }

    /// Determine if this cache contains the following key.
    pub fn contains_key<'b, Q: ?Sized>(&mut self, k: &'b Q) -> bool
    where
        K: Borrow<Q> + From<Q>,
        Q: Hash + Eq + Ord + Clone,
    {
        self.get(k).is_some()
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
    pub fn insert(&mut self, k: K, mut v: V) {
        // Send a copy forward through time and space.
        self.tx
            .send(CacheEvent::Include(
                self.ts.clone(),
                k.clone(),
                v.clone(),
                self.cache.get_txid(),
            ))
            .expect("Invalid tx state!");

        // We have a cache, so lets update it.
        if let Some(ref mut cache) = self.tlocal {
            let n = if cache.tlru.len() >= cache.read_size {
                let n = cache.tlru.pop();
                // swap the old_key/old_val out
                let mut k_clone = k.clone();
                unsafe {
                    mem::swap(&mut k_clone, &mut (*n).as_mut().0);
                    mem::swap(&mut v, &mut (*n).as_mut().1);
                }
                // remove old K from the tree:
                cache.set.remove(&k_clone);
                n
            } else {
                // Just add it!
                cache.tlru.append_k((k.clone(), v))
            };
            let r = cache.set.insert(k, n);
            // There should never be a previous value.
            assert!(r.is_none());
        }
    }
}

impl<'a, K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> Drop for ArcReadTxn<'a, K, V> {
    fn drop(&mut self) {
        self.caller.try_quiesce();
    }
}

#[cfg(test)]
mod tests {
    use crate::cache::arc::Arc;
    use crate::cache::arc::CStat;
    use crate::cache::arc::CacheState;

    #[test]
    fn test_cache_arc_basic() {
        let arc: Arc<usize, usize> = Arc::new_size(4, 4);
        let mut wr_txn = arc.write();

        assert!(wr_txn.get(&1) == None);
        assert!(wr_txn.peek_hit().len() == 0);
        wr_txn.insert(1, 1);
        assert!(wr_txn.get(&1) == Some(&1));
        assert!(wr_txn.peek_hit().len() == 1);

        wr_txn.commit();

        // Now we start the second txn, and see if it's in there.
        let mut wr_txn = arc.write();
        assert!(wr_txn.peek_cache(&1) == CacheState::Rec);
        assert!(wr_txn.get(&1) == Some(&1));
        assert!(wr_txn.peek_hit().len() == 1);
        wr_txn.commit();
        // And now check it's moved to Freq due to the extra
        let mut wr_txn = arc.write();
        assert!(wr_txn.peek_cache(&1) == CacheState::Freq);
        println!("{:?}", wr_txn.peek_stat());
    }

    #[test]
    fn test_cache_evict() {
        println!("== 1");
        let arc: Arc<usize, usize> = Arc::new_size(4, 4);
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
        let mut wr_txn = arc.write();
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
        let mut wr_txn = arc.write();
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
        assert!(wr_txn.peek_cache(&1) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&2) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&3) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&4) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&5) == CacheState::Rec);

        // Now we touch 4/5 to make them freq.
        assert!(wr_txn.get(&4) == Some(&4));
        assert!(wr_txn.get(&5) == Some(&5));
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
        assert!(wr_txn.peek_cache(&1) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&2) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&3) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&4) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&5) == CacheState::Freq);

        // Now, we touch 3, 10, 11, 12, 13. Due to how many additions there are, 3 should be
        // evicted from ghost rec, and all the current freq items should be removed?
        wr_txn.insert(3, 3);
        assert!(wr_txn.get(&3) == Some(&3));
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
        assert!(wr_txn.peek_cache(&1) == CacheState::GhostFreq);
        assert!(wr_txn.peek_cache(&2) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&3) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&4) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&5) == CacheState::Freq);

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
        assert!(wr_txn.peek_cache(&1) == CacheState::GhostFreq);
        assert!(wr_txn.peek_cache(&2) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&3) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&4) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&5) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&10) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&11) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&12) == CacheState::Rec);

        // Eight txn - now that we had a demand for items before, we re-demand them - this will trigger
        // a shift in p, causing some more to be in the rec cache.
        wr_txn.insert(10, 10);
        wr_txn.insert(11, 11);
        wr_txn.insert(3, 3);
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
        assert!(wr_txn.peek_cache(&1) == CacheState::GhostFreq);
        assert!(wr_txn.peek_cache(&2) == CacheState::GhostFreq);
        assert!(wr_txn.peek_cache(&3) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&4) == CacheState::GhostFreq);
        assert!(wr_txn.peek_cache(&5) == CacheState::GhostFreq);
        assert!(wr_txn.peek_cache(&10) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&11) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&12) == CacheState::Rec);

        // Now lets go back the other way - we want freq items to come back.
        wr_txn.insert(1, 1);
        wr_txn.insert(2, 1);
        wr_txn.insert(4, 1);
        wr_txn.insert(5, 1);
        wr_txn.commit();

        println!("== 9");
        let mut wr_txn = arc.write();
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
        assert!(wr_txn.peek_cache(&1) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&2) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&3) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&4) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&5) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&10) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&11) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&12) == CacheState::GhostRec);

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
        let arc: Arc<usize, usize> = Arc::new_size(4, 4);
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
        let mut wr_txn = arc.write();
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
        let mut wr_txn = arc.write();
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
        let mut wr_txn = arc.write();
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
        println!("{:?}", *stats);
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
        let arc: Arc<usize, usize> = Arc::new_size(4, 4);

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
        let mut wr_txn = arc.write();
        // assert that 1 is haunted.
        assert!(wr_txn.peek_cache(&1) == CacheState::Haunted);
        // assert 1 is not in rd.
        assert!(rd_txn.get(&1) == None);
        // now that 1 is hanuted, in rd attempt to insert 1, X
        rd_txn.insert(1, 100);
        // commit wr
        wr_txn.commit();

        // start wr
        let mut wr_txn = arc.write();
        // assert that 1 is still haunted.
        assert!(wr_txn.peek_cache(&1) == CacheState::Haunted);
        // assert that 1, x is in rd.
        assert!(rd_txn.get(&1) == Some(&100));
        // done!
    }

    #[test]
    fn test_cache_clear() {
        let arc: Arc<usize, usize> = Arc::new_size(4, 4);

        // Start a wr
        let mut wr_txn = arc.write();
        // Add a bunch of values, and touch some twice.
        wr_txn.insert(10, 1);
        wr_txn.insert(11, 1);
        wr_txn.insert(12, 1);
        wr_txn.insert(13, 1);
        wr_txn.insert(14, 1);
        wr_txn.insert(15, 1);
        wr_txn.insert(16, 1);
        wr_txn.insert(17, 1);
        wr_txn.commit();
        // Begin a new write.
        let mut wr_txn = arc.write();
        assert!(wr_txn.get(&16) == Some(&1));
        assert!(wr_txn.get(&17) == Some(&1));
        // commit wr
        wr_txn.commit();
        // Begin a new write.
        let mut wr_txn = arc.write();
        assert!(wr_txn.peek_cache(&10) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&11) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&12) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&13) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&14) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&15) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&16) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&17) == CacheState::Freq);

        // Clear
        wr_txn.clear();
        // Now commit
        wr_txn.commit();

        // Now check their states.
        let mut wr_txn = arc.write();

        assert!(wr_txn.peek_cache(&10) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&11) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&12) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&13) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&14) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&15) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&16) == CacheState::GhostFreq);
        assert!(wr_txn.peek_cache(&17) == CacheState::GhostFreq);
        // See what stats did
        let stats = arc.view_stats();
        println!("{:?}", *stats);
    }

    #[test]
    fn test_cache_clear_rollback() {
        let arc: Arc<usize, usize> = Arc::new_size(4, 4);

        // Start a wr
        let mut wr_txn = arc.write();
        // Add a bunch of values, and touch some twice.
        wr_txn.insert(10, 1);
        wr_txn.insert(11, 1);
        wr_txn.insert(12, 1);
        wr_txn.insert(13, 1);
        wr_txn.insert(14, 1);
        wr_txn.insert(15, 1);
        wr_txn.insert(16, 1);
        wr_txn.insert(17, 1);
        wr_txn.commit();
        // Begin a new write.
        let mut wr_txn = arc.write();
        assert!(wr_txn.get(&16) == Some(&1));
        assert!(wr_txn.get(&17) == Some(&1));
        // commit wr
        wr_txn.commit();
        // Begin a new write.
        let mut wr_txn = arc.write();
        assert!(wr_txn.peek_cache(&10) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&11) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&12) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&13) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&14) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&15) == CacheState::Rec);
        println!("--> {:?}", wr_txn.peek_cache(&16));
        assert!(wr_txn.peek_cache(&16) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&17) == CacheState::Freq);

        // Clear
        wr_txn.clear();
        // Now abort the clear - should do nothing!
        drop(wr_txn);
        // Check the states, should not have changed
        let mut wr_txn = arc.write();
        assert!(wr_txn.peek_cache(&10) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&11) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&12) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&13) == CacheState::GhostRec);
        assert!(wr_txn.peek_cache(&14) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&15) == CacheState::Rec);
        assert!(wr_txn.peek_cache(&16) == CacheState::Freq);
        assert!(wr_txn.peek_cache(&17) == CacheState::Freq);
    }

    #[test]
    fn test_cache_clear_cursed() {
        let arc: Arc<usize, usize> = Arc::new_size(4, 4);
        // Setup for the test
        // --
        let mut wr_txn = arc.write();
        wr_txn.insert(10, 1);
        wr_txn.commit();
        // --
        let mut wr_txn = arc.write();
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
        let mut wr_txn = arc.write();
        assert!(wr_txn.peek_cache(&10) == CacheState::GhostRec);
        println!("--> {:?}", wr_txn.peek_cache(&11));
        assert!(wr_txn.peek_cache(&11) == CacheState::None);
    }

    #[test]
    fn test_cache_dirty_write() {
        let arc: Arc<usize, usize> = Arc::new_size(4, 4);
        let mut wr_txn = arc.write();
        wr_txn.insert_dirty(10, 1);
        wr_txn.iter_mut_mark_clean().for_each(|(_k, _v)| {});
        wr_txn.commit();
    }

    #[test]
    fn test_cache_read_no_tlocal() {
        // Check a cache with no read local thread capacity
        // Setup the cache.
        let arc: Arc<usize, usize> = Arc::new_size(4, 0);
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
        let mut wr_txn = arc.write();
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
    }
}
