use crate::hashmap::asynch::*;

use super::ll::LL;
use super::common::*;
use std::borrow::Borrow;
use std::mem;
use std::convert::TryFrom;
use std::cell::UnsafeCell;
use std::collections::HashMap as Map;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{DerefMut, Deref};
use crossbeam::channel::{unbounded, Sender};
use crate::cowcell::{CowCell, CowCellReadTxn};
use std::time::Instant;
use parking_lot::{Mutex, RwLock};

/// A concurrently readable adaptive replacement cache. Operations are performed on the
/// cache via read and write operations.
pub struct ARCache<K, V>
where
    K: Hash + Eq + Ord + Clone + Debug,
    V: Clone + Debug,
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
}

unsafe impl<K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> Send for ARCache<K, V> {}
unsafe impl<K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> Sync for ARCache<K, V> {}

/// An active read transaction over the cache. The data is this cache is guaranteed to be
/// valid at the point in time the read is created. You may include items during a cache
/// miss via the "insert" function.
pub struct ARCacheReadTxn<'a, K, V>
where
    K: Hash + Eq + Ord + Clone + Debug,
    V: Clone + Debug,
{
    caller: &'a ARCache<K, V>,
    // ro_txn to cache
    cache: HashMapReadTxn<'a, K, CacheItem<K, V>>,
    tlocal: Option<ReadCache<K, V>>,
    // tx channel to send forward events.
    tx: Sender<CacheEvent<K, V>>,
    ts: Instant,
}

/// An active write transaction over the cache. The data in this cache is isolated
/// from readers, and may be rolled-back if an error occurs. Changes only become
/// globally visible once you call "commit". Items may be added to the cache on
/// a miss via "insert", and you can explicitly remove items by calling "remove".
pub struct ARCacheWriteTxn<'a, K, V>
where
    K: Hash + Eq + Ord + Clone + Debug,
    V: Clone + Debug,
{
    caller: &'a ARCache<K, V>,
    // wr_txn to cache
    cache: HashMapWriteTxn<'a, K, CacheItem<K, V>>,
    // Cache of missed items (w_ dirty/clean)
    // On COMMIT we drain this to the main cache
    tlocal: Map<K, ThreadCacheItem<V>>,
    hit: UnsafeCell<Vec<K>>,
    clear: UnsafeCell<bool>,
}

impl<K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> ARCache<K, V> {
    /// Create a new ARCache, that derives it's size based on your expected workload.
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

        Self::new_size(max, read_max)
    }

    /// Create a new ARCache, with a capacity of `max` main cache items and `read_max`
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
        ARCache {
            cache: HashMap::new(),
            shared,
            inner,
            stats,
        }
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
        ARCacheReadTxn {
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
    pub async fn write<'a>(&'a self) -> ARCacheWriteTxn<'a, K, V> {
        ARCacheWriteTxn {
            caller: &self,
            cache: self.cache.write().await,
            tlocal: Map::new(),
            hit: UnsafeCell::new(Vec::new()),
            clear: UnsafeCell::new(false),
        }
    }

    /// View the statistics for this cache. These values are a snapshot of a point in
    /// time and may not be accurate at "this exact moment".
    pub fn view_stats(&self) -> CowCellReadTxn<CacheStats> {
        self.stats.read()
    }

    fn try_write(&self) -> Option<ARCacheWriteTxn<K, V>> {
        self.cache.try_write().map(|cache| ARCacheWriteTxn {
            caller: &self,
            cache,
            tlocal: Map::new(),
            hit: UnsafeCell::new(Vec::new()),
            clear: UnsafeCell::new(false),
        })
    }

    fn try_quiesce(&self) {
        if let Some(wr_txn) = self.try_write() {
            wr_txn.commit()
        };
    }

    #[allow(clippy::unnecessary_mut_passed)]
    fn commit<'a>(
        &'a self,
        mut cache: HashMapWriteTxn<'a, K, CacheItem<K, V>>,
        tlocal: Map<K, ThreadCacheItem<V>>,
        hit: Vec<K>,
        clear: bool,
    ) {
        arc_commit!(self, cache, tlocal, hit, clear)
    }
}

impl<'a, K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> ARCacheWriteTxn<'a, K, V> {
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
    pub fn iter_mut_mark_clean(&mut self) -> impl Iterator<Item = (&K, Option<&mut V>)> {
        self.tlocal
            .iter_mut()
            .filter(|(_k, v)| match v {
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
                    ThreadCacheItem::Present(v, _c) => Some(v),
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

impl<'a, K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> ARCacheReadTxn<'a, K, V> {
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
                self.cache.get(k).and_then(|v| {
                    (*v).to_vref().map(|vin| unsafe {
                        let vin = vin as *const _;
                        &(*vin)
                    })
                })
            });

        if r.is_some() {
            let hk: K = k.to_owned().into();
            self.tx
                .send(CacheEvent::Hit(self.ts, hk))
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
                self.ts,
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

impl<'a, K: Hash + Eq + Ord + Clone + Debug, V: Clone + Debug> Drop for ARCacheReadTxn<'a, K, V> {
    fn drop(&mut self) {
        self.caller.try_quiesce();
    }
}

#[cfg(test)]
mod tests {
    use crate::arcache::asynch::ARCache as Arc;
    use crate::arcache::common::CStat;
    use crate::arcache::common::CacheState;
    use async_std::task;

    #[test]
    fn test_cache_arc_basic() {
        let arc: Arc<usize, usize> = Arc::new_size(4, 4);
        let mut wr_txn = task::block_on(arc.write());

        assert!(wr_txn.get(&1) == None);
        assert!(wr_txn.peek_hit().len() == 0);
        wr_txn.insert(1, 1);
        assert!(wr_txn.get(&1) == Some(&1));
        assert!(wr_txn.peek_hit().len() == 1);

        wr_txn.commit();

        // Now we start the second txn, and see if it's in there.
        let wr_txn = task::block_on(arc.write());
        assert!(wr_txn.peek_cache(&1) == CacheState::Rec);
        assert!(wr_txn.get(&1) == Some(&1));
        assert!(wr_txn.peek_hit().len() == 1);
        wr_txn.commit();
        // And now check it's moved to Freq due to the extra
        let wr_txn = task::block_on(arc.write());
        assert!(wr_txn.peek_cache(&1) == CacheState::Freq);
        println!("{:?}", wr_txn.peek_stat());
    }

    #[test]
    fn test_cache_evict() {
        println!("== 1");
        let arc: Arc<usize, usize> = Arc::new_size(4, 4);
        let mut wr_txn = task::block_on(arc.write());
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
        let wr_txn = task::block_on(arc.write());
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
        let mut wr_txn = task::block_on(arc.write());
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
        let wr_txn = task::block_on(arc.write());
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
        let mut wr_txn = task::block_on(arc.write());
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
        let mut wr_txn = task::block_on(arc.write());
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
        let mut wr_txn = task::block_on(arc.write());
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
        let mut wr_txn = task::block_on(arc.write());
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
        let wr_txn = task::block_on(arc.write());
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
        let wr_txn = task::block_on(arc.write());
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
        let wr_txn = task::block_on(arc.write());
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
        let wr_txn = task::block_on(arc.write());
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
        let mut wr_txn = task::block_on(arc.write());
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
        let mut wr_txn = task::block_on(arc.write());
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
        let wr_txn = task::block_on(arc.write());
        // assert that 1 is haunted.
        assert!(wr_txn.peek_cache(&1) == CacheState::Haunted);
        // assert 1 is not in rd.
        assert!(rd_txn.get(&1) == None);
        // now that 1 is hanuted, in rd attempt to insert 1, X
        rd_txn.insert(1, 100);
        // commit wr
        wr_txn.commit();

        // start wr
        let wr_txn = task::block_on(arc.write());
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
        let mut wr_txn = task::block_on(arc.write());
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
        let wr_txn = task::block_on(arc.write());

        // Touch two values that are in the rec set.
        let rec_set: Vec<usize> = wr_txn.iter_rec().take(2).copied().collect();
        println!("{:?}", rec_set);
        assert!(wr_txn.get(&rec_set[0]) == Some(&rec_set[0]));
        assert!(wr_txn.get(&rec_set[1]) == Some(&rec_set[1]));

        // commit wr
        wr_txn.commit();
        // Begin a new write.
        let mut wr_txn = task::block_on(arc.write());
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
        let wr_txn = task::block_on(arc.write());
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
        let arc: Arc<usize, usize> = Arc::new_size(4, 4);

        // Start a wr
        let mut wr_txn = task::block_on(arc.write());
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
        let wr_txn = task::block_on(arc.write());
        let rec_set: Vec<usize> = wr_txn.iter_rec().take(2).copied().collect();
        println!("{:?}", rec_set);
        let r = wr_txn.get(&rec_set[0]);
        println!("{:?}", r);
        assert!(r == Some(&rec_set[0]));
        assert!(wr_txn.get(&rec_set[1]) == Some(&rec_set[1]));

        // commit wr
        wr_txn.commit();
        // Begin a new write.
        let mut wr_txn = task::block_on(arc.write());
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
        let wr_txn = task::block_on(arc.write());
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
        let arc: Arc<usize, usize> = Arc::new_size(4, 4);
        // Setup for the test
        // --
        let mut wr_txn = task::block_on(arc.write());
        wr_txn.insert(10, 1);
        wr_txn.commit();
        // --
        let wr_txn = task::block_on(arc.write());
        assert!(wr_txn.peek_cache(&10) == CacheState::Rec);
        wr_txn.commit();
        // --
        // Okay, now the test starts. First, we begin a read
        let mut rd_txn = arc.read();
        // Then while that read exists, we open a write, and conduct
        // a cache clear.
        let mut wr_txn = task::block_on(arc.write());
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
        let wr_txn = task::block_on(arc.write());
        assert!(wr_txn.peek_cache(&10) == CacheState::GhostRec);
        println!("--> {:?}", wr_txn.peek_cache(&11));
        assert!(wr_txn.peek_cache(&11) == CacheState::None);
    }

    #[test]
    fn test_cache_dirty_write() {
        let arc: Arc<usize, usize> = Arc::new_size(4, 4);
        let mut wr_txn = task::block_on(arc.write());
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
        let wr_txn = task::block_on(arc.write());
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

