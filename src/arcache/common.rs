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

pub(crate) fn calc_p_freq(ghost_rec_len: usize, ghost_freq_len: usize, p: &mut usize) {
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

pub(crate) fn calc_p_rec(cap: usize, ghost_rec_len: usize, ghost_freq_len: usize, p: &mut usize) {
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

/*
    fn drain_tlocal_inc<'a>(
        &'a self,
        cache: &mut HashMapWriteTxn<'a, K, CacheItem<K, V>>,
        $inner: &mut ArcInner<K, V>,
        shared: &ArcShared<K, V>,
        // stats: &mut CacheStats,
        tlocal: Map<K, ThreadCacheItem<V>>,
        commit_txid: u64,
    ) 
*/
macro_rules! drain_tlocal_inc {
    (
        $cache:expr,
        $inner:expr,
        $shared:expr,
        $tlocal:expr,
        $commit_txid:expr
    ) => {{
        // drain tlocal into the main cache.
        $tlocal.into_iter().for_each(|(k, tcio)| {
            let r = $cache.get_mut(&k);
            match (r, tcio) {
                (None, ThreadCacheItem::Present(tci, clean)) => {
                    assert!(clean);
                    let llp = $inner.rec.append_k(CacheItemInner {
                        k: k.clone(),
                        txid: $commit_txid,
                    });
                    $cache.insert(k, CacheItem::Rec(llp, tci));
                }
                (None, ThreadCacheItem::Removed(clean)) => {
                    assert!(clean);
                    // Mark this as haunted
                    let llp = $inner.haunted.append_k(CacheItemInner {
                        k: k.clone(),
                        txid: $commit_txid,
                    });
                    $cache.insert(k, CacheItem::Haunted(llp));
                }
                (Some(ref mut ci), ThreadCacheItem::Removed(clean)) => {
                    assert!(clean);
                    // From whatever set we were in, pop and move to haunted.
                    let mut next_state = match ci {
                        CacheItem::Freq(llp, _v) => {
                            // println!("tlocal {:?} Freq -> Freq", k);
                            unsafe { (**llp).as_mut().txid = $commit_txid };
                            $inner.freq.extract(*llp);
                            $inner.haunted.append_n(*llp);
                            CacheItem::Haunted(*llp)
                        }
                        CacheItem::Rec(llp, _v) => {
                            // println!("tlocal {:?} Rec -> Freq", k);
                            // Remove the node and put it into freq.
                            unsafe { (**llp).as_mut().txid = $commit_txid };
                            $inner.rec.extract(*llp);
                            $inner.haunted.append_n(*llp);
                            CacheItem::Haunted(*llp)
                        }
                        CacheItem::GhostFreq(llp) => {
                            // println!("tlocal {:?} GhostFreq -> Freq", k);
                            unsafe { (**llp).as_mut().txid = $commit_txid };
                            $inner.ghost_freq.extract(*llp);
                            $inner.haunted.append_n(*llp);
                            CacheItem::Haunted(*llp)
                        }
                        CacheItem::GhostRec(llp) => {
                            // println!("tlocal {:?} GhostRec -> Rec", k);
                            unsafe { (**llp).as_mut().txid = $commit_txid };
                            $inner.ghost_rec.extract(*llp);
                            $inner.haunted.append_n(*llp);
                            CacheItem::Haunted(*llp)
                        }
                        CacheItem::Haunted(llp) => {
                            // println!("tlocal {:?} Haunted -> Rec", k);
                            unsafe { (**llp).as_mut().txid = $commit_txid };
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
                            unsafe { (**llp).as_mut().txid = $commit_txid };
                            // println!("tlocal {:?} Freq -> Freq", k);
                            // Move the list item to it's head.
                            $inner.freq.touch(*llp);
                            // Update v.
                            CacheItem::Freq(*llp, (*tci).clone())
                        }
                        CacheItem::Rec(llp, _v) => {
                            // println!("tlocal {:?} Rec -> Freq", k);
                            // Remove the node and put it into freq.
                            unsafe { (**llp).as_mut().txid = $commit_txid };
                            $inner.rec.extract(*llp);
                            $inner.freq.append_n(*llp);
                            CacheItem::Freq(*llp, (*tci).clone())
                        }
                        CacheItem::GhostFreq(llp) => {
                            // println!("tlocal {:?} GhostFreq -> Freq", k);
                            // Ajdust p
                            calc_p_freq(
                                $inner.ghost_rec.len(),
                                $inner.ghost_freq.len(),
                                &mut $inner.p,
                            );
                            unsafe { (**llp).as_mut().txid = $commit_txid };
                            $inner.ghost_freq.extract(*llp);
                            $inner.freq.append_n(*llp);
                            CacheItem::Freq(*llp, (*tci).clone())
                        }
                        CacheItem::GhostRec(llp) => {
                            // println!("tlocal {:?} GhostRec -> Rec", k);
                            // Ajdust p
                            calc_p_rec(
                                $shared.max,
                                $inner.ghost_rec.len(),
                                $inner.ghost_freq.len(),
                                &mut $inner.p,
                            );
                            unsafe { (**llp).as_mut().txid = $commit_txid };
                            $inner.ghost_rec.extract(*llp);
                            $inner.rec.append_n(*llp);
                            CacheItem::Rec(*llp, (*tci).clone())
                        }
                        CacheItem::Haunted(llp) => {
                            // println!("tlocal {:?} Haunted -> Rec", k);
                            unsafe { (**llp).as_mut().txid = $commit_txid };
                            $inner.haunted.extract(*llp);
                            $inner.rec.append_n(*llp);
                            CacheItem::Rec(*llp, (*tci).clone())
                        }
                    };
                    // Now change the state.
                    mem::swap(*ci, &mut next_state);
                }
            }
        });
    }}
}

/*
    fn drain_rx<'a>(
        &'a self,
        cache: &mut HashMapWriteTxn<'a, K, CacheItem<K, V>>,
        $inner: &mut ArcInner<K, V>,
        shared: &ArcShared<K, V>,
        stats: &mut CacheStats,
        commit_ts: Instant,
    ) {
*/

macro_rules! drain_rx {
    (
        $cache:expr,
        $inner:expr,
        $shared:expr,
        $stats:expr,
        $commit_ts:expr
    ) => {{
        // * for each item
        while let Ok(ce) = $inner.rx.try_recv() {
            let t = match ce {
                // Update if it was hit.
                CacheEvent::Hit(t, k) => {
                    $stats.reader_hits += 1;
                    if let Some(ref mut ci) = $cache.get_mut(&k) {
                        let mut next_state = match &ci {
                            CacheItem::Freq(llp, v) => {
                                // println!("rxhit {:?} Freq -> Freq", k);
                                $inner.freq.touch(*llp);
                                CacheItem::Freq(*llp, v.clone())
                            }
                            CacheItem::Rec(llp, v) => {
                                // println!("rxhit {:?} Rec -> Freq", k);
                                $inner.rec.extract(*llp);
                                $inner.freq.append_n(*llp);
                                CacheItem::Freq(*llp, v.clone())
                            }
                            // While we can't add this from nothing, we can
                            // at least keep it in the ghost sets.
                            CacheItem::GhostFreq(llp) => {
                                // println!("rxhit {:?} GhostFreq -> GhostFreq", k);
                                $inner.ghost_freq.touch(*llp);
                                CacheItem::GhostFreq(*llp)
                            }
                            CacheItem::GhostRec(llp) => {
                                // println!("rxhit {:?} GhostRec -> GhostRec", k);
                                $inner.ghost_rec.touch(*llp);
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
                    // Do nothing, it must have been evicted.
                    t
                }
                // Update if it was inc
                CacheEvent::Include(t, k, iv, txid) => {
                    $stats.reader_includes += 1;
                    let mut r = $cache.get_mut(&k);
                    match r {
                        Some(ref mut ci) => {
                            let mut next_state = match &ci {
                                CacheItem::Freq(llp, _v) => {
                                    $inner.freq.touch(*llp);
                                    if unsafe { (**llp).as_ref().txid >= txid }
                                        || $inner.min_txid > txid
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
                                    $inner.rec.extract(*llp);
                                    $inner.freq.append_n(*llp);
                                    if unsafe { (**llp).as_ref().txid >= txid }
                                        || $inner.min_txid > txid
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
                                    calc_p_freq(
                                        $inner.ghost_rec.len(),
                                        $inner.ghost_freq.len(),
                                        &mut $inner.p,
                                    );
                                    $inner.ghost_freq.extract(*llp);
                                    if unsafe { (**llp).as_ref().txid > txid }
                                        || $inner.min_txid > txid
                                    {
                                        // println!("rxinc {:?} GhostFreq -> GhostFreq", k);
                                        // The cache version is newer, this is just a hit.
                                        $inner.ghost_freq.append_n(*llp);
                                        None
                                    } else {
                                        // This item is newer, so we can include it.
                                        // println!("rxinc {:?} GhostFreq -> Rec", k);
                                        $inner.freq.append_n(*llp);
                                        unsafe { (**llp).as_mut().txid = txid };
                                        Some(CacheItem::Freq(*llp, iv))
                                    }
                                }
                                CacheItem::GhostRec(llp) => {
                                    // Adjust p
                                    calc_p_rec(
                                        $shared.max,
                                        $inner.ghost_rec.len(),
                                        $inner.ghost_freq.len(),
                                        &mut $inner.p,
                                    );
                                    if unsafe { (**llp).as_ref().txid > txid }
                                        || $inner.min_txid > txid
                                    {
                                        // println!("rxinc {:?} GhostRec -> GhostRec", k);
                                        $inner.ghost_rec.touch(*llp);
                                        None
                                    } else {
                                        // println!("rxinc {:?} GhostRec -> Rec", k);
                                        $inner.ghost_rec.extract(*llp);
                                        $inner.rec.append_n(*llp);
                                        unsafe { (**llp).as_mut().txid = txid };
                                        Some(CacheItem::Rec(*llp, iv))
                                    }
                                }
                                CacheItem::Haunted(llp) => {
                                    if unsafe { (**llp).as_ref().txid > txid }
                                        || $inner.min_txid > txid
                                    {
                                        // println!("rxinc {:?} Haunted -> Haunted", k);
                                        None
                                    } else {
                                        // println!("rxinc {:?} Haunted -> Rec", k);
                                        $inner.haunted.extract(*llp);
                                        $inner.rec.append_n(*llp);
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
                            if txid >= $inner.min_txid {
                                let llp = $inner.rec.append_k(CacheItemInner { k: k.clone(), txid });
                                $cache.insert(k, CacheItem::Rec(llp, iv));
                            }
                        }
                    };
                    t
                }
            };
            // Stop processing the queue, we are up to "now".
            if t >= $commit_ts {
                break;
            }
        }
    }}
}

    /*
    fn drain_tlocal_hits<'a>(
        &'a self,
        cache: &mut HashMapWriteTxn<'a, K, CacheItem<K, V>>,
        $inner: &mut ArcInner<K, V>,
        // shared: &ArcShared<K, V>,
        // stats: &mut CacheStats,
        commit_txid: u64,
        hit: Vec<K>,
    ) {
    */

macro_rules! drain_tlocal_hits {
    (
        $cache:expr,
        $inner:expr,
        $commit_txid:expr,
        $hit:expr
    ) => {{
        $hit.into_iter().for_each(|k| {
            // * everything hit must be in main cache now, so bring these
            //   all to the relevant item heads.
            // * Why do this last? Because the write is the "latest" we want all the fresh
            //   written items in the cache over the "read" hits, it gives us some aprox
            //   of time ordering, but not perfect.

            // Find the item in the cache.
            // * based on it's type, promote it in the correct list, or move it.
            // How does this prevent incorrect promotion from rec to freq? txid?
            // println!("Checking Hit ... {:?}", k);
            let mut r = $cache.get_mut(&k);
            match r {
                Some(ref mut ci) => {
                    // This differs from above - we skip if we don't touch anything
                    // that was added in this txn. This is to prevent double touching
                    // anything that was included in a write.
                    let mut next_state = match &ci {
                        CacheItem::Freq(llp, v) => {
                            if unsafe { (**llp).as_ref().txid != $commit_txid } {
                                // println!("hit {:?} Freq -> Freq", k);
                                $inner.freq.touch(*llp);
                                Some(CacheItem::Freq(*llp, v.clone()))
                            } else {
                                None
                            }
                        }
                        CacheItem::Rec(llp, v) => {
                            if unsafe { (**llp).as_ref().txid != $commit_txid } {
                                // println!("hit {:?} Rec -> Freq", k);
                                $inner.rec.extract(*llp);
                                $inner.freq.append_n(*llp);
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
    }}
}

/*
    #[allow(clippy::cognitive_complexity)]
    fn evict<'a>(
        &'a self,
        cache: &mut HashMapWriteTxn<'a, K, CacheItem<K, V>>,
        $inner: &mut ArcInner<K, V>,
        shared: &ArcShared<K, V>,
        stats: &mut CacheStats,
        commit_txid: u64,
    ) {
    */

macro_rules! evict {
    (
        $cache:expr,
        $inner:expr,
        $shared:expr,
        $stats:expr,
        $commit_txid:expr
    ) => {{
        debug_assert!($inner.p <= $shared.max);
        // Convince the compiler copying is okay.
        let p = $inner.p;
        $stats.p_weight = p;

        if $inner.rec.len() + $inner.freq.len() > $shared.max {
            // println!("Checking cache evict");
            /*
            println!(
                "from -> rec {:?}, freq {:?}",
                $inner.rec.len(),
                $inner.freq.len()
            );
            */
            let delta = ($inner.rec.len() + $inner.freq.len()) - $shared.max;
            // We have overflowed by delta. As we are not "evicting as we go" we have to work out
            // what we should have evicted up to now.
            //
            // keep removing from rec until == p OR delta == 0, and if delta remains, then remove from freq.

            let rec_to_len = if $inner.p == 0 {
                // println!("p == 0 => {:?}", $inner.rec.len());
                debug_assert!(delta <= $inner.rec.len());
                // We are fully weight to freq, so only remove in rec.
                $inner.rec.len() - delta
            } else if $inner.rec.len() > $inner.p {
                // There is a partial weighting, how much do we need to move?
                let rec_delta = $inner.rec.len() - $inner.p;
                if rec_delta > delta {
                    /*
                    println!(
                        "p ({:?}) <= rec ({:?}), rec_delta ({:?}) > delta ({:?})",
                        $inner.p,
                        $inner.rec.len(),
                        rec_delta,
                        delta
                    );
                    */
                    // We will have removed enough through delta alone in rec.
                    $inner.rec.len() - delta
                } else {
                    /*
                    println!(
                        "p ({:?}) <= rec ({:?}), rec_delta ({:?}) <= delta ({:?})",
                        $inner.p,
                        $inner.rec.len(),
                        rec_delta,
                        delta
                    );
                    */
                    // Remove the full delta, and excess will be removed from freq.
                    $inner.rec.len() - rec_delta
                }
            } else {
                // rec is already below p, therefore we must need to remove in freq, and
                // we need to consider how much is in rec.
                // println!("p ({:?}) > rec ({:?})", $inner.p, $inner.rec.len());
                $inner.rec.len()
            };

            // Now we can get the expected sizes;
            debug_assert!($shared.max >= rec_to_len);
            let freq_to_len = $shared.max - rec_to_len;
            // println!("move to -> rec {:?}, freq {:?}", rec_to_len, freq_to_len);
            debug_assert!(freq_to_len + rec_to_len <= $shared.max);

            $stats.freq_evicts += $inner.freq.len() - freq_to_len;
            $stats.recent_evicts += $inner.rec.len() - rec_to_len;

            evict_to_len!(
                $cache,
                $inner.rec,
                &mut $inner.ghost_rec,
                rec_to_len,
                $commit_txid
            );
            evict_to_len!(
                $cache,
                $inner.freq,
                &mut $inner.ghost_freq,
                freq_to_len,
                $commit_txid
            );

            // Finally, do an evict of the ghost sets if they are too long - these are weighted
            // inverse to the above sets. Note the freq to len in ghost rec, and rec to len in
            // ghost freq!
            if $inner.ghost_rec.len() > ($shared.max - p) {
                evict_to_haunted_len!(
                    $cache,
                    $inner.ghost_rec,
                    &mut $inner.haunted,
                    freq_to_len,
                    $commit_txid
                );
            }

            if $inner.ghost_freq.len() > p {
                evict_to_haunted_len!(
                    $cache,
                    $inner.ghost_freq,
                    &mut $inner.haunted,
                    rec_to_len,
                    $commit_txid
                );
            }
        }
    }}
}


macro_rules! arc_commit {
    (
        $self:expr,
        $cache:expr,
        $tlocal:expr,
        $hit:expr,
        $clear:expr
    ) => {{
        // What is the time?
        let commit_ts = Instant::now();
        let commit_txid = $cache.get_txid();
        // Copy p + init cache sizes for adjustment.
        let mut inner = $self.inner.lock();
        let shared = $self.shared.read();
        let mut stat_guard = $self.stats.write();
        let stats = stat_guard.get_mut();

        // Did we request to be cleared? If so, we move everything to a ghost set
        // that was live.
        //
        // we also set the min_txid watermark which prevents any inclusion of
        // any item that existed before this point in time.
        if $clear {
            // Set the watermark of this txn.
            inner.min_txid = commit_txid;

            // Indicate that we evicted all to ghost/freq
            stats.freq_evicts += inner.freq.len();
            stats.recent_evicts += inner.rec.len();

            // Move everything active into ghost sets.
            drain_ll_to_ghost!(
                &mut $cache,
                inner.freq,
                inner.ghost_freq,
                inner.ghost_rec,
                commit_txid
            );
            drain_ll_to_ghost!(
                &mut $cache,
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

        stats.write_inc_or_mod += $tlocal.len();

        drain_tlocal_inc!(
            &mut $cache,
            inner.deref_mut(),
            shared.deref(),
            $tlocal,
            commit_txid
        );

        // drain rx until empty or time >= time.
        drain_rx!(
            &mut $cache,
            inner.deref_mut(),
            shared.deref(),
            stats,
            commit_ts
        );

        stats.write_hits += $hit.len();
        // drain the tlocal hits into the main cache.

        drain_tlocal_hits!(&mut $cache, inner.deref_mut(), commit_txid, $hit);

        // now clean the space for each of the primary caches, evicting into the ghost sets.
        // * It's possible that both caches are now over-sized if rx was empty
        //   but wr inc many items.
        // * p has possibly changed width, causing a balance shift
        // * and ghost items have been included changing ghost list sizes.
        // so we need to do a clean up/balance of all the list lengths.
        evict!(
            &mut $cache,
            inner.deref_mut(),
            shared.deref(),
            stats,
            commit_txid
        );

        stats.shared_max = shared.max;
        stats.freq = inner.freq.len();
        stats.recent = inner.rec.len();
        stats.all_seen_keys = $cache.len();

        // Commit the stats
        stat_guard.commit();
        // commit on the wr txn.
        $cache.commit();
        // done!
    }}
}

