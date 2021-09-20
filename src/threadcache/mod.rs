
// Thread local cache.

// Writer locks on an array of sender queues for invalidations.
// When taking read/write, ack our invalidation queue til empty OR up to current txid.

// Max size div by threads.

// Each thread has own ARC.

use std::sync::mpsc::{channel, Sender, Receiver};
use std::sync::Arc;
use parking_lot::{Mutex, MutexGuard};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use std::hash::Hash;
use std::fmt::Debug;
use std::borrow::Borrow;

struct ThreadLocal<K, V>
where
    K: Hash + Eq + Debug + Clone,
{
    tid: usize,
    rx: Receiver<Invalidate<K>>,
    wrlock: Arc<Mutex<Writer<K>>>,
    inv_up_to_txid: Arc<AtomicU64>,
    last_inv: Option<Invalidate<K>>,
    cache: HashMap<K, V>,
}

struct Writer<K>
where
    K: Hash + Eq + Debug + Clone,
{
    txs: Vec<Sender<Invalidate<K>>>,
    txid: usize,
}

struct ThreadLocalWriteTxn<'a, K, V>
where
    K: Hash + Eq + Debug + Clone,
{
    txid: u64,
    parent: &'a mut ThreadLocal<K, V>,
    guard: MutexGuard<'a, Writer<K>>,
    rollback: (),
}

struct ThreadLocalReadTxn<'a, K, V>
where
    K: Hash + Eq + Debug + Clone,
{
    txid: u64,
    parent: &'a mut ThreadLocal<K, V>,
}

struct Invalidate<K>
where
    K: Hash + Eq + Debug + Clone,
{
    k: K,
    txid: u64,
}

impl<K, V> ThreadLocal<K, V>
where
    K: Hash + Eq + Debug + Clone,
{
    pub fn new(
        threads: usize,
        capacity: usize,
    ) -> Vec<Self> {
        assert!(threads > 0);
        let (txs, rxs):
            (Vec<_>, Vec<_>)
            = (0..threads)
            .into_iter()
            .map(|_| {
                channel::<Invalidate<K>>()
            })
            .unzip();

        // Create an Arc<Mutex<txs>> for the writer.
        let inv_up_to_txid = Arc::new(AtomicU64::new(0));
        let wrlock = Arc::new(Mutex::new(
            Writer {
                txs,
                txid: 0,
            }
        ));

        // Then for each thread, take one rx and a clone of the broadcast tbl.
        // Allocate a threadid (tid).

        rxs
            .into_iter()
            .enumerate()
            .map(|(tid, rx)| {
                ThreadLocal {
                    tid,
                    rx,
                    wrlock: wrlock.clone(),
                    inv_up_to_txid: inv_up_to_txid.clone(),
                    last_inv: None,
                    cache: HashMap::with_capacity(capacity),
                }
            })
            .collect()
    }

    pub fn read<'a>(&'a mut self) -> ThreadLocalReadTxn<'a, K, V> {
        let txid = self.inv_up_to_txid.load(Ordering::Acquire);
        self.invalidate(txid);
        ThreadLocalReadTxn {
            txid,
            parent: self,
        }
    }

    pub fn write<'a>(&'a mut self) -> ThreadLocalWriteTxn<'a, K, V> {
        // SAFETY this is safe, because while we are duplicating the mutable reference
        // which conflicts with the mutex, we aren't change the wrlock value so the mutex
        // is fine.
        let parent: &mut Self = unsafe { &mut *(self as *mut _) };
        // We are the only writer!
        let guard = self.wrlock.lock();
        let txid = parent.inv_up_to_txid.load(Ordering::Acquire);
        parent.invalidate(txid);
        ThreadLocalWriteTxn {
            txid,
            parent,
            guard,
            rollback: (),
        }
    }

    pub fn try_write<'a>(&'a mut self) -> Option<ThreadLocalWriteTxn<'a, K, V>> {
        unimplemented!();
    }

    fn invalidate(&mut self, up_to: u64) {
        if let Some(inv) = self.last_inv.as_ref() {
            if inv.txid >= up_to {
                return;
            } else {
                self.cache.remove(&inv.k);
            }
        }

        // We got here, so we must have acted on last_inv.
        self.last_inv = None;

        while let Ok(inv) = self.rx.try_recv() {
            if inv.txid >= up_to {
                // Stash this for next loop.
                self.last_inv = Some(inv);
                return;
            } else {
                self.cache.remove(&inv.k);
            }
        }
    }
}

impl<'a, K, V> ThreadLocalWriteTxn<'a, K, V>
where
    K: Hash + Eq + Debug + Clone,
{
    pub fn get<Q>(&mut self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.parent.cache.get(k)
    }

    pub fn contains_key<Q>(&mut self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.parent.cache.get(k).is_some()
    }

    // insert
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        unimplemented!();
    }

    pub fn invalidate(&mut self, k: &K) {
    }

    // commit
    pub fn commit(self) {
        
    }
}

// impl Drop (for rollback)


impl<'a, K, V> ThreadLocalReadTxn<'a, K, V>
where
    K: Hash + Eq + Debug + Clone,
{
    // get
    // contains key
}


#[cfg(test)]
mod tests {
    use super::ThreadLocal;

    #[test]
    fn test_basic() {
        let mut cache: Vec<ThreadLocal<u32, u32>> = ThreadLocal::new(2, 8);
        let mut cache_a = cache.pop().unwrap();
        let mut cache_b = cache.pop().unwrap();

        let wr_txn = cache_a.write();
        let rd_txn = cache_b.read();
    }
}

