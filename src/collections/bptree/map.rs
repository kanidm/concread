use super::cursor::{CursorRead, CursorWrite};
use super::node::ABNode;
use parking_lot::{Mutex, MutexGuard};
use std::fmt::Debug;

pub struct BptreeMap<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    write: Mutex<()>,
    active: Mutex<ABNode<K, V>>,
}

pub struct BptreeMapReadTxn<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    work: CursorRead<K, V>,
}

pub struct BptreeMapWriteTxn<'a, K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    work: CursorWrite<K, V>,
    caller: &'a BptreeMap<K, V>,
    _guard: MutexGuard<'a, ()>,
}

impl<K: Clone + Ord + Debug, V: Clone> BptreeMap<K, V> {
    pub fn new() -> Self {
        unimplemented!();
    }

    pub fn read(&self) -> BptreeMapReadTxn<K, V> {
        let rguard = self.active.lock();
        BptreeMapReadTxn {
            work: CursorRead::new(rguard.clone()),
        }
        // rguard is dropped, the ABNode lives on!
    }

    pub fn write(&self) -> BptreeMapWriteTxn<K, V> {
        /* Take the exclusive write lock first */
        let mguard = self.write.lock();
        /* Now take a ro-txn to get the data copied */
        let rguard = self.active.lock();
        /*
         * Take a ref to the root, we want to minimise our time in the.
         * active lock. We could do a full clone here but that would trigger
         * node-width worth of atomics, and if the write is dropped without
         * action we've save a lot of cycles.
         */
        let data: ABNode<K, V> = rguard.clone();
        /* Setup the cursor that will work on the tree */
        let cursor = CursorWrite::new(data);

        /* Now build the write struct */
        BptreeMapWriteTxn {
            work: cursor,
            caller: self,
            _guard: mguard,
        }
        /* rguard dropped here */
    }
}

impl<K: Clone + Ord + Debug, V: Clone> BptreeMapReadTxn<K, V> {
    // new

    // clear

    // get

    // contains_key

    // get_mut

    // insert (or update)

    // remove

    // split_off

    // ADVANCED
    // append (join two sets)

    // range/range_mut

    // entry

    // iter

    // iter_mut

    // keys

    // values

    // values_mut

    // len

    // is_empty
}
