//! EbrCell - A concurrently readable cell with Ebr
//!
//! An [EbrCell] can be used in place of a `RwLock`. Readers are guaranteed that
//! the data will not change during the lifetime of the read. Readers do
//! not block writers, and writers do not block readers. Writers are serialised
//! same as the write in a `RwLock`.
//!
//! This is the Ebr collected implementation.
//! Ebr is the crossbeam-epoch based reclaim system for async memory
//! garbage collection. Ebr is faster than `Arc`,
//! but long transactions can cause the memory usage to grow very quickly
//! before a garbage reclaim. This is a space time trade, where you gain
//! performance at the expense of delaying garbage collection. Holding Ebr
//! reads for too long may impact garbage collection of other epoch structures
//! or crossbeam library components.
//! If you need accurate memory reclaim, use the Arc (`CowCell`) implementation.

use crossbeam_epoch as epoch;
use crossbeam_epoch::{Atomic, Guard, Owned};
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};

use std::marker::Send;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard};

/// An `EbrCell` Write Transaction handle.
///
/// This allows mutation of the content of the `EbrCell` without blocking or
/// affecting current readers.
///
/// Changes are only stored in the structure until you call commit: to
/// abort a change, don't call commit and allow the write transaction to
/// go out of scope. This causes the `EbrCell` to unlock allowing other
/// writes to proceed.
pub struct EbrCellWriteTxn<'a, T: 'static + Clone + Send + Sync> {
    data: Option<T>,
    // This way we know who to contact for updating our data ....
    caller: &'a EbrCell<T>,
    _guard: MutexGuard<'a, ()>,
}

impl<'a, T> EbrCellWriteTxn<'a, T>
where
    T: Clone + Sync + Send + 'static,
{
    /// Access a mutable pointer of the data in the `EbrCell`. This data is only
    /// visible to this write transaction object in this thread until you call
    /// 'commit'.
    pub fn get_mut(&mut self) -> &mut T {
        self.data.as_mut().unwrap()
    }

    /// Commit the changes in this write transaction to the `EbrCell`. This will
    /// consume the transaction so that further changes can not be made to it
    /// after this function is called.
    pub fn commit(mut self) {
        /* Write our data back to the EbrCell */
        // Now make a new dummy element, and swap it into the mutex
        // This fixes up ownership of some values for lifetimes.
        let mut element: Option<T> = None;
        mem::swap(&mut element, &mut self.data);
        self.caller.commit(element);
    }
}

impl<'a, T> Deref for EbrCellWriteTxn<'a, T>
where
    T: Clone + Sync + Send,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.data.as_ref().unwrap()
    }
}

impl<'a, T> DerefMut for EbrCellWriteTxn<'a, T>
where
    T: Clone + Sync + Send,
{
    fn deref_mut(&mut self) -> &mut T {
        self.data.as_mut().unwrap()
    }
}

/// A concurrently readable cell.
///
/// This structure behaves in a similar manner to a `RwLock<Box<T>>`. However
/// unlike a read-write lock, writes and parallel reads can be performed
/// simultaneously. This means writes do not block reads or reads do not
/// block writes.
///
/// To achieve this a form of "copy-on-write" (or for Rust, clone on write) is
/// used. As a write transaction begins, we clone the existing data to a new
/// location that is capable of being mutated.
///
/// Readers are guaranteed that the content of the `EbrCell` will live as long
/// as the read transaction is open, and will be consistent for the duration
/// of the transaction. There can be an "unlimited" number of readers in parallel
/// accessing different generations of data of the `EbrCell`.
///
/// Data that is copied is garbage collected using the crossbeam-epoch library.
///
/// Writers are serialised and are guaranteed they have exclusive write access
/// to the structure.
///
/// # Examples
/// ```
/// use concread::ebrcell::EbrCell;
///
/// let data: i64 = 0;
/// let ebrcell = EbrCell::new(data);
///
/// // Begin a read transaction
/// let read_txn = ebrcell.read();
/// assert_eq!(*read_txn, 0);
/// {
///     // Now create a write, and commit it.
///     let mut write_txn = ebrcell.write();
///     *write_txn = 1;
///     // Commit the change
///     write_txn.commit();
/// }
/// // Show the previous generation still reads '0'
/// assert_eq!(*read_txn, 0);
/// let new_read_txn = ebrcell.read();
/// // And a new read transaction has '1'
/// assert_eq!(*new_read_txn, 1);
/// ```
#[derive(Debug)]
pub struct EbrCell<T: Clone + Sync + Send + 'static> {
    write: Mutex<()>,
    active: Atomic<T>,
}

impl<T> EbrCell<T>
where
    T: Clone + Sync + Send + 'static,
{
    /// Create a new `EbrCell` storing type `T`. `T` must implement `Clone`.
    pub fn new(data: T) -> Self {
        EbrCell {
            write: Mutex::new(()),
            active: Atomic::new(data),
        }
    }

    /// Begin a write transaction, returning a write guard.
    pub fn write(&self) -> EbrCellWriteTxn<T> {
        /* Take the exclusive write lock first */
        let mguard = self.write.lock().unwrap();
        /* Do an atomic load of the current value */
        let guard = epoch::pin();
        let cur_shared = self.active.load(Acquire, &guard);
        /* Now build the write struct, we'll discard the pin shortly! */
        EbrCellWriteTxn {
            /* This is the 'copy' of the copy on write! */
            data: Some(unsafe { cur_shared.deref().clone() }),
            caller: self,
            _guard: mguard,
        }
    }

    /// Attempt to begin a write transaction. If it's already held,
    /// `None` is returned.
    pub fn try_write(&self) -> Option<EbrCellWriteTxn<T>> {
        self.write.try_lock().ok().map(|mguard| {
            let guard = epoch::pin();
            let cur_shared = self.active.load(Acquire, &guard);
            /* Now build the write struct, we'll discard the pin shortly! */
            EbrCellWriteTxn {
                /* This is the 'copy' of the copy on write! */
                data: Some(unsafe { cur_shared.deref().clone() }),
                caller: self,
                _guard: mguard,
            }
        })
    }

    /// This is an internal compontent of the commit cycle. It takes ownership
    /// of the value stored in the writetxn, and commits it to the main EbrCell
    /// safely.
    ///
    /// In theory you could use this as a "lock free" version, but you don't
    /// know if you are trampling a previous change, so it's private and we
    /// let the writetxn struct serialise and protect this interface.
    fn commit(&self, element: Option<T>) {
        // Yield a read txn?
        let guard = epoch::pin();

        // Load the previous data ready for unlinking
        let prev_data = self.active.load(Acquire, &guard);
        // Make the data Owned, and set it in the active.
        let owned_data: Owned<T> = Owned::new(element.unwrap());
        let _shared_data = self
            .active
            .compare_exchange(prev_data, owned_data, Release, Relaxed, &guard);
        // Finally, set our previous data for cleanup.
        unsafe { guard.defer_destroy(prev_data) };
        // Then return the current data with a readtxn. Do we need a new guard scope?
    }

    /// Begin a read transaction. The returned [`EbrCellReadTxn'] guarantees
    /// the data lives long enough via crossbeam's Epoch type. When this is
    /// dropped the data *may* be freed at some point in the future.
    pub fn read(&self) -> EbrCellReadTxn<T> {
        let guard = epoch::pin();

        // This option returns None on null pointer, but we can never be null
        // as we have to init with data, and all replacement ALWAYS gives us
        // a ptr, so unwrap?
        let cur = {
            let c = self.active.load(Acquire, &guard);
            c.as_raw()
        };

        EbrCellReadTxn {
            _guard: guard,
            data: cur,
        }
    }
}

impl<T> Drop for EbrCell<T>
where
    T: Clone + Sync + Send + 'static,
{
    fn drop(&mut self) {
        // Right, we are dropping! Everything is okay here *except*
        // that we need to tell our active data to be unlinked, else it may
        // be dropped "unsafely".
        let guard = epoch::pin();

        let prev_data = self.active.load(Acquire, &guard);
        unsafe { guard.defer_destroy(prev_data) };
    }
}

/// A read transaction. This stores a reference to the data from the main
/// `EbrCell`, and guarantees it is alive for the duration of the read.
// #[derive(Debug)]
pub struct EbrCellReadTxn<T> {
    _guard: Guard,
    data: *const T,
}

impl<T> Deref for EbrCellReadTxn<T> {
    type Target = T;

    /// Derference and access the value within the read transaction.
    fn deref(&self) -> &T {
        unsafe { &(*self.data) }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::EbrCell;
    use std::thread::scope;

    #[test]
    fn test_deref_mut() {
        let data: i64 = 0;
        let cc = EbrCell::new(data);
        {
            /* Take a write txn */
            let mut cc_wrtxn = cc.write();
            *cc_wrtxn = 1;
            cc_wrtxn.commit();
        }
        let cc_rotxn = cc.read();
        assert_eq!(*cc_rotxn, 1);
    }

    #[test]
    fn test_try_write() {
        let data: i64 = 0;
        let cc = EbrCell::new(data);
        /* Take a write txn */
        let cc_wrtxn_a = cc.try_write();
        assert!(cc_wrtxn_a.is_some());
        /* Because we already hold the writ, the second is guaranteed to fail */
        let cc_wrtxn_a = cc.try_write();
        assert!(cc_wrtxn_a.is_none());
    }

    #[test]
    fn test_simple_create() {
        let data: i64 = 0;
        let cc = EbrCell::new(data);

        let cc_rotxn_a = cc.read();
        assert_eq!(*cc_rotxn_a, 0);

        {
            /* Take a write txn */
            let mut cc_wrtxn = cc.write();
            /* Get the data ... */
            {
                let mut_ptr = cc_wrtxn.get_mut();
                /* Assert it's 0 */
                assert_eq!(*mut_ptr, 0);
                *mut_ptr = 1;
                assert_eq!(*mut_ptr, 1);
            }
            assert_eq!(*cc_rotxn_a, 0);

            let cc_rotxn_b = cc.read();
            assert_eq!(*cc_rotxn_b, 0);
            /* The write txn and it's lock is dropped here */
            cc_wrtxn.commit();
        }

        /* Start a new txn and see it's still good */
        let cc_rotxn_c = cc.read();
        assert_eq!(*cc_rotxn_c, 1);
        assert_eq!(*cc_rotxn_a, 0);
    }

    const MAX_TARGET: i64 = 2000;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_multithread_create() {
        let start = time::Instant::now();
        // Create the new ebrcell.
        let data: i64 = 0;
        let cc = EbrCell::new(data);

        assert!(scope(|scope| {
            let cc_ref = &cc;

            let readers: Vec<_> = (0..7)
                .map(|_| {
                    scope.spawn(move || {
                        let mut last_value: i64 = 0;
                        while last_value < MAX_TARGET {
                            let cc_rotxn = cc_ref.read();
                            {
                                assert!(*cc_rotxn >= last_value);
                                last_value = *cc_rotxn;
                            }
                        }
                    })
                })
                .collect();

            let writers: Vec<_> = (0..3)
                .map(|_| {
                    scope.spawn(move || {
                        let mut last_value: i64 = 0;
                        while last_value < MAX_TARGET {
                            let mut cc_wrtxn = cc_ref.write();
                            {
                                let mut_ptr = cc_wrtxn.get_mut();
                                assert!(*mut_ptr >= last_value);
                                last_value = *mut_ptr;
                                *mut_ptr = *mut_ptr + 1;
                            }
                            cc_wrtxn.commit();
                        }
                    })
                })
                .collect();

            for h in readers.into_iter() {
                h.join().unwrap();
            }
            for h in writers.into_iter() {
                h.join().unwrap();
            }
            true
        }));

        let end = time::Instant::now();
        print!("Ebr MT create :{:?} ", end - start);
    }

    static GC_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug, Clone)]
    struct TestGcWrapper<T> {
        data: T,
    }

    impl<T> Drop for TestGcWrapper<T> {
        fn drop(&mut self) {
            // Add to the atomic counter ...
            GC_COUNT.fetch_add(1, Ordering::Release);
        }
    }

    fn test_gc_operation_thread(cc: &EbrCell<TestGcWrapper<i64>>) {
        while GC_COUNT.load(Ordering::Acquire) < 50 {
            // thread::sleep(std::time::Duration::from_millis(200));
            {
                let mut cc_wrtxn = cc.write();
                {
                    let mut_ptr = cc_wrtxn.get_mut();
                    mut_ptr.data = mut_ptr.data + 1;
                }
                cc_wrtxn.commit();
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_gc_operation() {
        GC_COUNT.store(0, Ordering::Release);
        let data = TestGcWrapper { data: 0 };
        let cc = EbrCell::new(data);

        assert!(scope(|scope| {
            let cc_ref = &cc;
            let writers: Vec<_> = (0..3)
                .map(|_| {
                    scope.spawn(move || {
                        test_gc_operation_thread(cc_ref);
                    })
                })
                .collect();

            for h in writers.into_iter() {
                h.join().unwrap();
            }
            true
        }));

        assert!(GC_COUNT.load(Ordering::Acquire) >= 50);
    }
}

#[cfg(test)]
mod tests_linear {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::EbrCell;

    static GC_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug, Clone)]
    struct TestGcWrapper<T> {
        data: T,
    }

    impl<T> Drop for TestGcWrapper<T> {
        fn drop(&mut self) {
            // Add to the atomic counter ...
            GC_COUNT.fetch_add(1, Ordering::Release);
        }
    }

    #[test]
    fn test_gc_operation_linear() {
        /*
         * Test if epoch drops in order (or ordered enough).
         * A property required for b+tree with cow is that txn's
         * are dropped in order so that tree states are not invalidated.
         *
         * A -> B -> C
         *
         * If B is dropped, it invalidates nodes copied from A
         * causing the tree to corrupt txn A (and maybe C).
         *
         * EBR due to it's design while it won't drop in order,
         * it drops generationally, in blocks. This is probably
         * good enough. This means that:
         *
         * A -> B -> C .. -> X -> Y
         *
         * EBR will drop in blocks such as:
         *
         * |  g1   |  g2   |  live |
         * A -> B -> C .. -> X -> Y
         *
         * This test is "small" but asserts a basic sanity of drop
         * ordering, but it's not conclusive for b+tree. More testing
         * (likely multi-thread strees test) is needed, or analysis from
         * other EBR developers.
         */
        GC_COUNT.store(0, Ordering::Release);
        let data = TestGcWrapper { data: 0 };
        let cc = EbrCell::new(data);

        // Open a read A.
        let cc_rotxn_a = cc.read();
        // open a write, change and commit
        {
            let mut cc_wrtxn = cc.write();
            {
                let mut_ptr = cc_wrtxn.get_mut();
                mut_ptr.data = mut_ptr.data + 1;
            }
            cc_wrtxn.commit();
        }
        // open a read B.
        let cc_rotxn_b = cc.read();
        // open a write, change and commit
        {
            let mut cc_wrtxn = cc.write();
            {
                let mut_ptr = cc_wrtxn.get_mut();
                mut_ptr.data = mut_ptr.data + 1;
            }
            cc_wrtxn.commit();
        }
        // open a read C
        let cc_rotxn_c = cc.read();

        assert!(GC_COUNT.load(Ordering::Acquire) == 0);

        // drop B
        drop(cc_rotxn_b);

        // gc count should be 0.
        assert!(GC_COUNT.load(Ordering::Acquire) == 0);

        // drop C
        drop(cc_rotxn_c);

        // gc count should be 0
        assert!(GC_COUNT.load(Ordering::Acquire) == 0);

        // drop A
        drop(cc_rotxn_a);

        // gc count should be 2 (A + B, C is still live)
        assert!(GC_COUNT.load(Ordering::Acquire) <= 2);
    }
}
