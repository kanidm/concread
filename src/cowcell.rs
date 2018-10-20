//! CowCell - A concurrently readable cell with Arc
//!
//! A CowCell can be used in place of a RwLock. Readers are guaranteed that
//! the data will not change during the lifetime of the read. Readers do
//! not block writers, and writers do not block readers. Writers are serialised
//! same as the write in a RwLock.
//!
//! This is the Arc collected implementation. Arc is slightly slower than EBR,
//! but has better behaviour with very long running read operations, and more
//! accurate memory reclaim behaviour.

extern crate parking_lot;

use std::sync::Arc;
use parking_lot::{Mutex, MutexGuard};
use std::ops::Deref;

#[derive(Debug)]
pub struct CowCellInner<T> {
    data: T,
}

impl<T> CowCellInner<T> {
    pub fn new(data: T) -> Self {
        CowCellInner {
            data: data,
        }
    }
}

type CowCellReadTxn<T> = Arc<CowCellInner<T>>;

#[derive(Debug)]
pub struct CowCell<T> {
    write: Mutex<()>,
    // I suspect that Mutex is faster here due to lack of needing draining.
    // RWlock MT: PT2.354443857S
    // Mutex MT: PT0.006423466S
    // EBR MT: PT0.003360303S
    active: Mutex<CowCellReadTxn<T>>,
}

pub struct CowCellWriteTxn<'a, T: 'a> {
    // Hold open the guard, and initiate the copy to here.
    work: T,
    // This way we know who to contact for updating our data ....
    caller: &'a CowCell<T>,
    _guard: MutexGuard<'a, ()>
}


impl<T> CowCell<T>
    where T: Clone
{
    pub fn new(data: T) -> Self {
        CowCell {
            write: Mutex::new(()),
            active: Mutex::new(
                Arc::new(
                    CowCellInner::new(data)
                )
            ),
        }
    }

    pub fn read(&self) -> CowCellReadTxn<T> {
        let rwguard = self.active.lock();
        rwguard.clone()
        // rwguard ends here
    }

    pub fn write(&self) -> CowCellWriteTxn<T> {
        /* Take the exclusive write lock first */
        let mguard = self.write.lock();
        /* Now take a ro-txn to get the data copied */
        let rwguard = self.active.lock();
        /* This copies the data */
        let data: T = (***rwguard).clone();
        /* Now build the write struct */
        CowCellWriteTxn {
            work: data,
            caller: self,
            _guard: mguard,
        }
    }

    fn commit(&self, newdata: T) {
        let mut rwguard = self.active.lock();
        let new_inner = Arc::new(CowCellInner::new(newdata));
        // now over-write the last value in the mutex.
        *rwguard = new_inner;
    }
}

impl<T> Deref for CowCellInner<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.data
    }
}

impl<T> AsRef<T> for CowCellInner<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.data
    }
}

impl<'a, T> CowCellWriteTxn<'a, T>
    where T: Clone
{
    /* commit */
    /* get_mut data */
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.work
    }

    pub fn commit(self) {
        /* Write our data back to the CowCell */
        self.caller.commit(self.work);
    }
}


#[cfg(test)]
mod tests {
    extern crate time;
    extern crate crossbeam_utils;
    extern crate parking_lot;

    use std::sync::atomic::{AtomicUsize, Ordering};
    use super::CowCell;

    use crossbeam_utils::thread::scope;

    #[test]
    fn test_simple_create() {
        let data: i64 = 0;
        let cc = CowCell::new(data);

        let cc_rotxn_a = cc.read();
        assert_eq!(**cc_rotxn_a, 0);

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
            assert_eq!(**cc_rotxn_a, 0);

            let cc_rotxn_b = cc.read();
            assert_eq!(**cc_rotxn_b, 0);
            /* The write txn and it's lock is dropped here */
            cc_wrtxn.commit();
        }

        /* Start a new txn and see it's still good */
        let cc_rotxn_c = cc.read();
        assert_eq!(**cc_rotxn_c, 1);
        assert_eq!(**cc_rotxn_a, 0);
    }

    const MAX_TARGET: i64 = 2000;

    #[test]
    fn test_multithread_create() {
        let start = time::now();
        // Create the new cowcell.
        let data: i64 = 0;
        let cc = CowCell::new(data);

        scope(|scope| {
            let cc_ref = &cc;

            let _readers: Vec<_> = (0..7).map(|_| {
                scope.spawn(move || {
                    let mut last_value: i64 = 0;
                    while last_value < MAX_TARGET {
                        let cc_rotxn = cc_ref.read();
                        {
                            assert!(**cc_rotxn >= last_value);
                            last_value = **cc_rotxn;
                        }
                    }
                })
            }).collect();

            let _writers: Vec<_> = (0..3).map(|_| {
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
            }).collect();
        });

        let end = time::now();
        print!("Arc MT create :{} ", end - start);
    }

    static GC_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug, Clone)]
    struct TestGcWrapper<T> {
        data: T
    }

    impl<T> Drop for TestGcWrapper<T> {
        fn drop(&mut self) {
            // Add to the atomic counter ...
            GC_COUNT.fetch_add(1, Ordering::Release);
        }
    }

    fn test_gc_operation_thread(cc: &CowCell<TestGcWrapper<i64>>) {
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
    fn test_gc_operation() {
        GC_COUNT.store(0, Ordering::Release);
        let data = TestGcWrapper{data: 0};
        let cc = CowCell::new(data);

        scope(|scope| {
            let cc_ref = &cc;
            let _writers: Vec<_> = (0..3).map(|_| {
                scope.spawn(move || {
                    test_gc_operation_thread(cc_ref);
                })
            }).collect();
        });

        assert!(GC_COUNT.load(Ordering::Acquire) >= 50);
    }

}


