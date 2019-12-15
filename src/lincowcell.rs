//! A CowCell with linear drop behaviour
//!
//! DO NOT USE THIS TYPE! It's private as it's required for the future
//! BTree type. Normaly concread values do not require the linear dropping
//! behaviour that this implements, and it will only make your application
//! slower for it. Consider `CowCell` and `EbrCell` instead.

use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Debug)]
pub struct LinCowCellInner<T> {
    data: T,
    next: Mutex<Option<LinCowCellReadTxn<T>>>,
}

impl<T> LinCowCellInner<T> {
    pub fn new(data: T) -> Self {
        LinCowCellInner {
            data: data,
            next: Mutex::new(None),
        }
    }
}

type LinCowCellReadTxn<T> = Arc<LinCowCellInner<T>>;

#[derive(Debug)]
pub struct LinCowCell<T> {
    write: Mutex<()>,
    // I suspect that Mutex is faster here due to lack of needing draining.
    // RWlock 500 MT: PT2.354443857S
    // Mutex 500 MT: PT0.006423466S
    // EBR 500 MT: PT0.003360303S
    active: Mutex<LinCowCellReadTxn<T>>,
}

#[derive(Debug)]
pub struct LinCowCellWriteTxn<'a, T: 'a> {
    // Hold open the guard, and initiate the copy to here.
    work: T,
    // This way we know who to contact for updating our data ....
    caller: &'a LinCowCell<T>,
    guard: MutexGuard<'a, ()>,
}

impl<T> LinCowCell<T>
where
    T: Clone,
{
    pub fn new(data: T) -> Self {
        LinCowCell {
            write: Mutex::new(()),
            active: Mutex::new(Arc::new(LinCowCellInner::new(data))),
        }
    }

    pub fn read(&self) -> LinCowCellReadTxn<T> {
        let rwguard = self.active.lock().unwrap();
        rwguard.clone()
        /*
        LinCowCellReadTxn {
            data: rwguard.data.clone()
        }
        */
        // rwguard ends here
    }

    pub fn write(&self) -> LinCowCellWriteTxn<T> {
        /* Take the exclusive write lock first */
        let mguard = self.write.lock().unwrap();
        /* Now take a ro-txn to get the data copied */
        let rwguard = self.active.lock().unwrap();
        /* This copies the data */
        let data: T = (***rwguard).clone();
        /* Now build the write struct */
        LinCowCellWriteTxn {
            work: data,
            caller: self,
            guard: mguard,
        }
    }

    fn commit(&self, newdata: T) {
        let mut rwguard = self.active.lock().unwrap();
        let new_inner = Arc::new(LinCowCellInner::new(newdata));
        {
            // This modiries the next pointer of the existing read txns
            let mut rwguard_inner = rwguard.next.lock().unwrap();
            // Create the arc pointer to our new data
            // add it to the last value
            *rwguard_inner = Some(new_inner.clone());
        }
        // now over-write the last value in the mutex.
        *rwguard = new_inner;
    }
}

impl<T> Deref for LinCowCellInner<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.data
    }
}

impl<T> AsRef<T> for LinCowCellInner<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.data
    }
}

impl<'a, T> LinCowCellWriteTxn<'a, T>
where
    T: Clone,
{
    /* commit */
    /* get_mut data */
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.work
    }

    pub fn commit(self) {
        /* Write our data back to the LinCowCell */
        self.caller.commit(self.work);
    }
}

#[cfg(test)]
mod tests {
    extern crate crossbeam_utils;
    extern crate time;

    use super::LinCowCell;
    use crossbeam_utils::thread::scope;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_simple_create() {
        let data: i64 = 0;
        let cc = LinCowCell::new(data);

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

    fn mt_writer(cc: &LinCowCell<i64>) {
        let mut last_value: i64 = 0;
        while last_value < 500 {
            let mut cc_wrtxn = cc.write();
            {
                let mut_ptr = cc_wrtxn.get_mut();
                assert!(*mut_ptr >= last_value);
                last_value = *mut_ptr;
                *mut_ptr = *mut_ptr + 1;
            }
            cc_wrtxn.commit();
        }
    }

    fn rt_writer(cc: &LinCowCell<i64>) {
        let mut last_value: i64 = 0;
        while last_value < 500 {
            let cc_rotxn = cc.read();
            {
                assert!(**cc_rotxn >= last_value);
                last_value = **cc_rotxn;
            }
        }
    }

    #[test]
    fn test_multithread_create() {
        let start = time::now();
        // Create the new cowcell.
        let data: i64 = 0;
        let cc = LinCowCell::new(data);

        scope(|scope| {
            let cc_ref = &cc;

            let _readers: Vec<_> = (0..7)
                .map(|_| {
                    scope.spawn(move || {
                        rt_writer(cc_ref);
                    })
                })
                .collect();

            let _writers: Vec<_> = (0..3)
                .map(|_| {
                    scope.spawn(move || {
                        mt_writer(cc_ref);
                    })
                })
                .collect();
        });

        let end = time::now();
        print!("Arc MT create :{} ", end - start);
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

    fn test_gc_operation_thread(cc: &LinCowCell<TestGcWrapper<i64>>) {
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
        let data = TestGcWrapper { data: 0 };
        let cc = LinCowCell::new(data);

        scope(|scope| {
            let cc_ref = &cc;
            let _writers: Vec<_> = (0..3)
                .map(|_| {
                    scope.spawn(move || {
                        test_gc_operation_thread(cc_ref);
                    })
                })
                .collect();
        });

        assert!(GC_COUNT.load(Ordering::Acquire) >= 50);
    }

}

#[cfg(test)]
mod tests_linear {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::LinCowCell;

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

    /*
     * This tests an important property of the lincowcell over the cow cell
     * that read txns are dropped *in order*.
     */
    #[test]
    fn test_gc_operation_linear() {
        GC_COUNT.store(0, Ordering::Release);
        assert!(GC_COUNT.load(Ordering::Acquire) == 0);
        let data = TestGcWrapper { data: 0 };
        let cc = LinCowCell::new(data);

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
        assert!(GC_COUNT.load(Ordering::Acquire) == 2);
    }
}
