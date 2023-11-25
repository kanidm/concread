//! See the documentation for [BptreeMap](crate::bptree::BptreeMap)

#[cfg(feature = "asynch")]
pub mod asynch;

use crate::internals::lincowcell::{LinCowCell, LinCowCellReadTxn, LinCowCellWriteTxn};

include!("impl.rs");

impl<K: Clone + Ord + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static>
    BptreeMap<K, V>
{
    /// Initiate a read transaction for the tree, concurrent to any
    /// other readers or writers.
    pub fn read(&self) -> BptreeMapReadTxn<K, V> {
        let inner = self.inner.read();
        BptreeMapReadTxn { inner }
    }

    /// Initiate a write transaction for the tree, exclusive to this
    /// writer, and concurrently to all existing reads.
    pub fn write(&self) -> BptreeMapWriteTxn<K, V> {
        let inner = self.inner.write();
        BptreeMapWriteTxn { inner }
    }
}

impl<K: Clone + Ord + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static>
    BptreeMapWriteTxn<'_, K, V>
{
    /// Commit the changes from this write transaction. Readers after this point
    /// will be able to percieve these changes.
    ///
    /// To abort (unstage changes), just do not call this function.
    pub fn commit(self) {
        self.inner.commit();
    }
}

#[cfg(test)]
mod tests {
    use super::BptreeMap;
    use crate::internals::bptree::node::{assert_released, L_CAPACITY};
    // use rand::prelude::*;
    use rand::seq::SliceRandom;
    use std::iter::FromIterator;

    #[test]
    fn test_bptree2_map_basic_write() {
        let bptree: BptreeMap<usize, usize> = BptreeMap::new();
        {
            let mut bpwrite = bptree.write();
            // We should be able to insert.
            bpwrite.insert(0, 0);
            bpwrite.insert(1, 1);
            assert!(bpwrite.get(&0) == Some(&0));
            assert!(bpwrite.get(&1) == Some(&1));
            bpwrite.insert(2, 2);
            bpwrite.commit();
            // println!("commit");
        }
        {
            // Do a clear, but roll it back.
            let mut bpwrite = bptree.write();
            bpwrite.clear();
            // DO NOT commit, this triggers the rollback.
            // println!("post clear");
        }
        {
            let bpwrite = bptree.write();
            assert!(bpwrite.get(&0) == Some(&0));
            assert!(bpwrite.get(&1) == Some(&1));
            // println!("fin write");
        }
        std::mem::drop(bptree);
        assert_released();
    }

    #[test]
    fn test_bptree2_map_cursed_get_mut() {
        let bptree: BptreeMap<usize, usize> = BptreeMap::new();
        {
            let mut w = bptree.write();
            w.insert(0, 0);
            w.commit();
        }
        let r1 = bptree.read();
        {
            let mut w = bptree.write();
            let cursed_zone = w.get_mut(&0).unwrap();
            *cursed_zone = 1;
            // Correctly fails to work as it's a second borrow, which isn't
            // possible once w.remove occurs
            // w.remove(&0);
            // *cursed_zone = 2;
            w.commit();
        }
        let r2 = bptree.read();
        assert!(r1.get(&0) == Some(&0));
        assert!(r2.get(&0) == Some(&1));

        /*
        // Correctly fails to compile. PHEW!
        let fail = {
            let mut w = bptree.write();
            w.get_mut(&0).unwrap()
        };
        */
        std::mem::drop(r1);
        std::mem::drop(r2);
        std::mem::drop(bptree);
        assert_released();
    }

    #[test]
    fn test_bptree2_map_from_iter_1() {
        let ins: Vec<usize> = (0..(L_CAPACITY << 4)).collect();

        let map = BptreeMap::from_iter(ins.into_iter().map(|v| (v, v)));

        {
            let w = map.write();
            assert!(w.verify());
            println!("{:?}", w.tree_density());
        }
        // assert!(w.tree_density() == ((L_CAPACITY << 4), (L_CAPACITY << 4)));
        std::mem::drop(map);
        assert_released();
    }

    #[test]
    fn test_bptree2_map_from_iter_2() {
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (0..(L_CAPACITY << 4)).collect();
        ins.shuffle(&mut rng);

        let map = BptreeMap::from_iter(ins.into_iter().map(|v| (v, v)));

        {
            let w = map.write();
            assert!(w.verify());
            // w.compact_force();
            assert!(w.verify());
            // assert!(w.tree_density() == ((L_CAPACITY << 4), (L_CAPACITY << 4)));
        }

        std::mem::drop(map);
        assert_released();
    }

    fn bptree_map_basic_concurrency(lower: usize, upper: usize) {
        // Create a map
        let map = BptreeMap::new();

        // add values
        {
            let mut w = map.write();
            w.extend((0..lower).map(|v| (v, v)));
            w.commit();
        }

        // read
        let r = map.read();
        assert!(r.len() == lower);
        for i in 0..lower {
            assert!(r.contains_key(&i))
        }

        // Check a second write doesn't interfere
        {
            let mut w = map.write();
            w.extend((lower..upper).map(|v| (v, v)));
            w.commit();
        }

        assert!(r.len() == lower);

        // But a new write can see
        let r2 = map.read();
        assert!(r2.len() == upper);
        for i in 0..upper {
            assert!(r2.contains_key(&i))
        }

        // Now drain the tree, and the reader should be unaffected.
        {
            let mut w = map.write();
            for i in 0..upper {
                assert!(w.remove(&i).is_some())
            }
            w.commit();
        }

        // All consistent!
        assert!(r.len() == lower);
        assert!(r2.len() == upper);
        for i in 0..upper {
            assert!(r2.contains_key(&i))
        }

        let r3 = map.read();
        // println!("{:?}", r3.len());
        assert!(r3.len() == 0);

        std::mem::drop(r);
        std::mem::drop(r2);
        std::mem::drop(r3);

        std::mem::drop(map);
        assert_released();
    }

    #[test]
    fn test_bptree2_map_acb_order() {
        // Need to ensure that txns are dropped in order.

        // Add data, enouugh to cause a split. All data should be *2
        let map = BptreeMap::new();
        // add values
        {
            let mut w = map.write();
            w.extend((0..(L_CAPACITY * 2)).map(|v| (v * 2, v * 2)));
            w.commit();
        }
        let ro_txn_a = map.read();

        // New write, add 1 val
        {
            let mut w = map.write();
            w.insert(1, 1);
            w.commit();
        }

        let ro_txn_b = map.read();
        // ro_txn_b now owns nodes from a

        // New write, update a value
        {
            let mut w = map.write();
            w.insert(1, 10001);
            w.commit();
        }

        let ro_txn_c = map.read();
        // ro_txn_c
        // Drop ro_txn_b
        assert!(ro_txn_b.verify());
        std::mem::drop(ro_txn_b);
        // Are both still valid?
        assert!(ro_txn_a.verify());
        assert!(ro_txn_c.verify());
        // Drop remaining
        std::mem::drop(ro_txn_a);
        std::mem::drop(ro_txn_c);
        std::mem::drop(map);
        assert_released();
    }

    #[test]
    fn test_bptree2_map_weird_txn_behaviour() {
        let map: BptreeMap<usize, usize> = BptreeMap::new();

        let mut wr = map.write();
        let rd = map.read();

        wr.insert(1, 1);
        assert!(rd.get(&1) == None);
        wr.commit();
        assert!(rd.get(&1) == None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_bptree2_map_basic_concurrency_small() {
        bptree_map_basic_concurrency(100, 200)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_bptree2_map_basic_concurrency_large() {
        bptree_map_basic_concurrency(10_000, 20_000)
    }

    #[test]
    fn test_bptree2_map_rangeiter_1() {
        let ins: Vec<usize> = (0..100).collect();

        let map = BptreeMap::from_iter(ins.into_iter().map(|v| (v, v)));

        {
            let w = map.write();
            assert!(w.range(0..100).count() == 100);
            assert!(w.range(25..100).count() == 75);
            assert!(w.range(0..75).count() == 75);
            assert!(w.range(25..75).count() == 50);
        }
        // assert!(w.tree_density() == ((L_CAPACITY << 4), (L_CAPACITY << 4)));
        std::mem::drop(map);
        assert_released();
    }

    /*
    #[test]
    fn test_bptree2_map_write_compact() {
        let mut rng = rand::thread_rng();
        let insa: Vec<usize> = (0..(L_CAPACITY << 4)).collect();

        let map = BptreeMap::from_iter(insa.into_iter().map(|v| (v, v)));

        let mut w = map.write();
        // Created linearly, should not need compact
        assert!(w.compact() == false);
        assert!(w.verify());
        assert!(w.tree_density() == ((L_CAPACITY << 4), (L_CAPACITY << 4)));

        // Even in reverse, we shouldn't need it ...
        let insb: Vec<usize> = (0..(L_CAPACITY << 4)).collect();
        let bmap = BptreeMap::from_iter(insb.into_iter().rev().map(|v| (v, v)));
        let mut bw = bmap.write();
        assert!(bw.compact() == false);
        assert!(bw.verify());
        // Assert the density is "best"
        assert!(bw.tree_density() == ((L_CAPACITY << 4), (L_CAPACITY << 4)));

        // Random however, may.
        let mut insc: Vec<usize> = (0..(L_CAPACITY << 4)).collect();
        insc.shuffle(&mut rng);
        let cmap = BptreeMap::from_iter(insc.into_iter().map(|v| (v, v)));
        let mut cw = cmap.write();
        let (_n, d1) = cw.tree_density();
        cw.compact_force();
        assert!(cw.verify());
        let (_n, d2) = cw.tree_density();
        assert!(d2 <= d1);
    }
    */

    /*
    use std::sync::atomic::{AtomicUsize, Ordering};
    use crossbeam_utils::thread::scope;
    use rand::Rng;
    const MAX_TARGET: usize = 210_000;

    #[test]
    fn test_bptree2_map_thread_stress() {
        let start = time::now();
        let reader_completions = AtomicUsize::new(0);
        // Setup a tree with some initial data.
        let map: BptreeMap<usize, usize> = BptreeMap::from_iter(
            (0..10_000).map(|v| (v, v))
        );
        // now setup the threads.
        scope(|scope| {
            let mref = &map;
            let rref = &reader_completions;

            let _readers: Vec<_> = (0..7)
                .map(|_| {
                    scope.spawn(move || {
                        println!("Started reader ...");
                        let mut rng = rand::thread_rng();
                        let mut proceed = true;
                        while proceed {
                            let m_read = mref.read();
                            proceed = ! m_read.contains_key(&MAX_TARGET);
                            // Get a random number.
                            // Add 10_000 * random
                            // Remove 10_000 * random
                            let v1 = rng.gen_range(1, 18) * 10_000;
                            let r1 = v1 + 10_000;
                            for i in v1..r1 {
                                m_read.get(&i);
                            }
                            assert!(m_read.verify());
                            rref.fetch_add(1, Ordering::Relaxed);
                        }
                        println!("Closing reader ...");
                    })
                })
                .collect();

            let _writers: Vec<_> = (0..3)
                .map(|_| {
                    scope.spawn(move || {
                        println!("Started writer ...");
                        let mut rng = rand::thread_rng();
                        let mut proceed = true;
                        while proceed {
                            let mut m_write = mref.write();
                            proceed = ! m_write.contains_key(&MAX_TARGET);
                            // Get a random number.
                            // Add 10_000 * random
                            // Remove 10_000 * random
                            let v1 = rng.gen_range(1, 18) * 10_000;
                            let r1 = v1 + 10_000;
                            let v2 = rng.gen_range(1, 19) * 10_000;
                            let r2 = v2 + 10_000;

                            for i in v1..r1 {
                                m_write.insert(i, i);
                            }
                            for i in v2..r2 {
                                m_write.remove(&i);
                            }
                            m_write.commit();
                        }
                        println!("Closing writer ...");
                    })
                })
                .collect();

            let _complete = scope.spawn(move || {
                let mut last_value = 200_000;
                while last_value < MAX_TARGET {
                    let mut m_write = mref.write();
                    last_value += 1;
                    if last_value % 1000 == 0 {
                        println!("{:?}", last_value);
                    }
                    m_write.insert(last_value, last_value);
                    assert!(m_write.verify());
                    m_write.commit();
                }
            });

        });
        let end = time::now();
        print!("BptreeMap MT create :{} reader completions :{}", end - start, reader_completions.load(Ordering::Relaxed));
        // Done!
    }

    #[test]
    fn test_std_mutex_btreemap_thread_stress() {
        use std::collections::BTreeMap;
        use std::sync::Mutex;

        let start = time::now();
        let reader_completions = AtomicUsize::new(0);
        // Setup a tree with some initial data.
        let map: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::from_iter(
            (0..10_000).map(|v| (v, v))
        ));
        // now setup the threads.
        scope(|scope| {
            let mref = &map;
            let rref = &reader_completions;

            let _readers: Vec<_> = (0..7)
                .map(|_| {
                    scope.spawn(move || {
                        println!("Started reader ...");
                        let mut rng = rand::thread_rng();
                        let mut proceed = true;
                        while proceed {
                            let m_read = mref.lock().unwrap();
                            proceed = ! m_read.contains_key(&MAX_TARGET);
                            // Get a random number.
                            // Add 10_000 * random
                            // Remove 10_000 * random
                            let v1 = rng.gen_range(1, 18) * 10_000;
                            let r1 = v1 + 10_000;
                            for i in v1..r1 {
                                m_read.get(&i);
                            }
                            rref.fetch_add(1, Ordering::Relaxed);
                        }
                        println!("Closing reader ...");
                    })
                })
                .collect();

            let _writers: Vec<_> = (0..3)
                .map(|_| {
                    scope.spawn(move || {
                        println!("Started writer ...");
                        let mut rng = rand::thread_rng();
                        let mut proceed = true;
                        while proceed {
                            let mut m_write = mref.lock().unwrap();
                            proceed = ! m_write.contains_key(&MAX_TARGET);
                            // Get a random number.
                            // Add 10_000 * random
                            // Remove 10_000 * random
                            let v1 = rng.gen_range(1, 18) * 10_000;
                            let r1 = v1 + 10_000;
                            let v2 = rng.gen_range(1, 19) * 10_000;
                            let r2 = v2 + 10_000;

                            for i in v1..r1 {
                                m_write.insert(i, i);
                            }
                            for i in v2..r2 {
                                m_write.remove(&i);
                            }
                        }
                        println!("Closing writer ...");
                    })
                })
                .collect();

            let _complete = scope.spawn(move || {
                let mut last_value = 200_000;
                while last_value < MAX_TARGET {
                    let mut m_write = mref.lock().unwrap();
                    last_value += 1;
                    if last_value % 1000 == 0 {
                        println!("{:?}", last_value);
                    }
                    m_write.insert(last_value, last_value);
                }
            });

        });
        let end = time::now();
        print!("Mutex<BTreeMap> MT create :{} reader completions :{}", end - start, reader_completions.load(Ordering::Relaxed));
        // Done!
    }

    #[test]
    fn test_std_rwlock_btreemap_thread_stress() {
        use std::collections::BTreeMap;
        use std::sync::RwLock;

        let start = time::now();
        let reader_completions = AtomicUsize::new(0);
        // Setup a tree with some initial data.
        let map: RwLock<BTreeMap<usize, usize>> = RwLock::new(BTreeMap::from_iter(
            (0..10_000).map(|v| (v, v))
        ));
        // now setup the threads.
        scope(|scope| {
            let mref = &map;
            let rref = &reader_completions;

            let _readers: Vec<_> = (0..7)
                .map(|_| {
                    scope.spawn(move || {
                        println!("Started reader ...");
                        let mut rng = rand::thread_rng();
                        let mut proceed = true;
                        while proceed {
                            let m_read = mref.read().unwrap();
                            proceed = ! m_read.contains_key(&MAX_TARGET);
                            // Get a random number.
                            // Add 10_000 * random
                            // Remove 10_000 * random
                            let v1 = rng.gen_range(1, 18) * 10_000;
                            let r1 = v1 + 10_000;
                            for i in v1..r1 {
                                m_read.get(&i);
                            }
                            rref.fetch_add(1, Ordering::Relaxed);
                        }
                        println!("Closing reader ...");
                    })
                })
                .collect();

            let _writers: Vec<_> = (0..3)
                .map(|_| {
                    scope.spawn(move || {
                        println!("Started writer ...");
                        let mut rng = rand::thread_rng();
                        let mut proceed = true;
                        while proceed {
                            let mut m_write = mref.write().unwrap();
                            proceed = ! m_write.contains_key(&MAX_TARGET);
                            // Get a random number.
                            // Add 10_000 * random
                            // Remove 10_000 * random
                            let v1 = rng.gen_range(1, 18) * 10_000;
                            let r1 = v1 + 10_000;
                            let v2 = rng.gen_range(1, 19) * 10_000;
                            let r2 = v2 + 10_000;

                            for i in v1..r1 {
                                m_write.insert(i, i);
                            }
                            for i in v2..r2 {
                                m_write.remove(&i);
                            }
                        }
                        println!("Closing writer ...");
                    })
                })
                .collect();

            let _complete = scope.spawn(move || {
                let mut last_value = 200_000;
                while last_value < MAX_TARGET {
                    let mut m_write = mref.write().unwrap();
                    last_value += 1;
                    if last_value % 1000 == 0 {
                        println!("{:?}", last_value);
                    }
                    m_write.insert(last_value, last_value);
                }
            });

        });
        let end = time::now();
        print!("RwLock<BTreeMap> MT create :{} reader completions :{}", end - start, reader_completions.load(Ordering::Relaxed));
        // Done!
    }
    */
}
