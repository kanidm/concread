//! Async `BptreeMap` - See the documentation for the sync `BptreeMap`

use crate::internals::lincowcell_async::{LinCowCell, LinCowCellReadTxn, LinCowCellWriteTxn};

include!("impl.rs");

impl<K: Clone + Ord + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static>
    BptreeMap<K, V>
{
    /// Initiate a read transaction for the tree, concurrent to any
    /// other readers or writers.
    pub async fn read<'x>(&'x self) -> BptreeMapReadTxn<'x, K, V> {
        let inner = self.inner.read().await;
        BptreeMapReadTxn { inner }
    }

    /// Initiate a write transaction for the tree, exclusive to this
    /// writer, and concurrently to all existing reads.
    pub async fn write<'x>(&'x self) -> BptreeMapWriteTxn<'x, K, V> {
        let inner = self.inner.write().await;
        BptreeMapWriteTxn { inner }
    }
}

impl<'a, K: Clone + Ord + Debug + Sync + Send + 'static, V: Clone + Sync + Send + 'static>
    BptreeMapWriteTxn<'a, K, V>
{
    /// Commit the changes from this write transaction. Readers after this point
    /// will be able to percieve these changes.
    ///
    /// To abort (unstage changes), just do not call this function.
    pub async fn commit(self) {
        self.inner.commit().await;
    }
}

#[cfg(test)]
mod tests {
    use super::BptreeMap;
    use crate::internals::bptree::node::{assert_released, L_CAPACITY};
    // use rand::prelude::*;
    use rand::seq::SliceRandom;
    use std::iter::FromIterator;

    #[tokio::test]
    async fn test_bptree2_map_basic_write() {
        let bptree: BptreeMap<usize, usize> = BptreeMap::new();
        {
            let mut bpwrite = bptree.write().await;
            // We should be able to insert.
            bpwrite.insert(0, 0);
            bpwrite.insert(1, 1);
            assert!(bpwrite.get(&0) == Some(&0));
            assert!(bpwrite.get(&1) == Some(&1));
            bpwrite.insert(2, 2);
            bpwrite.commit().await;
            // println!("commit");
        }
        {
            // Do a clear, but roll it back.
            let mut bpwrite = bptree.write().await;
            bpwrite.clear();
            // DO NOT commit, this triggers the rollback.
            // println!("post clear");
        }
        {
            let bpwrite = bptree.write().await;
            assert!(bpwrite.get(&0) == Some(&0));
            assert!(bpwrite.get(&1) == Some(&1));
            // println!("fin write");
        }
        std::mem::drop(bptree);
        assert_released();
    }

    #[tokio::test]
    async fn test_bptree2_map_cursed_get_mut() {
        let bptree: BptreeMap<usize, usize> = BptreeMap::new();
        {
            let mut w = bptree.write().await;
            w.insert(0, 0);
            w.commit().await;
        }
        let r1 = bptree.read().await;
        {
            let mut w = bptree.write().await;
            let cursed_zone = w.get_mut(&0).unwrap();
            *cursed_zone = 1;
            // Correctly fails to work as it's a second borrow, which isn't
            // possible once w.remove occurs
            // w.remove(&0);
            // *cursed_zone = 2;
            w.commit().await;
        }
        let r2 = bptree.read().await;
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

    #[tokio::test]
    async fn test_bptree2_map_from_iter_1() {
        let ins: Vec<usize> = (0..(L_CAPACITY << 4)).collect();

        let map = BptreeMap::from_iter(ins.into_iter().map(|v| (v, v)));

        {
            let w = map.write().await;
            assert!(w.verify());
            println!("{:?}", w.tree_density());
        }
        // assert!(w.tree_density() == ((L_CAPACITY << 4), (L_CAPACITY << 4)));
        std::mem::drop(map);
        assert_released();
    }

    #[tokio::test]
    async fn test_bptree2_map_from_iter_2() {
        let mut rng = rand::thread_rng();
        let mut ins: Vec<usize> = (0..(L_CAPACITY << 4)).collect();
        ins.shuffle(&mut rng);

        let map = BptreeMap::from_iter(ins.into_iter().map(|v| (v, v)));

        {
            let w = map.write().await;
            assert!(w.verify());
            // w.compact_force();
            assert!(w.verify());
            // assert!(w.tree_density() == ((L_CAPACITY << 4), (L_CAPACITY << 4)));
        }

        std::mem::drop(map);
        assert_released();
    }

    async fn bptree_map_basic_concurrency(lower: usize, upper: usize) {
        // Create a map
        let map = BptreeMap::new();

        // add values
        {
            let mut w = map.write().await;
            w.extend((0..lower).map(|v| (v, v)));
            w.commit().await;
        }

        // read
        let r = map.read().await;
        assert!(r.len() == lower);
        for i in 0..lower {
            assert!(r.contains_key(&i))
        }

        // Check a second write doesn't interfere
        {
            let mut w = map.write().await;
            w.extend((lower..upper).map(|v| (v, v)));
            w.commit().await;
        }

        assert!(r.len() == lower);

        // But a new write can see
        let r2 = map.read().await;
        assert!(r2.len() == upper);
        for i in 0..upper {
            assert!(r2.contains_key(&i))
        }

        // Now drain the tree, and the reader should be unaffected.
        {
            let mut w = map.write().await;
            for i in 0..upper {
                assert!(w.remove(&i).is_some())
            }
            w.commit().await;
        }

        // All consistent!
        assert!(r.len() == lower);
        assert!(r2.len() == upper);
        for i in 0..upper {
            assert!(r2.contains_key(&i))
        }

        let r3 = map.read().await;
        // println!("{:?}", r3.len());
        assert!(r3.len() == 0);

        std::mem::drop(r);
        std::mem::drop(r2);
        std::mem::drop(r3);

        std::mem::drop(map);
        assert_released();
    }

    #[tokio::test]
    async fn test_bptree2_map_acb_order() {
        // Need to ensure that txns are dropped in order.

        // Add data, enouugh to cause a split. All data should be *2
        let map = BptreeMap::new();
        // add values
        {
            let mut w = map.write().await;
            w.extend((0..(L_CAPACITY * 2)).map(|v| (v * 2, v * 2)));
            w.commit().await;
        }
        let ro_txn_a = map.read().await;

        // New write, add 1 val
        {
            let mut w = map.write().await;
            w.insert(1, 1);
            w.commit().await;
        }

        let ro_txn_b = map.read().await;
        // ro_txn_b now owns nodes from a

        // New write, update a value
        {
            let mut w = map.write().await;
            w.insert(1, 10001);
            w.commit().await;
        }

        let ro_txn_c = map.read().await;
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

    #[tokio::test]
    async fn test_bptree2_map_weird_txn_behaviour() {
        let map: BptreeMap<usize, usize> = BptreeMap::new();

        let mut wr = map.write().await;
        let rd = map.read().await;

        wr.insert(1, 1);
        assert!(rd.get(&1) == None);
        wr.commit().await;
        assert!(rd.get(&1) == None);
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_bptree2_map_basic_concurrency_small() {
        bptree_map_basic_concurrency(100, 200).await
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_bptree2_map_basic_concurrency_large() {
        bptree_map_basic_concurrency(10_000, 20_000).await
    }
}
