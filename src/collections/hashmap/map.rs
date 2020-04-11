use std::borrow::Borrow;
use std::collections::hash_map::DefaultHasher;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::mem;

use super::iter::*;
use crate::collections::bptree::{
    BptreeMap, BptreeMapReadSnapshot, BptreeMapReadTxn, BptreeMapWriteTxn,
};

use smallvec::SmallVec;

const DEFAULT_STACK_ALLOC: usize = 1;

pub(crate) type vinner<K, V> = SmallVec<[(K, V); DEFAULT_STACK_ALLOC]>;

macro_rules! hash_key {
    ($k:expr) => {{
        let mut hasher = DefaultHasher::new();
        $k.hash(&mut hasher);
        hasher.finish()
    }};
}

pub struct HashMap<K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    map: BptreeMap<u64, vinner<K, V>>,
}

pub struct HashMapReadTxn<K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    map: BptreeMapReadTxn<u64, vinner<K, V>>,
}

pub struct HashMapWriteTxn<'a, K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    map: BptreeMapWriteTxn<'a, u64, vinner<K, V>>,
}

pub struct HashMapReadSnapshot<'a, K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    map: BptreeMapReadSnapshot<'a, u64, vinner<K, V>>,
}

impl<K: Hash + Eq + Clone + Debug, V: Clone> HashMap<K, V> {
    pub fn new() -> Self {
        HashMap {
            map: BptreeMap::new(),
        }
    }

    pub fn read(&self) -> HashMapReadTxn<K, V> {
        HashMapReadTxn {
            map: self.map.read(),
        }
    }

    pub fn write(&self) -> HashMapWriteTxn<K, V> {
        HashMapWriteTxn {
            map: self.map.write(),
        }
    }
}

impl<K: Hash + Eq + Clone + Debug, V: Clone> FromIterator<(K, V)> for HashMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let hmap = HashMap::new();
        let mut hmap_write = hmap.write();
        hmap_write.extend(iter);
        hmap_write.commit();
        hmap
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> Extend<(K, V)> for HashMapWriteTxn<'a, K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        iter.into_iter().for_each(|(k, v)| {
            let _ = self.insert(k, v);
        });
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> HashMapWriteTxn<'a, K, V> {
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k_hash = hash_key!(k);
        self.map.get(&k_hash).and_then(|va| {
            va.iter()
                .filter(|(ki, _vi)| k.eq(ki.borrow()))
                .take(1)
                .map(|(_kr, vr)| vr)
                .next()
        })
    }

    pub fn contains_key<'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(k).is_some()
    }

    /*
    pub fn len(&self) -> usize {
        unimplemented!();
    }

    pub fn is_empty(&self) -> bool {
        unimplemented!();
    }
    */

    pub fn iter(&self) -> Iter<K, V> {
        Iter::new(self.map.iter())
    }

    pub fn values(&self) -> ValueIter<K, V> {
        ValueIter::new(self.map.iter())
    }

    pub fn keys(&self) -> KeyIter<K, V> {
        KeyIter::new(self.map.iter())
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        // Hash the key.
        let k_hash = hash_key!(k);
        // Does it exist?
        match self.map.get_mut(&k_hash) {
            Some(mut va) => {
                // Does our k exist in va?
                for (ki, vi) in va.as_mut_slice().iter_mut() {
                    if *ki == k {
                        // swap v and vi
                        let mut ov = v;
                        mem::swap(&mut ov, vi);
                        // Return the previous value.
                        return Some(ov);
                    }
                }
                // If we get here, it wasn't present.
                va.push((k, v));
                None
            }
            None => {
                let mut va = SmallVec::new();
                va.push((k, v));
                self.map.insert(k_hash, va);
                None
            }
        }
    }

    pub fn remove(&mut self, k: &K) -> Option<V> {
        let k_hash = hash_key!(k);
        match self.map.get_mut(&k_hash) {
            Some(mut va) => {
                let mut idx = 0;
                for (ki, _vi) in va.iter() {
                    if k.eq(ki.borrow()) {
                        break;
                    }
                    idx += 1;
                }
                if idx > va.len() {
                    None
                } else {
                    let (_ki, vi) = va.remove(idx);
                    Some(vi)
                }
            }
            None => None,
        }
    }

    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let k_hash = hash_key!(k);
        self.map.get_mut(&k_hash).and_then(|va| {
            va.iter_mut()
                .filter(|(ki, _vi)| k.eq(ki.borrow()))
                .take(1)
                .map(|(_kr, vr)| vr)
                .next()
        })
    }

    pub fn to_snapshot(&'a self) -> HashMapReadSnapshot<K, V> {
        HashMapReadSnapshot {
            map: self.map.to_snapshot(),
        }
    }

    pub fn commit(self) {
        self.map.commit()
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> HashMapReadTxn<K, V> {
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k_hash = hash_key!(k);
        self.map.get(&k_hash).and_then(|va| {
            va.iter()
                .filter(|(ki, _vi)| k.eq(ki.borrow()))
                .take(1)
                .map(|(_kr, vr)| {
                    // This is some lifetime stripping to deal with the fact that
                    // this ref IS valid, but it's bound to k_hash, not to &self
                    // so we ... cheat.
                    vr as *const V
                })
                // ThIs Is ThE GuD RuSt
                .map(|v| unsafe { &*v as &'a V })
                .next()
        })
    }

    pub fn contains_key<'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(k).is_some()
    }

    /*
    pub fn len(&self) -> usize {
        unimplemented!();
    }

    pub fn is_empty(&self) -> bool {
        unimplemented!();
    }
    */

    pub fn iter(&self) -> Iter<K, V> {
        Iter::new(self.map.iter())
    }

    pub fn values(&self) -> ValueIter<K, V> {
        ValueIter::new(self.map.iter())
    }

    pub fn keys(&self) -> KeyIter<K, V> {
        KeyIter::new(self.map.iter())
    }

    pub fn to_snapshot(&'a self) -> HashMapReadSnapshot<'a, K, V> {
        HashMapReadSnapshot {
            map: self.map.to_snapshot(),
        }
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> HashMapReadSnapshot<'a, K, V> {
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k_hash = hash_key!(k);
        self.map.get(&k_hash).and_then(|va| {
            va.iter()
                .filter(|(ki, _vi)| k.eq(ki.borrow()))
                .take(1)
                .map(|(_kr, vr)| vr as *const V)
                .map(|v| unsafe { &*v as &'a V })
                .next()
        })
    }

    pub fn contains_key<'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(k).is_some()
    }

    /*
    pub fn len(&self) -> usize {
        unimplemented!();
    }

    pub fn is_empty(&self) -> bool {
        unimplemented!();
    }
    */

    pub fn iter(&self) -> Iter<K, V> {
        Iter::new(self.map.iter())
    }

    pub fn values(&self) -> ValueIter<K, V> {
        ValueIter::new(self.map.iter())
    }

    pub fn keys(&self) -> KeyIter<K, V> {
        KeyIter::new(self.map.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::HashMap;

    #[test]
    fn test_hashmap_basic_write() {
        let hmap: HashMap<usize, usize> = HashMap::new();
        let mut hmap_write = hmap.write();

        hmap_write.insert(10, 10);
        hmap_write.insert(15, 15);

        assert!(hmap_write.contains_key(&10));
        assert!(hmap_write.contains_key(&15));
        assert!(!hmap_write.contains_key(&20));

        assert!(hmap_write.get(&10) == Some(&10));
        {
            let mut v = hmap_write.get_mut(&10).unwrap();
            *v = 11;
        }
        assert!(hmap_write.get(&10) == Some(&11));

        assert!(hmap_write.remove(&10).is_some());
        assert!(!hmap_write.contains_key(&10));
        assert!(hmap_write.contains_key(&15));

        assert!(hmap_write.remove(&30).is_none());

        hmap_write.clear();
        assert!(!hmap_write.contains_key(&10));
        assert!(!hmap_write.contains_key(&15));
        hmap_write.commit();
    }

    #[test]
    fn test_hashmap_basic_read_write() {
        let hmap: HashMap<usize, usize> = HashMap::new();
        let mut hmap_w1 = hmap.write();
        hmap_w1.insert(10, 10);
        hmap_w1.insert(15, 15);
        hmap_w1.commit();

        let hmap_r1 = hmap.read();
        assert!(hmap_r1.contains_key(&10));
        assert!(hmap_r1.contains_key(&15));
        assert!(!hmap_r1.contains_key(&20));

        let mut hmap_w2 = hmap.write();
        hmap_w2.insert(20, 20);
        hmap_w2.commit();

        assert!(hmap_r1.contains_key(&10));
        assert!(hmap_r1.contains_key(&15));
        assert!(!hmap_r1.contains_key(&20));

        let hmap_r2 = hmap.read();
        assert!(hmap_r2.contains_key(&10));
        assert!(hmap_r2.contains_key(&15));
        assert!(hmap_r2.contains_key(&20));
    }

    #[test]
    fn test_hashmap_basic_read_snapshot() {
        let hmap: HashMap<usize, usize> = HashMap::new();
        let mut hmap_w1 = hmap.write();
        hmap_w1.insert(10, 10);
        hmap_w1.insert(15, 15);

        let snap = hmap_w1.to_snapshot();
        assert!(snap.contains_key(&10));
        assert!(snap.contains_key(&15));
        assert!(!snap.contains_key(&20));
    }

    #[test]
    fn test_hashmap_basic_iter() {
        let hmap: HashMap<usize, usize> = HashMap::new();
        let mut hmap_w1 = hmap.write();
        assert!(hmap_w1.iter().count() == 0);

        hmap_w1.insert(10, 10);
        hmap_w1.insert(15, 15);

        assert!(hmap_w1.iter().count() == 2);
    }

    #[test]
    fn test_hashmap_from_iter() {
        let hmap: HashMap<usize, usize> = vec![(10, 10), (15, 15), (20, 20)].into_iter().collect();
        let hmap_r2 = hmap.read();
        assert!(hmap_r2.contains_key(&10));
        assert!(hmap_r2.contains_key(&15));
        assert!(hmap_r2.contains_key(&20));
    }
}
