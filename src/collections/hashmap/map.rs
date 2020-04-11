use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::fmt::Debug;
use std::iter::FromIterator;
use std::borrow::Borrow;

use crate::collections::bptree::{BptreeMap, BptreeMapReadTxn, BptreeMapWriteTxn, BptreeMapReadSnapshot};
// use super::iter::*;

use smallvec::SmallVec;

const DEFAULT_STACK_ALLOC: usize = 1;

type vinner<K, V> = SmallVec<[(K, V); DEFAULT_STACK_ALLOC]>;

pub struct HashMap<K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    map: BptreeMap<u64, vinner<K, V>>
}

pub struct HashMapReadTxn<K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    map: BptreeMapReadTxn<u64, vinner<K, V>>
}


pub struct HashMapWriteTxn<'a, K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    map: BptreeMapWriteTxn<'a, u64, vinner<K, V>>
}


pub struct HashMapReadSnapshot<'a, K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    map: BptreeMapReadSnapshot<'a, u64, vinner<K, V>>
}

impl<K: Hash + Eq + Clone + Debug, V: Clone> HashMap<K, V> {
    pub fn new() -> Self {
        HashMap {
            map: BptreeMap::new()
        }
    }

    pub fn read(&self) -> HashMapReadTxn<K, V> {
        unimplemented!();
    }

    pub fn write(&self) -> HashMapWriteTxn<K, V> {
        unimplemented!();
    }
}

impl<K: Hash + Eq + Clone + Debug, V: Clone> FromIterator<(K, V)> for HashMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        unimplemented!();
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> HashMapWriteTxn<'a, K, V> {
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        unimplemented!();
    }

    pub fn contains_key<'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        unimplemented!();
    }

    pub fn len(&self) -> usize {
        unimplemented!();
    }

    pub fn is_empty(&self) -> bool {
        unimplemented!();
    }

    /*
    pub fn iter(&self) -> Iter<K, V> {
        unimplemented!();
    }

    pub fn values(&self) -> ValueIter<K, V> {
        unimplemented!();
    }

    pub fn keys(&self) -> KeyIter<K, V> {
        unimplemented!();
    }
    */

    pub fn clear(&mut self) {
        unimplemented!();
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        unimplemented!();
    }

    pub fn remove(&mut self, k: &K) -> Option<V> {
        unimplemented!();
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        unimplemented!();
    }

    #[cfg(test)]
    pub(crate) fn verify(&self) -> bool {
        unimplemented!();
    }

    pub fn to_snapshot(&'a self) -> HashMapReadSnapshot<K, V> {
        unimplemented!();
    }

    pub fn commit(self) {
        unimplemented!();
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> Extend<(K, V)> for HashMapWriteTxn<'a, K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        unimplemented!();
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> HashMapReadTxn<K, V> {
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        unimplemented!();
    }

    pub fn contains_key<'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        unimplemented!();
    }

    pub fn len(&self) -> usize {
        unimplemented!();
    }

    pub fn is_empty(&self) -> bool {
        unimplemented!();
    }

    /*
    pub fn iter(&self) -> Iter<K, V> {
        unimplemented!();
    }

    pub fn values(&self) -> ValueIter<K, V> {
        unimplemented!();
    }

    pub fn keys(&self) -> KeyIter<K, V> {
        unimplemented!();
    }
    */

    #[cfg(test)]
    pub(crate) fn verify(&self) -> bool {
        unimplemented!();
    }

    pub fn to_snapshot(&'a self) -> HashMapReadSnapshot<'a, K, V> {
        unimplemented!();
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> HashMapReadSnapshot<'a, K, V> {
    pub fn get<'b, Q: ?Sized>(&'a self, k: &'b Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        unimplemented!();
    }

    pub fn contains_key<'b, Q: ?Sized>(&'a self, k: &'b Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        unimplemented!();
    }

    pub fn len(&self) -> usize {
        unimplemented!();
    }

    pub fn is_empty(&self) -> bool {
        unimplemented!();
    }

    /*
    pub fn iter(&self) -> Iter<K, V> {
        unimplemented!();
    }

    pub fn values(&self) -> ValueIter<K, V> {
        unimplemented!();
    }

    pub fn keys(&self) -> KeyIter<K, V> {
        unimplemented!();
    }
    */
}


#[cfg(test)]
mod tests {
    use super::HashMap;

    #[test]
    fn test_hashmap_basic_write() {
        let bptree: HashMap<usize, usize> = HashMap::new();
    }
}


