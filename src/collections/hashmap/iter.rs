use crate::collections::bptree::iter::Iter as BIter;
use std::fmt::Debug;
use std::hash::Hash;

use super::map::vinner;

pub struct Iter<'a, K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    iter: BIter<'a, u64, vinner<K, V>>,
    cur_va: Option<&'a vinner<K, V>>,
    next_idx: usize,
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> Iter<'a, K, V> {
    pub(crate) fn new(mut biter: BIter<'a, u64, vinner<K, V>>) -> Self {
        let next = biter.next().map(|(_k, v)| v);
        Iter {
            iter: biter,
            cur_va: next,
            next_idx: 0,
        }
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    /// Yield the next key value reference, or `None` if exhausted.
    fn next(&mut self) -> Option<Self::Item> {
        match self.cur_va {
            Some(va) => {
                let maybe = va.get(self.next_idx);
                if maybe.is_some() {
                    self.next_idx += 1;
                    maybe.map(|(k, v)| (k, v))
                } else {
                    self.next_idx = 0;
                    self.cur_va = self.iter.next().map(|(_k, v)| v);
                    self.next()
                }
            }
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub struct KeyIter<'a, K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    iter: Iter<'a, K, V>,
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> KeyIter<'a, K, V> {
    pub(crate) fn new(biter: BIter<'a, u64, vinner<K, V>>) -> Self {
        KeyIter {
            iter: Iter::new(biter),
        }
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> Iterator for KeyIter<'a, K, V> {
    type Item = &'a K;

    /// Yield the next key value reference, or `None` if exhausted.
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub struct ValueIter<'a, K, V>
where
    K: Hash + Eq + Clone + Debug,
    V: Clone,
{
    iter: Iter<'a, K, V>,
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> ValueIter<'a, K, V> {
    pub(crate) fn new(biter: BIter<'a, u64, vinner<K, V>>) -> Self {
        ValueIter {
            iter: Iter::new(biter),
        }
    }
}

impl<'a, K: Hash + Eq + Clone + Debug, V: Clone> Iterator for ValueIter<'a, K, V> {
    type Item = &'a V;

    /// Yield the next key value reference, or `None` if exhausted.
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
