use super::constants::{CAPACITY, R_CAPACITY};
use super::dense_leaf::DenseLeaf;
use super::range_branch::RangeBranch;
use super::range_leaf::RangeLeaf;
use super::sparse_leaf::SparseLeaf;
use std::fmt::Debug;

#[derive(Debug)]
enum NodeTag<K, V> {
    SL(SparseLeaf<K, V>),
    DL(DenseLeaf<V>),
    RL(RangeLeaf<K, V>),
    RB(RangeBranch<K, V>),
}

#[derive(Debug)]
pub struct Node<K, V> {
    tid: u64,
    // checksum: u32,
    inner: NodeTag<K, V>,
}
