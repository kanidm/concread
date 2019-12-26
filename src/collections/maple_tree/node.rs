use super::constants::{CAPACITY, R_CAPACITY};
use super::denseLeaf::DenseLeaf;
use super::rangeBranch::RangeBranch;
use super::rangeLeaf::RangeLeaf;
use super::sparseLeaf::SparseLeaf;
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
