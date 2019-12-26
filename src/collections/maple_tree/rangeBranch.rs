use super::utils::M;
use std::fmt::Debug;
use std::mem;
use super::constants::{CAPACITY, R_CAPACITY};
use super::node::Node; 

#[derive(Debug)]
pub struct RangeBranch<K, V> {
    // Implied Pivots
    // Cap - 2
    pivot: [M<K>; R_CAPACITY],
    links: [M<*mut Node<K, V>>; CAPACITY],
}
