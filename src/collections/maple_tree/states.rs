use std::fmt::Debug;
use super::rangeLeaf::InternalPivots;
use super::utils::M;

#[derive(Debug, PartialEq)]
pub enum RangeInsertState<K, V> {
    Ok(Option<V>),
    Err(K, K, M<V>, InternalPivots), // lowerKey, upperKey, value, InternalPivots for insert
}

