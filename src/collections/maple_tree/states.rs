use super::rangeLeaf::InternalPivots;
use super::utils::M;
use std::fmt::Debug;

#[derive(Debug, PartialEq)]
pub enum RangeInsertState<K, V> {
    Ok(Option<V>),
    Err(K, K, M<V>, InternalPivots), // lowerKey, upperKey, value, InternalPivots for insert
}
