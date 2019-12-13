use super::leaf::Leaf;
use super::node::Node;

#[derive(Debug)]
pub(crate) enum BLInsertState<K, V> {
    Ok(Option<V>),
    // Return's a K,V that should be put into a new leaf.
    Split(K, V),
}

pub(crate) enum BLRemoveState<V> {
    Ok(Option<V>),
    // Indicate that we found the associated value, but this
    // removal means we no longer exist so should be removed.
    Shrink(Option<V>),
}

pub(crate) enum BNClone<K, V> {
    // Not needed
    Ok,
    Clone(Box<Node<K, V>>),
}
