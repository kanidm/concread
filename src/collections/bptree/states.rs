use super::leaf::Leaf;
use super::node::{ABNode, Node};

#[derive(Debug)]
pub(crate) enum BLInsertState<K, V> {
    Ok(Option<V>),
    // Return's a K,V that should be put into a new leaf.
    Split(K, V),
}

#[derive(Debug)]
pub(crate) enum BLRemoveState<V> {
    Ok(Option<V>),
    // Indicate that we found the associated value, but this
    // removal means we no longer exist so should be removed.
    Shrink(Option<V>),
}

#[derive(Debug)]
pub(crate) enum BRInsertState<K, V> {
    Ok,
    // Returns two nodes (l, r) that need to be handled.
    Split(ABNode<K, V>, ABNode<K, V>),
}

#[derive(Debug)]
pub(crate) enum BNClone<K, V> {
    // Not needed
    Ok,
    Clone(Box<Node<K, V>>),
}
