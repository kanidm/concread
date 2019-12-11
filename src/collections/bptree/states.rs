
use super::leaf::Leaf;
use super::node::Node;


pub(crate) enum BLInsertState<K, V> {
    Ok(Option<V>),
    Split(Option<V>, Leaf<K, V>),
}

pub(crate) enum BLRemoveState<V> {
    Ok(Option<V>),
    // Do we need to return the key we are removing?
    Shrink(Option<V>),
}

pub(crate) enum BNClone<K, V> {
    // Not needed
    Ok,
    Clone(Box<Node<K, V>>)
}

