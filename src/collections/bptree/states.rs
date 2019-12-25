use super::node::{ABNode, Node};

#[derive(Debug)]
pub(crate) enum BLInsertState<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    Ok(Option<V>),
    // Return's a K,V that should be put into a new leaf.
    Split(K, V),
}

#[derive(Debug)]
pub(crate) enum BLRemoveState<V>
where
    V: Clone,
{
    Ok(Option<V>),
    // Indicate that we found the associated value, but this
    // removal means we no longer exist so should be removed.
    Shrink(Option<V>),
}

#[derive(Debug)]
pub(crate) enum BRInsertState<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    Ok,
    // Returns two nodes (l, r) that need to be handled.
    Split(ABNode<K, V>, ABNode<K, V>),
}

#[derive(Debug)]
pub(crate) enum BNClone<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    // Not needed
    Ok,
    Clone(Box<Node<K, V>>),
}

#[derive(Debug)]
pub(crate) enum CRInsertState<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    // We did not need to clone, here is the result.
    NoClone(Option<V>),
    // We had to clone the referenced node provided.
    Clone(Option<V>, ABNode<K, V>),
    // We had to split, but did not need a clone.
    // REMEMBER: In all split cases it means the key MUST NOT have
    // previously existed, so it implies return none to the
    // caller.
    Split(ABNode<K, V>),
    // We had to clone and split.
    CloneSplit(ABNode<K, V>, ABNode<K, V>),
}
