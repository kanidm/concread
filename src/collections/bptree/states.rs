use super::node::ABNode;
use std::fmt::{self, Debug, Error};

#[derive(Debug)]
pub(crate) enum BLInsertState<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    Ok(Option<V>),
    // Return's a K,V that should be put into a new leaf.
    // Split(K, V),
    Split(ABNode<K, V>),
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
    K: Ord + Clone + Debug,
    V: Clone,
{
    Ok,
    // Returns two nodes (l, r) that need to be handled.
    Split(ABNode<K, V>, ABNode<K, V>),
}

#[derive(Debug)]
pub(crate) enum BRShrinkState {
    Balanced,
    Merge,
    Shrink,
}

#[derive(Debug)]
pub(crate) enum BLPruneState {
    Ok,
    Prune,
}

#[derive(Debug)]
pub(crate) enum BRTrimState<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    Complete,
    Promote(ABNode<K, V>),
}

pub(crate) enum CRTrimState<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    Complete,
    Clone(ABNode<K, V>),
    Promote(ABNode<K, V>),
}

impl<K: Ord + Clone + Debug, V: Clone> Debug for CRTrimState<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), Error> {
        match self {
            CRTrimState::Complete => write!(f, "CRTrimState::Complete"),
            CRTrimState::Clone(_) => write!(f, "CRTrimState::Clone"),
            CRTrimState::Promote(_) => write!(f, "CRTrimState::Promote"),
        }
    }
}

#[derive(Debug)]
pub(crate) enum CRInsertState<K, V>
where
    K: Ord + Clone + Debug,
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

#[derive(Debug)]
pub(crate) enum CRCloneState<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    Clone(ABNode<K, V>),
    NoClone,
}

#[derive(Debug)]
pub(crate) enum CRRemoveState<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    // We did not need to clone, here is the result.
    NoClone(Option<V>),
    // We had to clone the referenced node provided.
    Clone(Option<V>, ABNode<K, V>),
    //
    Shrink(Option<V>),
    //
    CloneShrink(Option<V>, ABNode<K, V>),
}

pub(crate) enum CRPruneState<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    // No action needed, node was not cloned.
    OkNoClone,
    // No action, node was cloned.
    OkClone(ABNode<K, V>),
    // The target node was pruned, so we don't care if it cloned or not as
    // we'll be removing it.
    Prune,
    ClonePrune(ABNode<K, V>),
}

impl<K: Ord + Clone + Debug, V: Clone> Debug for CRPruneState<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), Error> {
        match self {
            CRPruneState::OkNoClone => write!(f, "CRPruneState::OkNoClone"),
            CRPruneState::OkClone(_) => write!(f, "CRPruneState::OkClone"),
            CRPruneState::Prune => write!(f, "CRPruneState::Prune"),
            CRPruneState::ClonePrune(_) => write!(f, "CRPruneState::ClonePrune"),
        }
    }
}
