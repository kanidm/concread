use std::mem::MaybeUninit;
use std::sync::Arc;
use std::ptr;

use super::states::{BLInsertState, BNClone, BLRemoveState};
use super::constants::{BK_CAPACITY, BV_CAPACITY};
use super::leaf::Leaf;

pub(crate) struct Branch<K, V> {
    count: usize,
    key: [MaybeUninit<K>; BK_CAPACITY],
    node: [MaybeUninit<Arc<Box<Node<K, V>>>>; BV_CAPACITY],
}

pub(crate) enum T<K, V> {
    B(Branch<K, V>),
    L(Leaf<K, V>),
}

pub(crate) struct Node<K, V> {
    txid: usize,
    inner: T<K, V>
}

impl<K: Clone + PartialEq + PartialOrd, V: Clone> Node<K, V> {
    fn new_tree_root(txid: usize) -> Self {
        Node {
            txid: txid,
            inner: T::L(Leaf::new())
        }
    }

    fn req_clone(&self, txid: usize) -> BNClone<K, V> {
        // Do we need to clone this node before we work on it?
        if txid == self.txid {
            BNClone::Ok
        } else {
            BNClone::Clone(
                Box::new(Node {
                    txid: txid,
                    inner: match &self.inner {
                        T::L(leaf) => {
                            T::L(leaf.clone())
                        }
                        T::B(branch) => {
                            T::B(branch.clone())
                        }
                    }
                })
            )
        }
    }

    #[cfg(test)]
    fn verify(&self) -> bool {
        false
    }

    fn len(&self) -> usize {
        match &self.inner {
            T::L(leaf) => leaf.len(),
            T::B(branch) => branch.len(),
        }
    }
}

impl<K, V> Branch<K, V> {
    pub(crate) fn len(&self) -> usize {
        self.count
    }
}

impl<K, V> Clone for Branch<K, V> {
    fn clone(&self) -> Self {
        unimplemented!();
    }
}

impl<K, V> Drop for Branch<K, V> {
    fn drop(&mut self) {
        // Due to the use of maybe uninit we have to drop any contained values.
        for idx in 0..self.count {
            unsafe {
                ptr::drop_in_place(self.key[idx].as_mut_ptr());
            }
        }
        // Remember, a branch ALWAYS has two nodes per key, which means
        // it's N+1,so we have to increase this to ensure we drop them
        // all.
        for idx in 0..(self.count + 1) {
            unsafe {
                ptr::drop_in_place(self.node[idx].as_mut_ptr());
            }
        }
        println!("branch dropped {:?} + 1", self.count);
    }
}

#[cfg(test)]
mod tests {
    use super::Node;
    use super::super::states::BNClone;

    // check clone txid behaviour
    #[test]
    fn test_bptree_node_req_clone() {
        // Make a new node.
        let nroot: Node<usize, usize> = Node::new_tree_root(0);
        // Req to clone it.
        match nroot.req_clone(0) {
            BNClone::Ok => {},
            BNClone::Clone(_) => panic!(),
        };
        // Now do one where we do clone.
        let nnode = match nroot.req_clone(1) {
            BNClone::Ok => panic!(),
            BNClone::Clone(nnode) => nnode,
        };

        assert!(nnode.txid == 1);
        assert!(nnode.len() == nroot.len());
    }
}



