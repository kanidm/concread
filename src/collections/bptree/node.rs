use std::fmt::{self, Debug, Error};
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

use super::constants::{BK_CAPACITY, BV_CAPACITY};
use super::leaf::Leaf;
use super::states::{BLInsertState, BLRemoveState, BNClone};

use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
static NODE_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub(crate) struct Branch<K, V> {
    count: usize,
    key: [MaybeUninit<K>; BK_CAPACITY],
    node: [MaybeUninit<Arc<Box<Node<K, V>>>>; BV_CAPACITY],
}

#[derive(Debug)]
pub(crate) enum T<K, V> {
    B(Branch<K, V>),
    L(Leaf<K, V>),
}

#[derive(Debug)]
pub(crate) struct Node<K, V> {
    #[cfg(test)]
    nid: usize,
    txid: usize,
    inner: T<K, V>,
}

type ABNode<K, V> = Arc<Box<Node<K, V>>>;

impl<K: Clone + PartialEq + PartialOrd, V: Clone> Node<K, V> {
    fn new_leaf(txid: usize) -> Self {
        Node {
            #[cfg(test)]
            nid: NODE_COUNTER.fetch_add(1, Ordering::AcqRel),
            txid: txid,
            inner: T::L(Leaf::new()),
        }
    }

    fn req_clone(&self, txid: usize) -> BNClone<K, V> {
        // Do we need to clone this node before we work on it?
        if txid == self.txid {
            BNClone::Ok
        } else {
            BNClone::Clone(Box::new(Node {
                #[cfg(test)]
                nid: NODE_COUNTER.fetch_add(1, Ordering::AcqRel),
                txid: txid,
                inner: match &self.inner {
                    T::L(leaf) => T::L(leaf.clone()),
                    T::B(branch) => T::B(branch.clone()),
                },
            }))
        }
    }

    fn verify(&self) -> bool {
        match &self.inner {
            T::L(leaf) => leaf.verify(),
            T::B(branch) => branch.verify(),
        }
    }

    fn len(&self) -> usize {
        match &self.inner {
            T::L(leaf) => leaf.len(),
            T::B(branch) => branch.len(),
        }
    }

    fn min(&self) -> &K {
        match &self.inner {
            T::L(leaf) => leaf.min(),
            T::B(branch) => branch.min(),
        }
    }

    fn max(&self) -> &K {
        match &self.inner {
            T::L(leaf) => leaf.max(),
            T::B(branch) => branch.max(),
        }
    }

    fn as_mut_leaf(&mut self) -> &mut Leaf<K, V> {
        match &mut self.inner {
            T::L(ref mut leaf) => leaf,
            T::B(_) => panic!(),
        }
    }
}

impl<K: Clone + PartialEq + PartialOrd, V: Clone> Branch<K, V> {
    pub fn new(pivot: K, left: ABNode<K, V>, right: ABNode<K, V>) -> Self {
        let mut new = Branch {
            count: 1,
            key: unsafe { MaybeUninit::uninit().assume_init() },
            node: unsafe { MaybeUninit::uninit().assume_init() },
        };
        unsafe {
            new.key[0].as_mut_ptr().write(pivot);
            new.node[0].as_mut_ptr().write(left);
            new.node[1].as_mut_ptr().write(right);
        }
        new
    }

    // Add a new pivot + node.

    // remove a node by idx.

    // get a node containing some K - need to return our related idx.

    pub(crate) fn min(&self) -> &K {
        unsafe { &*self.key[0].as_ptr() }
    }

    pub(crate) fn max(&self) -> &K {
        unsafe { &*self.key[self.count - 1].as_ptr() }
    }

    pub(crate) fn len(&self) -> usize {
        self.count
    }

    fn check_sorted(&self) -> bool {
        // check the pivots are sorted.
        if self.count == 0 {
            false
        } else {
            let mut lk: &K = unsafe { &*self.key[0].as_ptr() };
            for work_idx in 1..self.count {
                let rk: &K = unsafe { &*self.key[work_idx].as_ptr() };
                if lk >= rk {
                    return false;
                }
                lk = rk;
            }
            println!("Passed sorting");
            true
        }
    }

    fn check_descendents_valid(&self) -> bool {
        for work_idx in 0..self.count {
            // get left max and right min
            let lnode = unsafe { &*self.node[work_idx].as_ptr() };
            let rnode = unsafe { &*self.node[work_idx + 1].as_ptr() };

            let pkey = unsafe { &*self.key[work_idx].as_ptr() };
            let lkey = lnode.max();
            let rkey = rnode.min();
            if lkey >= pkey || pkey > rkey {
                println!("out of order key found");
                return false;
            }
        }
        println!("Passed descendants");
        true
    }

    fn verify_children(&self) -> bool {
        // For each child node call verify on it.
        for work_idx in 0..self.count {
            let node = unsafe { &*self.node[work_idx].as_ptr() };
            if !node.verify() {
                println!("Failed children");
                return false;
            }
        }
        println!("Passed children");
        true
    }

    pub(crate) fn verify(&self) -> bool {
        self.check_sorted() && self.check_descendents_valid() && self.verify_children()
    }
}

impl<K, V> Clone for Branch<K, V> {
    fn clone(&self) -> Self {
        unimplemented!();
    }
}

impl<K, V> Debug for Branch<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), Error> {
        write!(f, "Branch -> {}", self.count)
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
    use super::super::states::BNClone;
    use super::{Branch, Node};
    use std::sync::Arc;

    // check clone txid behaviour
    #[test]
    fn test_bptree_node_req_clone() {
        // Make a new node.
        let nroot: Node<usize, usize> = Node::new_leaf(0);
        // Req to clone it.
        match nroot.req_clone(0) {
            BNClone::Ok => {}
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

    #[test]
    fn test_bptree_node_new() {
        let k: usize = 5;
        let mut left = Arc::new(Box::new(Node::new_leaf(0)));
        let mut right = Arc::new(Box::new(Node::new_leaf(0)));

        // add some k, vs to each.
        {
            let lmut = Arc::get_mut(&mut left).unwrap().as_mut().as_mut_leaf();
            lmut.insert_or_update(0, 0);
            lmut.insert_or_update(1, 1);
        }
        {
            let rmut = Arc::get_mut(&mut right).unwrap().as_mut().as_mut_leaf();
            rmut.insert_or_update(5, 5);
            rmut.insert_or_update(6, 6);
        }

        println!("{:?}", left);
        println!("{:?}", right);

        let branch = Branch::new(k, left, right);
        println!("{:?}", branch);

        assert!(branch.verify());
    }
}
