//! Iterators for the map.

// Iterators for the bptree
use super::leaf::Leaf;
use super::node::ABNode;
use std::collections::VecDeque;
use std::fmt::Debug;

pub(crate) struct LeafIter<'a, K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    length: Option<usize>,
    // idx: usize,
    stack: VecDeque<(&'a ABNode<K, V>, usize)>,
}

impl<'a, K: Clone + Ord + Debug, V: Clone> LeafIter<'a, K, V> {
    pub(crate) fn new(root: &'a ABNode<K, V>, size_hint: bool) -> Self {
        let length = if size_hint {
            Some(root.leaf_count())
        } else {
            None
        };

        // We probably need to position the VecDeque here.
        let mut stack = VecDeque::new();

        let mut work_node = root;
        loop {
            stack.push_back((work_node, 0));
            if work_node.is_leaf() {
                break;
            } else {
                work_node = work_node.as_branch().get_idx(0);
            }
        }

        LeafIter {
            length: length,
            // idx: 0,
            stack: stack,
        }
    }

    #[cfg(test)]
    pub(crate) fn new_base() -> Self {
        LeafIter {
            length: None,
            // idx: 0,
            stack: VecDeque::new(),
        }
    }

    pub(crate) fn stack_position(&mut self, idx: usize) {
        // Get the current branch, it must the the back.
        if let Some((bref, bpidx)) = self.stack.back() {
            let wbranch = bref.as_branch();
            if let Some(node) = wbranch.get_idx_checked(idx) {
                // Insert as much as possible now. First insert
                // our current idx, then all the 0, idxs.
                let mut work_node = node;
                let mut work_idx = idx;
                loop {
                    self.stack.push_back((work_node, work_idx));
                    if work_node.is_leaf() {
                        break;
                    } else {
                        work_idx = 0;
                        work_node = work_node.as_branch().get_idx(work_idx);
                    }
                }
            } else {
                // Unwind further.
                let bpidx = *bpidx + 1;
                let _ = self.stack.pop_back();
                self.stack_position(bpidx)
            }
        } else {
            // Must have been none, so we are exhausted. This means
            // the stack is empty, so return.
            return;
        }
    }

    /*
    fn peek(&'a mut self) -> Option<&'a Leaf<K, V>> {
        // I have no idea how peekable works, yolo.
        self.stack.back().map(|t| t.0.as_leaf())
    }
    */
}

impl<'a, K: Clone + Ord + Debug, V: Clone> Iterator for LeafIter<'a, K, V> {
    type Item = &'a Leaf<K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        // base case is the vecdeque is empty
        let (leafref, parent_idx) = match self.stack.pop_back() {
            Some(lr) => lr,
            None => return None,
        };

        // Setup the veqdeque for the next iteration.
        self.stack_position(parent_idx + 1);

        // Return the leaf as we found at the start, regardless of the
        // stack operations.
        Some(leafref.as_leaf())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.length {
            Some(l) => (l, Some(l)),
            // We aren't (shouldn't) be estimating
            None => (0, None),
        }
    }
}

/// Iterater over references to Key Value pairs stored in the map.
pub struct Iter<'a, K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    length: usize,
    idx: usize,
    curleaf: Option<&'a Leaf<K, V>>,
    leafiter: LeafIter<'a, K, V>,
}

impl<'a, K: Clone + Ord + Debug, V: Clone> Iter<'a, K, V> {
    pub(crate) fn new(root: &'a ABNode<K, V>, length: usize) -> Self {
        let mut liter = LeafIter::new(root, false);
        let leaf = liter.next();
        // We probably need to position the VecDeque here.
        Iter {
            length: length,
            idx: 0,
            curleaf: leaf,
            leafiter: liter,
        }
    }
}

impl<'a, K: Clone + Ord + Debug, V: Clone> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    /// Yield the next key value reference, or `None` if exhausted.
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(leaf) = self.curleaf {
            if let Some(r) = leaf.get_kv_idx_checked(self.idx) {
                self.idx += 1;
                Some(r)
            } else {
                self.curleaf = self.leafiter.next();
                self.idx = 0;
                self.next()
            }
        } else {
            None
        }
    }

    /// Provide a hint as to the number of items this iterator will yield.
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }
}

pub struct KeyIter<'a, K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    iter: Iter<'a, K, V>,
}

impl<'a, K: Clone + Ord + Debug, V: Clone> KeyIter<'a, K, V> {
    pub(crate) fn new(root: &'a ABNode<K, V>, length: usize) -> Self {
        KeyIter {
            iter: Iter::new(root, length),
        }
    }
}

impl<'a, K: Clone + Ord + Debug, V: Clone> Iterator for KeyIter<'a, K, V> {
    type Item = &'a K;

    /// Yield the next key value reference, or `None` if exhausted.
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub struct ValueIter<'a, K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
{
    iter: Iter<'a, K, V>,
}

impl<'a, K: Clone + Ord + Debug, V: Clone> ValueIter<'a, K, V> {
    pub(crate) fn new(root: &'a ABNode<K, V>, length: usize) -> Self {
        ValueIter {
            iter: Iter::new(root, length),
        }
    }
}

impl<'a, K: Clone + Ord + Debug, V: Clone> Iterator for ValueIter<'a, K, V> {
    type Item = &'a V;

    /// Yield the next key value reference, or `None` if exhausted.
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::super::constants::L_CAPACITY;
    use super::super::node::{ABNode, Node};
    use super::{Iter, LeafIter};
    use std::sync::Arc;

    fn create_leaf_node_full(vbase: usize) -> ABNode<usize, usize> {
        assert!(vbase % 10 == 0);
        let mut node = Arc::new(Node::new_leaf(0));
        {
            let nmut = Arc::get_mut(&mut node).unwrap().as_mut_leaf();
            for idx in 0..L_CAPACITY {
                let v = vbase + idx;
                nmut.insert_or_update(v, v);
            }
        }
        node
    }

    #[test]
    fn test_bptree_iter_leafiter_1() {
        let test_iter: LeafIter<usize, usize> = LeafIter::new_base();
        assert!(test_iter.count() == 0);
    }

    #[test]
    fn test_bptree_iter_leafiter_2() {
        let lnode = create_leaf_node_full(10);
        let mut test_iter = LeafIter::new(&lnode, true);

        assert!(test_iter.size_hint() == (1, Some(1)));

        let lref = test_iter.next().unwrap();
        assert!(lref.min() == &10);
        assert!(test_iter.next().is_none());
    }

    #[test]
    fn test_bptree_iter_leafiter_3() {
        let lnode = create_leaf_node_full(10);
        let rnode = create_leaf_node_full(20);
        let root = Node::new_branch(0, lnode, rnode);
        let mut test_iter = LeafIter::new(&root, true);

        assert!(test_iter.size_hint() == (2, Some(2)));
        let lref = test_iter.next().unwrap();
        let rref = test_iter.next().unwrap();
        assert!(lref.min() == &10);
        assert!(rref.min() == &20);
        assert!(test_iter.next().is_none());
    }

    #[test]
    fn test_bptree_iter_leafiter_4() {
        let l1node = create_leaf_node_full(10);
        let r1node = create_leaf_node_full(20);
        let l2node = create_leaf_node_full(30);
        let r2node = create_leaf_node_full(40);
        let b1node = Node::new_branch(0, l1node, r1node);
        let b2node = Node::new_branch(0, l2node, r2node);
        let root = Node::new_branch(0, b1node, b2node);
        let mut test_iter = LeafIter::new(&root, true);

        assert!(test_iter.size_hint() == (4, Some(4)));
        let l1ref = test_iter.next().unwrap();
        let r1ref = test_iter.next().unwrap();
        let l2ref = test_iter.next().unwrap();
        let r2ref = test_iter.next().unwrap();
        assert!(l1ref.min() == &10);
        assert!(r1ref.min() == &20);
        assert!(l2ref.min() == &30);
        assert!(r2ref.min() == &40);
        assert!(test_iter.next().is_none());
    }

    #[test]
    fn test_bptree_iter_leafiter_5() {
        let lnode = create_leaf_node_full(10);
        let mut test_iter = LeafIter::new(&lnode, true);

        assert!(test_iter.size_hint() == (1, Some(1)));

        let lref = test_iter.next().unwrap();
        assert!(lref.min() == &10);
        assert!(test_iter.next().is_none());
    }

    #[test]
    fn test_bptree_iter_iter_1() {
        // Make a tree
        let lnode = create_leaf_node_full(10);
        let rnode = create_leaf_node_full(20);
        let root = Node::new_branch(0, lnode, rnode);
        let test_iter = Iter::new(&root, L_CAPACITY * 2);

        assert!(test_iter.size_hint() == (L_CAPACITY * 2, Some(L_CAPACITY * 2)));
        assert!(test_iter.count() == L_CAPACITY * 2);
        // Iterate!
    }

    #[test]
    fn test_bptree_iter_iter_2() {
        let l1node = create_leaf_node_full(10);
        let r1node = create_leaf_node_full(20);
        let l2node = create_leaf_node_full(30);
        let r2node = create_leaf_node_full(40);
        let b1node = Node::new_branch(0, l1node, r1node);
        let b2node = Node::new_branch(0, l2node, r2node);
        let root = Node::new_branch(0, b1node, b2node);
        let test_iter = Iter::new(&root, L_CAPACITY * 4);

        println!("{:?}", test_iter.size_hint());

        assert!(test_iter.size_hint() == (L_CAPACITY * 4, Some(L_CAPACITY * 4)));
        assert!(test_iter.count() == L_CAPACITY * 4);
    }
}
