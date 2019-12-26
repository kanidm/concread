use std::fmt::{self, Debug, Error};
use std::mem::MaybeUninit;
use std::ptr;
use std::slice;
use std::sync::Arc;
use std::thread;

use super::constants::{BK_CAPACITY, BK_CAPACITY_MIN_N1, BV_CAPACITY};
use super::leaf::Leaf;
use super::states::BRInsertState;
use super::utils::*;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
thread_local!(static NODE_COUNTER: AtomicUsize = AtomicUsize::new(0));
#[cfg(test)]
thread_local!(static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0));

pub(crate) struct Branch<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    count: usize,
    key: [MaybeUninit<K>; BK_CAPACITY],
    node: [MaybeUninit<Arc<Box<Node<K, V>>>>; BV_CAPACITY],
}

#[derive(Debug)]
pub(crate) enum T<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    B(Branch<K, V>),
    L(Leaf<K, V>),
}

#[derive(Debug)]
pub(crate) struct Node<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    #[cfg(test)]
    pub nid: usize,
    pub txid: usize,
    inner: T<K, V>,
}

pub(crate) type ABNode<K, V> = Arc<Box<Node<K, V>>>;

impl<K: Clone + Ord + Debug, V: Clone> Node<K, V> {
    pub(crate) fn new_leaf(txid: usize) -> Self {
        Node {
            #[cfg(test)]
            nid: NODE_COUNTER.with(|nc| {
                nc.fetch_add(1, Ordering::AcqRel)
            }),
            txid: txid,
            inner: T::L(Leaf::new()),
        }
    }

    pub(crate) fn new_ableaf(txid: usize) -> ABNode<K, V> {
        Arc::new(Box::new(Self::new_leaf(txid)))
    }

    pub(crate) fn new_branch(txid: usize, l: ABNode<K, V>, r: ABNode<K, V>) -> ABNode<K, V> {
        Arc::new(Box::new(Node {
            #[cfg(test)]
            nid: NODE_COUNTER.with(|nc| {
                nc.fetch_add(1, Ordering::AcqRel)
            }),
            txid: txid,
            inner: T::B(Branch::new(l, r)),
        }))
    }

    pub(crate) fn new_leaf_ins(txid: usize, k: K, v: V) -> ABNode<K, V> {
        let mut node = Arc::new(Box::new(Node::new_leaf(txid)));
        {
            let nmut = Arc::get_mut(&mut node).unwrap().as_mut().as_mut_leaf();
            nmut.insert_or_update(k, v);
        }
        node
    }

    pub(crate) fn inner_clone(&self) -> T<K, V> {
        match &self.inner {
            T::L(leaf) => T::L(leaf.clone()),
            T::B(branch) => T::B(branch.clone()),
        }
    }

    pub(crate) fn req_clone(&self, txid: usize) -> ABNode<K, V> {
        debug_assert!(txid != self.txid);
        // Do we need to clone this node before we work on it?
        Arc::new(Box::new(Node {
            #[cfg(test)]
            nid: NODE_COUNTER.with(|nc| {
                nc.fetch_add(1, Ordering::AcqRel)
            }),
            txid: txid,
            inner: self.inner_clone(),
        }))
    }

    #[cfg(test)]
    pub(crate) fn verify(&self) -> bool {
        match &self.inner {
            T::L(leaf) => leaf.verify(),
            T::B(branch) => branch.verify(),
        }
    }

    pub(crate) fn len(&self) -> usize {
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

    pub(crate) fn get_ref(&self, k: &K) -> Option<&V> {
        match &self.inner {
            T::L(leaf) => {
                println!("leaf -> {:?}", leaf);
                leaf.get_ref(k)
            }
            T::B(branch) => {
                println!("branch -> {:?}", branch);
                branch.get_ref(k)
            }
        }
    }

    pub(crate) fn is_leaf(&self) -> bool {
        match &self.inner {
            T::L(_leaf) => true,
            T::B(_branch) => false,
        }
    }

    pub(crate) fn as_mut_leaf(&mut self) -> &mut Leaf<K, V> {
        match &mut self.inner {
            T::L(ref mut leaf) => leaf,
            T::B(_) => panic!(),
        }
    }

    pub(crate) fn as_leaf(&self) -> &Leaf<K, V> {
        match &self.inner {
            T::L(ref leaf) => leaf,
            T::B(_) => panic!(),
        }
    }

    pub(crate) fn as_mut_branch(&mut self) -> &mut Branch<K, V> {
        match &mut self.inner {
            T::L(_) => panic!(),
            T::B(ref mut branch) => branch,
        }
    }

    pub(crate) fn as_branch(&self) -> &Branch<K, V> {
        match &self.inner {
            T::L(_) => panic!(),
            T::B(ref branch) => branch,
        }
    }

    pub(crate) fn tree_density(&self) -> (usize, usize) {
        match &self.inner {
            T::L(leaf) => leaf.tree_density(),
            T::B(branch) => branch.tree_density(),
        }
    }

    pub(crate) fn leaf_count(&self) -> usize {
        match &self.inner {
            T::L(leaf) => 1,
            T::B(branch) => branch.leaf_count(),
        }
    }
}

impl<K: Clone + Ord + Debug, V: Clone> Branch<K, V> {
    pub fn new(left: ABNode<K, V>, right: ABNode<K, V>) -> Self {
        let pivot: K = (*right.min()).clone();
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
    pub(crate) fn add_node(&mut self, node: ABNode<K, V>) -> BRInsertState<K, V> {
        // Do we have space?
        if self.count == BK_CAPACITY {
            // if no space ->
            //    split and send two nodes back for new branch
            //
            // There are three possible states that this causes.
            // 1 * The inserted node is a low/middle value, causing max and max -1 to be returned.
            // 2 * The inserted node is the greater than all current values, causing l(max, node)
            //     to be returned.
            // 3 * The inserted node is between max - 1 and max, causing l(node, max) to be returned.
            //
            let kr: &K = node.min();
            // bst and find when min-key < key[idx]
            let r = {
                let (left, _) = self.key.split_at(self.count);
                let inited: &[K] =
                    unsafe { slice::from_raw_parts(left.as_ptr() as *const K, left.len()) };
                inited.binary_search(kr)
            };
            let ins_idx = r.unwrap_err();
            let res = match ins_idx {
                // Case 2
                BK_CAPACITY => {
                    // Greater than all current values, so we'll just return max and node.
                    let max = unsafe {
                        ptr::read(self.node.get_unchecked(BV_CAPACITY - 1)).assume_init()
                    };
                    // Drop the key between them.
                    let _kdrop =
                        unsafe { ptr::read(self.key.get_unchecked(BK_CAPACITY - 1)).assume_init() };
                    // Now setup the ret val
                    BRInsertState::Split(max, node)
                }
                // Case 3
                BK_CAPACITY_MIN_N1 => {
                    // Greater than all but max, so we return max and node in the correct order.
                    let max = unsafe {
                        ptr::read(self.node.get_unchecked(BV_CAPACITY - 1)).assume_init()
                    };
                    // Drop the key between them.
                    let _kdrop =
                        unsafe { ptr::read(self.key.get_unchecked(BK_CAPACITY - 1)).assume_init() };
                    // Now setup the ret val NOTICE compared to case 2 that we swap node and max?
                    BRInsertState::Split(node, max)
                }
                // Case 1
                ins_idx => {
                    // Get the max - 1 and max nodes out.
                    let maxn1 = unsafe {
                        ptr::read(self.node.get_unchecked(BV_CAPACITY - 2)).assume_init()
                    };
                    let max = unsafe {
                        ptr::read(self.node.get_unchecked(BV_CAPACITY - 1)).assume_init()
                    };
                    // Drop the key between them.
                    let _kdrop =
                        unsafe { ptr::read(self.key.get_unchecked(BK_CAPACITY - 1)).assume_init() };
                    // Drop the key before us that we are about to replace.
                    let _kdrop =
                        unsafe { ptr::read(self.key.get_unchecked(BK_CAPACITY - 2)).assume_init() };
                    #[cfg(test)]
                    {
                        println!("Removing -> {:?}, {:?}", maxn1.nid, max.nid);
                    }
                    // Add node and it's key to the correct location.
                    let k: K = kr.clone();
                    let leaf_ins_idx = ins_idx + 1;
                    unsafe {
                        slice_insert(&mut self.key, MaybeUninit::new(k), ins_idx);
                        slice_insert(&mut self.node, MaybeUninit::new(node), leaf_ins_idx);
                    }

                    BRInsertState::Split(maxn1, max)
                }
            };

            // Adjust the count, because we always remove at least 1 from the keys.
            self.count -= 1;
            res
        } else {
            // if space ->
            // Get the nodes min-key - we clone it because we'll certainly be inserting it!
            let k: K = node.min().clone();
            // bst and find when min-key < key[idx]
            let r = {
                let (left, _) = self.key.split_at(self.count);
                let inited: &[K] =
                    unsafe { slice::from_raw_parts(left.as_ptr() as *const K, left.len()) };
                inited.binary_search(&k)
            };
            // if r is ever found, I think this is a bug, because we should never be able to
            // add a node with an existing min.
            //
            //       [ 5 ]
            //        / \
            //    [0,]   [5,]
            //
            // So if we added here to [0, ], and it had to overflow to split, then everything
            // must be < 5. Why? Because to get to [0,] as your insert target, you must be < 5.
            // if we added to [5,] then a split must be greater than, or the insert would replace 5.
            //
            // if we consider
            //
            //       [ 5 ]
            //        / \
            //    [0,]   [7,]
            //
            // Now we insert 5, and 7, splits. 5 would remain in the tree and we'd split 7 to the right
            //
            // As a result, any "Ok(idx)" must represent a corruption of the tree.
            // debug_assert!(r.is_err());
            let ins_idx = r.unwrap_err();
            let leaf_ins_idx = ins_idx + 1;
            // So why do we only need to insert right? Because the left-most
            // leaf when it grows, it splits to the right. That importantly
            // means that we only need to insert to replace the min and it's
            // right leaf, or anything higher. As a result, we are always
            // targetting ins_idx and leaf_ins_idx = ins_idx + 1.
            //
            // We have a situation like:
            //
            //   [1, 3, 9, 18]
            //
            // and ins_idx is 2. IE:
            //
            //   [1, 3, 9, 18]
            //          ^-- k=6
            //
            // So this we need to shift those r-> and insert.
            //
            //   [1, 3, x, 9, 18]
            //          ^-- k=6
            //
            //   [1, 3, 6, 9, 18]
            //
            // Now we need to consider the leaves too:
            //
            //   [1, 3, 9, 18]
            //   | |  |  |   |
            //   v v  v  v   v
            //   0 1  3  9   18
            //
            // So that means we need to move leaf_ins_idx = (ins_idx + 1)
            // right also
            //
            //   [1, 3, x, 9, 18]
            //   | |  |  |  |   |
            //   v v  v  v  v   v
            //   0 1  3  x  9   18
            //           ^-- leaf for k=6 will go here.
            //
            // Now to talk about the right expand issue - lets say 0 conducted
            // a split, it returns the new right node - which would push
            // 3 to the right to insert a new right hand side as required. So we
            // really never need to consider the left most leaf to have to be
            // replaced in any conditions.
            //
            // Magic!
            unsafe {
                slice_insert(&mut self.key, MaybeUninit::new(k), ins_idx);
                slice_insert(&mut self.node, MaybeUninit::new(node), leaf_ins_idx);
            }
            // finally update the count
            self.count += 1;
            // Return that we are okay to go!
            BRInsertState::Ok
        }
    }

    // get a node containing some K - need to return our related idx.
    pub(crate) fn locate_node(&self, k: &K) -> usize {
        let r = {
            let (left, _) = self.key.split_at(self.count);
            let inited: &[K] =
                unsafe { slice::from_raw_parts(left.as_ptr() as *const K, left.len()) };
            inited.binary_search(&k)
        };

        // If the value is Ok(idx), then that means
        // we were located to the right node. This is because we
        // exactly hit and located on the key.
        //
        // If the value is Err(idx), then we have the exact index already.
        // as branches is one more.
        match r {
            Ok(v) => v + 1,
            Err(v) => v,
        }
    }

    // remove a node by idx.
    pub(crate) fn remove_by_idx(&mut self, idx: usize) -> () {
        debug_assert!(idx <= self.count);
        // remove by idx.
        unimplemented!();
    }

    pub(crate) fn replace_by_idx(&mut self, idx: usize, mut node: ABNode<K, V>) -> () {
        debug_assert!(idx <= self.count);
        // We have to swap the value at idx with this node, then ensure that the
        // prev is droped.
        unsafe {
            ptr::swap(self.node[idx].as_mut_ptr(), &mut node as *mut ABNode<K, V>);
        }
    }

    pub(crate) fn get_mut_idx(&mut self, idx: usize) -> &mut ABNode<K, V> {
        debug_assert!(idx <= self.count);
        unsafe { &mut *self.node[idx].as_mut_ptr() }
    }

    pub(crate) fn get_idx(&self, idx: usize) -> &ABNode<K, V> {
        debug_assert!(idx <= self.count);
        unsafe { &*self.node[idx].as_ptr() }
    }

    pub(crate) fn get_idx_checked(&self, idx: usize) -> Option<&ABNode<K, V>> {
        if idx <= self.count {
            Some(unsafe { &*self.node[idx].as_ptr() })
        } else {
            None
        }
    }

    pub(crate) fn min(&self) -> &K {
        unsafe { (*self.node[0].as_ptr()).min() }
    }

    pub(crate) fn max(&self) -> &K {
        unsafe { (*self.node[self.count].as_ptr()).max() }
    }

    pub(crate) fn len(&self) -> usize {
        self.count
    }

    pub(crate) fn get_ref(&self, k: &K) -> Option<&V> {
        // Which node should hold this value?
        let idx = self.locate_node(k);
        unsafe { (*self.node[idx].as_ptr()).get_ref(k) }
    }

    pub(crate) fn tree_density(&self) -> (usize, usize) {
        let mut lcount = 0; // leaf populated
        let mut mcount = 0; // leaf max possible
        for idx in 0..(self.count + 1) {
            let (l, m) = unsafe { (*self.node[idx].as_ptr()).tree_density() };
            lcount += l;
            mcount += m;
        }
        (lcount, mcount)
    }

    pub(crate) fn leaf_count(&self) -> usize {
        let mut lcount = 0;
        for idx in 0..(self.count + 1) {
            lcount += unsafe { (*self.node[idx].as_ptr()).leaf_count() };
        }
        lcount
    }

    #[cfg(test)]
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

    #[cfg(test)]
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

    #[cfg(test)]
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

    #[cfg(test)]
    pub(crate) fn verify(&self) -> bool {
        self.check_sorted() && self.check_descendents_valid() && self.verify_children()
    }
}

impl<K: Clone + Ord, V: Clone> Clone for Branch<K, V> {
    fn clone(&self) -> Self {
        let mut nkey: [MaybeUninit<K>; BK_CAPACITY] =
            unsafe { MaybeUninit::uninit().assume_init() };
        let mut nnode: [MaybeUninit<Arc<Box<Node<K, V>>>>; BV_CAPACITY] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for idx in 0..self.count {
            unsafe {
                let lkey = (*self.key[idx].as_ptr()).clone();
                nkey[idx].as_mut_ptr().write(lkey);
            }
            unsafe {
                let lnode = (*self.node[idx].as_ptr()).clone();
                nnode[idx].as_mut_ptr().write(lnode);
            }
        }
        // clone the last node.
        unsafe {
            let lnode = (*self.node[self.count].as_ptr()).clone();
            nnode[self.count].as_mut_ptr().write(lnode);
        }

        Branch {
            count: self.count,
            key: nkey,
            node: nnode,
        }
    }
}

impl<K: Clone + Ord + Debug, V: Clone> Debug for Branch<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), Error> {
        write!(f, "Branch -> {}\n", self.count)?;
        write!(f, "  \\-> [  |")?;
        for idx in 0..self.count {
            write!(f, "{:^6?}|", unsafe { &*self.key[idx].as_ptr() })?;
        }
        #[cfg(test)]
        {
            write!(f, " ]\n")?;
            write!(f, " nids [{:^6?}", unsafe { (*self.node[0].as_ptr()).nid })?;
            for idx in 0..self.count {
                write!(f, "{:^7?}", unsafe { (*self.node[idx + 1].as_ptr()).nid })?;
            }
            write!(f, " ]\n")?;
            write!(f, " mins [{:^6?}", unsafe {
                (*self.node[0].as_ptr()).min()
            })?;
            for idx in 0..self.count {
                write!(f, "{:^7?}", unsafe { (*self.node[idx + 1].as_ptr()).min() })?;
            }
        }
        write!(f, " ]\n")
    }
}

impl<K: Clone + Ord, V: Clone> Drop for Branch<K, V> {
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
        println!("branch dropped k:{}, v:{}", self.count, self.count + 1);
    }
}

#[cfg(test)]
impl<K: Clone + Ord, V: Clone> Drop for Node<K, V> {
    fn drop(&mut self) {
        let _ = DROP_COUNTER.with(|dc| {
            dc.fetch_add(1, Ordering::AcqRel)
        });
    }
}

#[cfg(test)]
pub(crate) fn check_drop_count() {
    let node = NODE_COUNTER.with(|nc| {
                nc.load(Ordering::Acquire)
            });
    let drop = DROP_COUNTER.with(|dc| {
            dc.load(Ordering::Acquire)
        });
    assert!(node == drop);
}

#[cfg(test)]
mod tests {
    use super::super::constants::BV_CAPACITY;
    use super::super::states::{BNClone, BRInsertState};
    use super::{ABNode, Branch, Node};
    use std::sync::Arc;

    #[test]
    fn test_bptree_node_new() {
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

        let branch = Branch::new(left, right);

        assert!(branch.verify());
    }

    fn create_branch_one_three() -> Branch<usize, usize> {
        let mut left = Arc::new(Box::new(Node::new_leaf(0)));
        let mut right = Arc::new(Box::new(Node::new_leaf(0)));
        {
            let lmut = Arc::get_mut(&mut left).unwrap().as_mut().as_mut_leaf();
            lmut.insert_or_update(1, 1);
            let rmut = Arc::get_mut(&mut right).unwrap().as_mut().as_mut_leaf();
            rmut.insert_or_update(3, 3);
        }
        Branch::new(left, right)
    }

    fn create_branch_one_three_max() -> Branch<usize, usize> {
        let mut branch = create_branch_one_three();
        // We - 3 here because we have two nodes from before
        // and we need 1 to be 100 so we know the max.
        assert!(BV_CAPACITY >= 3);
        for idx in 0..(BV_CAPACITY - 3) {
            let node = create_node(idx + 10);
            branch.add_node(node);
        }
        let node = create_node(100);
        branch.add_node(node);
        branch
    }

    fn create_node(v: usize) -> ABNode<usize, usize> {
        let mut node = Arc::new(Box::new(Node::new_leaf(0)));
        {
            let nmut = Arc::get_mut(&mut node).unwrap().as_mut().as_mut_leaf();
            nmut.insert_or_update(v, v);
        }
        node
    }

    #[test]
    fn test_bptree_node_add_min() {
        // Add a new node which is a new minimum. In theory this situation
        // should *never* occur as we always split *right*. But we handle it
        // for completeness sake.
        let node = create_node(0);
        let mut branch = create_branch_one_three();
        let r = branch.add_node(node);
        match r {
            BRInsertState::Ok => {}
            _ => panic!(),
        };
        // ALERT ALERT ALERT WARNING ATTENTION DANGER WILL ROBINSON
        // THIS IS ASSERTING THAT THE NODE IS NOW CORRUPTED AS INSERT MIN
        // SHOULD NEVER OCCUR!!!
        assert!(branch.verify() == false);
    }

    #[test]
    fn test_bptree_node_add_middle() {
        // Add a new node in "the middle" of existing nodes.
        let node = create_node(2);
        let mut branch = create_branch_one_three();
        let r = branch.add_node(node);
        match r {
            BRInsertState::Ok => {}
            _ => panic!(),
        };
        assert!(branch.verify());
    }

    #[test]
    fn test_bptree_node_add_max() {
        // Add a new max node.
        let node = create_node(4);
        let mut branch = create_branch_one_three();
        let r = branch.add_node(node);
        match r {
            BRInsertState::Ok => {}
            _ => panic!(),
        };
        assert!(branch.verify());
    }

    #[test]
    fn test_bptree_node_add_split_min() {
        // We don't test this, it should never occur.
        //
        assert!(true);
    }

    #[test]
    fn test_bptree_node_add_split_middle() {
        // Add a new middle node that wuld cause a split
        let node = create_node(4);
        let mut branch = create_branch_one_three_max();
        println!("test ins");
        let r = branch.add_node(node);
        match r {
            BRInsertState::Split(_, _) => {}
            _ => panic!(),
        };
        assert!(branch.verify());
    }

    #[test]
    fn test_bptree_node_add_split_max() {
        // Add a new max node that would cause this branch to split.
        let node = create_node(101);
        let mut branch = create_branch_one_three_max();
        println!("test ins");
        let r = branch.add_node(node);
        match r {
            BRInsertState::Split(_, r) => {
                assert!(r.min() == &101);
            }
            _ => panic!(),
        };
        assert!(branch.verify());
    }

    #[test]
    fn test_bptree_node_add_split_n1max() {
        // Add a value that is one before max that would trigger a split.
        let node = create_node(99);
        let mut branch = create_branch_one_three_max();
        println!("test ins");
        let r = branch.add_node(node);
        match r {
            BRInsertState::Split(l, _) => {
                assert!(l.min() == &99);
            }
            _ => panic!(),
        };
        assert!(branch.verify());
    }
}
