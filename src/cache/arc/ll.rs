use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ptr;

#[derive(Clone, Debug)]
pub(crate) struct LL<K>
where
    K: Hash + Eq + Ord + Clone + Debug,
{
    head: *mut LLNode<K>,
    tail: *mut LLNode<K>,
    size: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct LLNode<K>
where
    K: Hash + Eq + Ord + Clone + Debug,
{
    pub(crate) k: K,
    next: *mut LLNode<K>,
    prev: *mut LLNode<K>,
}

#[derive(Clone, Debug)]
pub(crate) struct LLIterMut<'a, K>
where
    K: Hash + Eq + Ord + Clone + Debug,
{
    next: *mut LLNode<K>,
    phantom: PhantomData<&'a K>,
}

impl<K> LL<K>
where
    K: Hash + Eq + Ord + Clone + Debug,
{
    pub(crate) fn new() -> Self {
        LL {
            head: ptr::null_mut(),
            tail: ptr::null_mut(),
            size: 0,
        }
    }

    pub(crate) fn iter_mut(&self) -> LLIterMut<K> {
        LLIterMut {
            next: self.head,
            phantom: PhantomData,
        }
    }

    // Append a k to the set, and return it's pointer.
    pub(crate) fn append_k(&mut self, k: K) -> *mut LLNode<K> {
        let n = LLNode::new(k);
        self.append_n(n);
        n
    }

    // Append an arbitrary node into this set.
    pub(crate) fn append_n(&mut self, n: *mut LLNode<K>) {
        if self.tail.is_null() {
            // If tail is null, head must also be null!
            debug_assert!(self.head.is_null());
            self.head = n;
            self.tail = n;
        } else {
            // If tail is set, head must NOT be null!
            debug_assert!(!self.head.is_null());
            unsafe {
                // And the tail's next must be null.
                debug_assert!((*self.tail).next.is_null());
            }
            unsafe {
                (*n).prev = self.tail;
                (*self.tail).next = n;
                self.tail = n;
            }
        }
        self.size += 1;
    }

    // Given a node ptr, extract and put it at the tail. IE hit.
    pub(crate) fn touch(&mut self, n: *mut LLNode<K>) {
        debug_assert!(self.size > 0);
        if n == self.tail {
            // Done, no-op
        } else {
            let _ = self.extract(n);
            self.append_n(n);
        }
    }

    // remove this node from the ll, and return it's ptr.
    pub(crate) fn pop(&mut self) -> *mut LLNode<K> {
        let n = self.head;
        debug_assert!(!n.is_null());
        self.extract(n);
        n
    }

    pub(crate) fn peek_head(&self) -> Option<&LLNode<K>> {
        if self.head.is_null() {
            None
        } else {
            let l = unsafe { &(*self.head) as &LLNode<K> };
            Some(l)
        }
    }

    pub(crate) fn peek_tail(&self) -> Option<&LLNode<K>> {
        if self.tail.is_null() {
            None
        } else {
            let l = unsafe { &(*self.tail) as &LLNode<K> };
            Some(l)
        }
    }

    // Cut a node out from this list from any location.
    pub(crate) fn extract(&mut self, n: *mut LLNode<K>) {
        debug_assert!(!n.is_null());
        if n == self.head && n == self.tail {
            // must be the only node, set head and tail to null.
            debug_assert!(self.size == 1);
            self.tail = ptr::null_mut();
            self.head = ptr::null_mut();
        } else if n == self.head {
            debug_assert!(self.size > 1);
            unsafe {
                self.head = (*n).next;
                (*self.head).prev = ptr::null_mut();
            }
        } else if n == self.tail {
            debug_assert!(self.size > 1);
            unsafe {
                self.tail = (*n).prev;
                (*self.tail).next = ptr::null_mut();
            }
        } else {
            unsafe {
                let left = (*n).prev;
                let right = (*n).next;
                debug_assert!(!left.is_null());
                debug_assert!(!right.is_null());
                (*left).next = right;
                (*right).prev = left;
            }
        }
        // null the links in the node we are poping.
        unsafe {
            (*n).next = ptr::null_mut();
            (*n).prev = ptr::null_mut();
        }
        self.size -= 1;
    }

    pub(crate) fn len(&self) -> usize {
        self.size
    }
}

impl<K> Drop for LL<K>
where
    K: Hash + Eq + Ord + Clone + Debug,
{
    fn drop(&mut self) {
        let mut n = self.head;
        while !n.is_null() {
            let next = unsafe { (*n).next };
            LLNode::free(n);
            n = next;
        }
    }
}

impl<K> LLNode<K>
where
    K: Hash + Eq + Ord + Clone + Debug,
{
    pub(crate) fn new(k: K) -> *mut Self {
        let b = Box::new(LLNode {
            k,
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
        });
        Box::into_raw(b)
    }

    pub(crate) fn free(v: *mut Self) {
        debug_assert!(!v.is_null());
        let _ = unsafe { Box::from_raw(v) };
    }
}

impl<'a, K> Iterator for LLIterMut<'a, K>
where
    K: Hash + Eq + Ord + Clone + Debug,
{
    type Item = &'a mut K;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next.is_null() {
            None
        } else {
            let r = Some(unsafe { &mut (*self.next).k });
            self.next = unsafe { (*self.next).next };
            r
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cache::arc::ll::LL;

    #[test]
    fn test_cache_arc_ll_basic() {
        let mut ll: LL<usize> = LL::new();
        assert!(ll.len() == 0);
        // Allocate new nodes
        let n1 = ll.append_k(1);
        let n2 = ll.append_k(2);
        let n3 = ll.append_k(3);
        let n4 = ll.append_k(4);
        // Check that n1 is the head, n3 is tail.
        assert!(ll.len() == 4);
        assert!(ll.peek_head().unwrap().k == 1);
        assert!(ll.peek_tail().unwrap().k == 4);

        // Touch 2, it's now tail.
        ll.touch(n2);
        assert!(ll.len() == 4);
        assert!(ll.peek_head().unwrap().k == 1);
        assert!(ll.peek_tail().unwrap().k == 2);

        // Touch 1 (head), it's the tail now.
        ll.touch(n1);
        assert!(ll.len() == 4);
        assert!(ll.peek_head().unwrap().k == 3);
        assert!(ll.peek_tail().unwrap().k == 1);

        // Touch 1 (tail), it stays as tail.
        ll.touch(n1);
        assert!(ll.len() == 4);
        assert!(ll.peek_head().unwrap().k == 3);
        assert!(ll.peek_tail().unwrap().k == 1);

        // pop from head
        let _n3 = ll.pop();
        assert!(ll.len() == 3);
        assert!(ll.peek_head().unwrap().k == 4);
        assert!(ll.peek_tail().unwrap().k == 1);

        // cut a node out from any (head, mid, tail)
        let _n2 = ll.extract(n2);
        assert!(ll.len() == 2);
        assert!(ll.peek_head().unwrap().k == 4);
        assert!(ll.peek_tail().unwrap().k == 1);

        let _n1 = ll.extract(n1);
        assert!(ll.len() == 1);
        assert!(ll.peek_head().unwrap().k == 4);
        assert!(ll.peek_tail().unwrap().k == 4);

        // test touch on ll of size 1
        ll.touch(n4);
        assert!(ll.len() == 1);
        assert!(ll.peek_head().unwrap().k == 4);
        assert!(ll.peek_tail().unwrap().k == 4);
        // Remove last
        let _n4 = ll.pop();
        assert!(ll.len() == 0);
        assert!(ll.peek_head().is_none());
        assert!(ll.peek_tail().is_none());
    }
}
