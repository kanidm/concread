use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr;

pub trait LLWeight {
    fn ll_weight(&self) -> usize;
}

/*
impl<T> LLWeight for T {
    #[inline]
    default fn ll_weight(&self) -> usize {
        1
    }
}
*/

#[derive(Clone, Debug)]
pub(crate) struct LL<K>
where
    K: LLWeight + Clone + Debug,
{
    head: *mut LLNode<K>,
    tail: *mut LLNode<K>,
    size: usize,
    // tag: usize,
}

#[derive(Debug)]
pub(crate) struct LLNode<K>
where
    K: LLWeight + Clone + Debug,
{
    pub(crate) k: MaybeUninit<K>,
    next: *mut LLNode<K>,
    prev: *mut LLNode<K>,
    // tag: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct LLIterMut<'a, K>
where
    K: LLWeight + Clone + Debug,
{
    next: *mut LLNode<K>,
    end: *mut LLNode<K>,
    phantom: PhantomData<&'a K>,
}

impl<K> LL<K>
where
    K: LLWeight + Clone + Debug,
{
    pub(crate) fn new(// tag: usize
    ) -> Self {
        // assert!(tag > 0);
        let (head, tail) = LLNode::create_markers();
        LL {
            head,
            tail,
            size: 0,
            // tag,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn iter_mut(&self) -> LLIterMut<K> {
        LLIterMut {
            next: unsafe { (*self.head).next },
            end: self.tail,
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
        // Who is to the left of tail?
        unsafe {
            self.size += (*(*n).k.as_ptr()).ll_weight();
            // must be untagged
            // assert!((*n).tag == 0);
            debug_assert!((*self.tail).next.is_null());
            debug_assert!(!(*self.tail).prev.is_null());
            let pred = (*self.tail).prev;
            debug_assert!(!pred.is_null());
            debug_assert!((*pred).next == self.tail);
            (*n).prev = pred;
            (*n).next = self.tail;
            // (*n).tag = self.tag;
            (*pred).next = n;
            (*self.tail).prev = n;
            // We should have a prev and next
            debug_assert!(!(*n).prev.is_null());
            debug_assert!(!(*n).next.is_null());
            // And that prev's next is us, and next's prev is us.
            debug_assert!(!(*(*n).prev).next.is_null());
            debug_assert!(!(*(*n).next).prev.is_null());
            debug_assert!((*(*n).prev).next == n);
            debug_assert!((*(*n).next).prev == n);
        }
    }

    // Given a node ptr, extract and put it at the tail. IE hit.
    pub(crate) fn touch(&mut self, n: *mut LLNode<K>) {
        debug_assert!(self.size > 0);
        if n == unsafe { (*self.tail).prev } {
            // Done, no-op
        } else {
            self.extract(n);
            self.append_n(n);
        }
    }

    // remove this node from the ll, and return it's ptr.
    pub(crate) fn pop(&mut self) -> *mut LLNode<K> {
        let n = unsafe { (*self.head).next };
        self.extract(n);
        debug_assert!(!n.is_null());
        debug_assert!(n != self.head);
        debug_assert!(n != self.tail);
        n
    }

    // Cut a node out from this list from any location.
    pub(crate) fn extract(&mut self, n: *mut LLNode<K>) {
        assert!(self.size > 0);
        assert!(!n.is_null());
        unsafe {
            // We should have a prev and next
            debug_assert!(!(*n).prev.is_null());
            debug_assert!(!(*n).next.is_null());
            // And that prev's next is us, and next's prev is us.
            debug_assert!(!(*(*n).prev).next.is_null());
            debug_assert!(!(*(*n).next).prev.is_null());
            debug_assert!((*(*n).prev).next == n);
            debug_assert!((*(*n).next).prev == n);
            // And we belong to this set
            // assert!((*n).tag == self.tag);
            self.size -= (*(*n).k.as_ptr()).ll_weight();
        }

        unsafe {
            let prev = (*n).prev;
            let next = (*n).next;
            // prev <-> n <-> next
            (*next).prev = prev;
            (*prev).next = next;
            // Null things for paranoia.
            if cfg!(test) || cfg!(debug_assertions) {
                (*n).prev = ptr::null_mut();
                (*n).next = ptr::null_mut();
            }
            // (*n).tag = 0;
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.size
    }

    #[cfg(test)]
    pub(crate) fn peek_head(&self) -> Option<&K> {
        debug_assert!(!self.head.is_null());
        let next = unsafe { (*self.head).next };
        if next == self.tail {
            None
        } else {
            let l = unsafe {
                let ptr = (*next).k.as_ptr();
                &(*ptr) as &K
            };
            Some(l)
        }
    }

    #[cfg(test)]
    pub(crate) fn peek_tail(&self) -> Option<&K> {
        debug_assert!(!self.tail.is_null());
        let prev = unsafe { (*self.tail).prev };
        if prev == self.head {
            None
        } else {
            let l = unsafe {
                let ptr = (*prev).k.as_ptr();
                &(*ptr) as &K
            };
            Some(l)
        }
    }
}

impl<K> Drop for LL<K>
where
    K: LLWeight + Clone + Debug,
{
    fn drop(&mut self) {
        let head = self.head;
        let tail = self.tail;
        let mut n = unsafe { (*head).next };
        while n != tail {
            let next = unsafe { (*n).next };
            unsafe { ptr::drop_in_place((*n).k.as_mut_ptr()) };
            LLNode::free(n);
            n = next;
        }
        LLNode::free(head);
        LLNode::free(tail);
    }
}

impl<K> LLNode<K>
where
    K: LLWeight + Clone + Debug,
{
    #[inline]
    pub(crate) fn create_markers() -> (*mut Self, *mut Self) {
        let head = Box::into_raw(Box::new(LLNode {
            k: MaybeUninit::uninit(),
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
            // tag: 0,
        }));
        let tail = Box::into_raw(Box::new(LLNode {
            k: MaybeUninit::uninit(),
            next: ptr::null_mut(),
            prev: head,
            // tag: 0,
        }));
        unsafe {
            (*head).next = tail;
        }
        (head, tail)
    }

    #[inline]
    pub(crate) fn new(
        k: K,
        // tag: usize
    ) -> *mut Self {
        let b = Box::new(LLNode {
            k: MaybeUninit::new(k),
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
            // tag,
        });
        Box::into_raw(b)
    }

    #[inline]
    fn free(v: *mut Self) {
        debug_assert!(!v.is_null());
        let _ = unsafe { Box::from_raw(v) };
    }
}

impl<K> AsRef<K> for LLNode<K>
where
    K: LLWeight + Clone + Debug,
{
    fn as_ref(&self) -> &K {
        unsafe {
            let ptr = self.k.as_ptr();
            &(*ptr) as &K
        }
    }
}

impl<K> AsMut<K> for LLNode<K>
where
    K: LLWeight + Clone + Debug,
{
    fn as_mut(&mut self) -> &mut K {
        unsafe {
            let ptr = self.k.as_mut_ptr();
            &mut (*ptr) as &mut K
        }
    }
}

impl<'a, K> Iterator for LLIterMut<'a, K>
where
    K: LLWeight + Clone + Debug,
{
    type Item = &'a mut K;

    fn next(&mut self) -> Option<Self::Item> {
        debug_assert!(!self.next.is_null());
        if self.next == self.end {
            None
        } else {
            let r = Some(unsafe { (*self.next).as_mut() });
            self.next = unsafe { (*self.next).next };
            r
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LLWeight, LL};

    impl LLWeight for Box<usize> {
        #[inline]
        fn ll_weight(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_cache_arc_ll_basic() {
        // We test with box so that we leak on error
        let mut ll: LL<Box<usize>> = LL::new();
        assert!(ll.len() == 0);
        // Allocate new nodes
        let n1 = ll.append_k(Box::new(1));
        let n2 = ll.append_k(Box::new(2));
        let n3 = ll.append_k(Box::new(3));
        let n4 = ll.append_k(Box::new(4));
        // Check that n1 is the head, n3 is tail.
        assert!(ll.len() == 4);
        assert!(ll.peek_head().unwrap().as_ref() == &1);
        assert!(ll.peek_tail().unwrap().as_ref() == &4);

        // Touch 2, it's now tail.
        ll.touch(n2);
        assert!(ll.len() == 4);
        assert!(ll.peek_head().unwrap().as_ref() == &1);
        assert!(ll.peek_tail().unwrap().as_ref() == &2);

        // Touch 1 (head), it's the tail now.
        ll.touch(n1);
        assert!(ll.len() == 4);
        assert!(ll.peek_head().unwrap().as_ref() == &3);
        assert!(ll.peek_tail().unwrap().as_ref() == &1);

        // Touch 1 (tail), it stays as tail.
        ll.touch(n1);
        assert!(ll.len() == 4);
        assert!(ll.peek_head().unwrap().as_ref() == &3);
        assert!(ll.peek_tail().unwrap().as_ref() == &1);

        // pop from head
        let _n3 = ll.pop();
        assert!(ll.len() == 3);
        assert!(ll.peek_head().unwrap().as_ref() == &4);
        assert!(ll.peek_tail().unwrap().as_ref() == &1);

        // cut a node out from any (head, mid, tail)
        ll.extract(n2);
        assert!(ll.len() == 2);
        assert!(ll.peek_head().unwrap().as_ref() == &4);
        assert!(ll.peek_tail().unwrap().as_ref() == &1);

        ll.extract(n1);
        assert!(ll.len() == 1);
        assert!(ll.peek_head().unwrap().as_ref() == &4);
        assert!(ll.peek_tail().unwrap().as_ref() == &4);

        // test touch on ll of size 1
        ll.touch(n4);
        assert!(ll.len() == 1);
        assert!(ll.peek_head().unwrap().as_ref() == &4);
        assert!(ll.peek_tail().unwrap().as_ref() == &4);
        // Remove last
        let _n4 = ll.pop();
        assert!(ll.len() == 0);
        assert!(ll.peek_head().is_none());
        assert!(ll.peek_tail().is_none());

        // Add them all back so they are dropped.
        ll.append_n(n1);
        ll.append_n(n2);
        ll.append_n(n3);
        ll.append_n(n4);
    }

    #[derive(Clone, Debug)]
    struct Weighted {
        _i: u64,
    }

    impl LLWeight for Weighted {
        fn ll_weight(&self) -> usize {
            8
        }
    }

    #[test]
    fn test_cache_arc_ll_weighted() {
        // We test with box so that we leak on error
        let mut ll: LL<Weighted> = LL::new();
        assert!(ll.len() == 0);
        let _n1 = ll.append_k(Weighted { _i: 1 });
        assert!(ll.len() == 8);
        let _n2 = ll.append_k(Weighted { _i: 2 });
        assert!(ll.len() == 16);
        let n1 = ll.pop();
        assert!(ll.len() == 8);
        let n2 = ll.pop();
        assert!(ll.len() == 0);
        // Add back so they drop
        ll.append_n(n1);
        ll.append_n(n2);
    }
}
