use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
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

#[derive(Debug)]
pub(crate) struct LLNode<K>
where
    K: Hash + Eq + Ord + Clone + Debug,
{
    pub(crate) k: MaybeUninit<K>,
    next: *mut LLNode<K>,
    prev: *mut LLNode<K>,
}

#[derive(Clone, Debug)]
pub(crate) struct LLIterMut<'a, K>
where
    K: Hash + Eq + Ord + Clone + Debug,
{
    next: *mut LLNode<K>,
    end: *mut LLNode<K>,
    phantom: PhantomData<&'a K>,
}

impl<K> LL<K>
where
    K: Hash + Eq + Ord + Clone + Debug,
{
    pub(crate) fn new() -> Self {
        let (head, tail) = LLNode::create_markers();
        LL {
            head,
            tail,
            size: 0,
        }
    }

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
            debug_assert!((*self.tail).next.is_null());
            debug_assert!(!(*self.tail).prev.is_null());
            let pred = (*self.tail).prev;
            debug_assert!(!pred.is_null());
            debug_assert!((*pred).next == self.tail);
            (*n).prev = pred;
            (*n).next = self.tail;
            (*pred).next = n;
            (*self.tail).prev = n;
        }
        self.size += 1;
    }

    // Given a node ptr, extract and put it at the tail. IE hit.
    pub(crate) fn touch(&mut self, n: *mut LLNode<K>) {
        debug_assert!(self.size > 0);
        if n == unsafe { (*self.tail).prev } {
            // Done, no-op
        } else {
            let _ = self.extract(n);
            self.append_n(n);
        }
    }

    // remove this node from the ll, and return it's ptr.
    pub(crate) fn pop(&mut self) -> *mut LLNode<K> {
        let n = unsafe { (*self.head).next };
        debug_assert!(!n.is_null());
        self.extract(n);
        n
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

    // Cut a node out from this list from any location.
    pub(crate) fn extract(&mut self, n: *mut LLNode<K>) {
        debug_assert!(!n.is_null());

        unsafe {
            let prev = (*n).prev;
            let next = (*n).next;
            // prev <-> n <-> next
            (*next).prev = prev;
            (*prev).next = next;
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
    #[inline]
    pub(crate) fn create_markers() -> (*mut Self, *mut Self) {
        let head = Box::into_raw(Box::new(LLNode {
            k: MaybeUninit::uninit(),
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
        }));
        let tail = Box::into_raw(Box::new(LLNode {
            k: MaybeUninit::uninit(),
            next: ptr::null_mut(),
            prev: head,
        }));
        unsafe {
            (*head).next = tail;
        }
        (head, tail)
    }

    #[inline]
    pub(crate) fn new(k: K) -> *mut Self {
        let b = Box::new(LLNode {
            k: MaybeUninit::new(k),
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
        });
        Box::into_raw(b)
    }

    #[inline]
    pub(crate) fn free(v: *mut Self) {
        debug_assert!(!v.is_null());
        let _ = unsafe { Box::from_raw(v) };
    }
}

impl<K> AsRef<K> for LLNode<K>
where
    K: Hash + Eq + Ord + Clone + Debug,
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
    K: Hash + Eq + Ord + Clone + Debug,
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
    K: Hash + Eq + Ord + Clone + Debug,
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
        assert!(ll.peek_head().unwrap() == &1);
        assert!(ll.peek_tail().unwrap() == &4);

        // Touch 2, it's now tail.
        ll.touch(n2);
        assert!(ll.len() == 4);
        assert!(ll.peek_head().unwrap() == &1);
        assert!(ll.peek_tail().unwrap() == &2);

        // Touch 1 (head), it's the tail now.
        ll.touch(n1);
        assert!(ll.len() == 4);
        assert!(ll.peek_head().unwrap() == &3);
        assert!(ll.peek_tail().unwrap() == &1);

        // Touch 1 (tail), it stays as tail.
        ll.touch(n1);
        assert!(ll.len() == 4);
        assert!(ll.peek_head().unwrap() == &3);
        assert!(ll.peek_tail().unwrap() == &1);

        // pop from head
        let _n3 = ll.pop();
        assert!(ll.len() == 3);
        assert!(ll.peek_head().unwrap() == &4);
        assert!(ll.peek_tail().unwrap() == &1);

        // cut a node out from any (head, mid, tail)
        let _n2 = ll.extract(n2);
        assert!(ll.len() == 2);
        assert!(ll.peek_head().unwrap() == &4);
        assert!(ll.peek_tail().unwrap() == &1);

        let _n1 = ll.extract(n1);
        assert!(ll.len() == 1);
        assert!(ll.peek_head().unwrap() == &4);
        assert!(ll.peek_tail().unwrap() == &4);

        // test touch on ll of size 1
        ll.touch(n4);
        assert!(ll.len() == 1);
        assert!(ll.peek_head().unwrap() == &4);
        assert!(ll.peek_tail().unwrap() == &4);
        // Remove last
        let _n4 = ll.pop();
        assert!(ll.len() == 0);
        assert!(ll.peek_head().is_none());
        assert!(ll.peek_tail().is_none());
    }
}
