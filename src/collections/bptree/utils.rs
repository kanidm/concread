use std::ptr;

pub(crate) unsafe fn slice_insert<T>(slice: &mut [T], new: T, idx: usize) {
    ptr::copy(
        slice.as_ptr().add(idx),
        slice.as_mut_ptr().add(idx + 1),
        slice.len() - idx - 1,
    );
    ptr::write(slice.get_unchecked_mut(idx), new);
}

// From std::collections::btree::node.rs
pub(crate) unsafe fn slice_remove<T>(slice: &mut [T], idx: usize) -> T {
    // setup the value to be returned, IE give ownership to ret.
    let ret = ptr::read(slice.get_unchecked(idx));
    ptr::copy(
        slice.as_ptr().add(idx + 1),
        slice.as_mut_ptr().add(idx),
        slice.len() - idx - 1,
    );
    ret
}
