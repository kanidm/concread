use std::mem::MaybeUninit;
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

pub(crate) unsafe fn slice_merge<T>(dst: &mut [T], start_idx: usize, src: &mut [T], count: usize) {
    let dst_ptr = dst.as_mut_ptr().offset(start_idx as isize);
    let src_ptr = src.as_ptr();

    ptr::copy_nonoverlapping(src_ptr, dst_ptr, count);
}

pub(crate) unsafe fn slice_move<T>(
    dst: &mut [T],
    dst_start_idx: usize,
    src: &mut [T],
    src_start_idx: usize,
    count: usize,
) {
    let dst_ptr = dst.as_mut_ptr().offset(dst_start_idx as isize);
    let src_ptr = src.as_ptr().offset(src_start_idx as isize);

    ptr::copy_nonoverlapping(src_ptr, dst_ptr, count);
}

pub(crate) unsafe fn slice_slide_and_drop<T>(
    slice: &mut [MaybeUninit<T>],
    idx: usize,
    count: usize,
) {
    // drop everything up to and including idx
    for didx in 0..(idx + 1) {
        // These are dropped here ...?
        ptr::drop_in_place(slice[didx].as_mut_ptr());
    }
    // now move everything down.
    ptr::copy(slice.as_ptr().add(idx + 1), slice.as_mut_ptr(), count);
}
