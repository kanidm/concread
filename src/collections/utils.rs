use std::borrow::Borrow;
use std::cmp::Ordering;
// use std::mem::MaybeUninit;
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
    let dst_ptr = dst.as_mut_ptr().add(start_idx);
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
    let dst_ptr = dst.as_mut_ptr().add(dst_start_idx);
    let src_ptr = src.as_ptr().add(src_start_idx);

    ptr::copy_nonoverlapping(src_ptr, dst_ptr, count);
}

/*
pub(crate) unsafe fn slice_slide_and_drop<T>(
    slice: &mut [MaybeUninit<T>],
    idx: usize,
    count: usize,
) {
    // drop everything up to and including idx
    for item in slice.iter_mut().take(idx + 1) {
        // These are dropped here ...?
        ptr::drop_in_place(item.as_mut_ptr());
    }
    // now move everything down.
    ptr::copy(slice.as_ptr().add(idx + 1), slice.as_mut_ptr(), count);
}

pub(crate) unsafe fn slice_slide<T>(slice: &mut [T], idx: usize, count: usize) {
    // now move everything down.
    ptr::copy(slice.as_ptr().add(idx + 1), slice.as_mut_ptr(), count);
}
*/

pub(crate) fn slice_search_linear<K, Q: ?Sized>(slice: &[K], k: &Q) -> Result<usize, usize>
where
    K: Borrow<Q>,
    Q: Ord,
{
    for (idx, nk) in slice.iter().enumerate() {
        let r = k.cmp(nk.borrow());
        match r {
            Ordering::Greater => {}
            Ordering::Equal => return Ok(idx),
            Ordering::Less => return Err(idx),
        }
    }
    Err(slice.len())
}
