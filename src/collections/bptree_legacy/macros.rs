macro_rules! arc_get_mut_unsafe {
    ($tgt:expr) => {{
        // This is a fucking terrible piece of code.
        let ptr = $tgt.as_ref() as *const Node<K, V>;
        let evil_ptr = ptr as *mut Node<K, V>;
        unsafe { &mut (*evil_ptr) }
    }};
}
