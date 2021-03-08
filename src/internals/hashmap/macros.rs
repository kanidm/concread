macro_rules! hash_key {
    ($self:expr, $k:expr) => {{
        let mut hasher = AHasher::new_with_keys($self.key1, $self.key2);
        $k.hash(&mut hasher);
        hasher.finish()
    }};
}

macro_rules! debug_assert_leaf {
    ($x:expr) => {{
        debug_assert!($x.meta.is_leaf());
    }};
}

macro_rules! debug_assert_branch {
    ($x:expr) => {{
        debug_assert!($x.meta.is_branch());
    }};
}

macro_rules! self_meta {
    ($x:expr) => {{
        #[allow(unused_unsafe)]
        unsafe {
            &mut *($x as *mut Meta)
        }
    }};
}

macro_rules! branch_ref {
    ($x:expr, $k:ty, $v:ty) => {{
        #[allow(unused_unsafe)]
        unsafe {
            debug_assert!((*$x).meta.is_branch());
            &mut *($x as *mut Branch<$k, $v>)
        }
    }};
}

macro_rules! leaf_ref {
    ($x:expr, $k:ty, $v:ty) => {{
        #[allow(unused_unsafe)]
        unsafe {
            debug_assert!((*$x).meta.is_leaf());
            &mut *($x as *mut Leaf<$k, $v>)
        }
    }};
}
