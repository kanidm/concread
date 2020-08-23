#[derive(PartialEq, PartialOrd, Clone, Eq, Ord, Debug, Hash)]
pub enum M<T> {
    Some(T),
    None,
}

impl<T> M<T> {
    pub fn is_some(&self) -> bool {
        match self {
            M::Some(_) => true,
            M::None => false,
        }
    }

    pub fn unwrap(self) -> T {
        match self {
            M::Some(v) => v,
            M::None => panic!(),
        }
    }
}
