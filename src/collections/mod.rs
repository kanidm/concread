//! Concurrent datastructures. These structures have MVCC or COW properties
//! allowing them to have one writer thread and multiple reader threads
//! exist at the same time. Readers have guaranteed "point in time" views
//! to these structures.

pub mod bptree;
pub mod hashmap;
mod utils;
// #[cfg(test)]
// mod maple_tree;
