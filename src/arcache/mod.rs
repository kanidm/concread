//! ARCache - A concurrently readable adaptive replacement cache.
//!
//! An ARCache is used in place of a `RwLock<LruCache>` or `Mutex<LruCache>`.
//! This structure is transactional, meaning that readers have guaranteed
//! point-in-time views of the cache and their items, while allowing writers
//! to proceed with inclusions and cache state management in parallel.
//!
//! This means that unlike a `RwLock` which can have many readers OR one writer
//! this cache is capable of many readers, over multiple data generations AND
//! writers that are serialised. This formally means that this is an ACID
//! compliant Cache.

mod ll;
#[macro_use]
mod common;
pub mod cache;
#[cfg(feature = "asynch")]
pub mod asynch;

pub use self::cache::{ARCache, ARCacheReadTxn, ARCacheWriteTxn};


