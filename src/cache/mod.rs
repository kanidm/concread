//! Concurrently Readable caches. These are transactional caches with guarantees
//! about their items temporal consistencies. These caches may have many transactional
//! readers, and serialised writers operating at the same time, with readers able to
//! indicate inclusions and cache hits to the system.
pub mod arc;
