//! BptreeMap - A concurrently readable B+Tree(*) with Arc
//!
//! A BptreeMap can be used in place of a `RwLock<BTreeMap>` or
//! `Mutex<BTreeMap>`. This structure is transactional, meaning that
//! a read transaction can live for an extended period with a consistent
//! point-in-time view of the strucutre. Additionally, readers can exist
//! over multiple data generations, and do not block writers. Writers
//! are serialised.
//!
//! This is the `Arc` collected implementation. `Arc` is slower than
//! hazard pointers, but has more accurate (and instant) reclaim behaviour.
//!
//! It is not safe to implement this as an Ebr version due to the length
//! of time readers exist for which may cause epochs to never increment
//! leading to out-of-memory conditions.
//!
//! This structure is very different to the `im` crate. The `im` crate is
//! sync + send over individual operations. This means that multiple writes can
//! be interleaved atomicly and safely, and the readers always see the latest
//! data. While this is potentially useful to a set of problems, transactional
//! structures are suited to problems where readers have to maintain consistent
//! data views for a duration of time, cpu cache friendly behaviours and
//! database like transaction properties (ACID).

mod constants;
mod cursor;
pub mod iter;
mod leaf;
pub mod map;
mod node;
mod states;
mod utils;

pub use self::map::BptreeMap;
