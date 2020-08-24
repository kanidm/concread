// Number of k,v in sparse, and number of range values/links.
pub const CAPACITY: usize = 8;
// Number of pivots in range
pub const R_CAPACITY: usize = CAPACITY - 1;
// Number of values in dense.
pub const D_CAPACITY: usize = CAPACITY * 2;
