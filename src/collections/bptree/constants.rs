

// This should be 8 or 16 to represent how many 64 bit data we want
// to contain (assuming a 64 bit type).
pub const CAPACITY: usize = 8;

// --> LEAVES
// Have two usize, one for txnid, one for count, which puts our math at.
// *  (8 - 2) / 2 = 3
// * (16 - 2) / 2 = 7
pub const L_CAPACITY: usize = (CAPACITY - 2) / 2;

// --> BRANCHES
// We need one more value than key.
pub const BK_CAPACITY: usize = L_CAPACITY - 1;
pub const BV_CAPACITY: usize = L_CAPACITY;

