use std::collections;
use std::fmt::Debug;
use std::usize::MAX;
use std::{fmt, mem, ops};

extern crate num;

// Number of k,v in sparse, and number of range values/links.
const CAPACITY: usize = 8;
// Number of pivots in range
const R_CAPACITY: usize = CAPACITY - 1;
// Number of values in dense.
const D_CAPACITY: usize = CAPACITY * 2;

#[derive(PartialEq, PartialOrd, Clone, Eq, Ord, Debug, Hash)]
enum M<T> {
    Some(T),
    None,
}

impl<T> M<T> {
    fn is_some(&self) -> bool {
        match self {
            M::Some(_) => true,
            M::None => false,
        }
    }

    fn unwrap(self) -> T {
        match self {
            M::Some(v) => v,
            M::None => panic!(),
        }
    }
}

#[derive(Debug)]
struct SparseLeaf<K, V> {
    // This is an unsorted set of K, V pairs. Insert just appends (if possible),
    //, remove does NOT move the slots. On split, we sort-compact then move some
    // values (if needed).
    key: [M<K>; CAPACITY],
    value: [M<V>; CAPACITY],
}

#[derive(Debug)]
struct DenseLeaf<V> {
    value: [M<V>; D_CAPACITY],
}

#[derive(Debug, PartialEq)]
struct InternalPivots {
    lower: usize,
    upper: usize,
    lower_is_min: bool,
    upper_is_max: bool,
}

#[derive(Debug)]
struct RangeLeaf<K, V> {
    pivot: [M<K>; R_CAPACITY],
    value: [M<V>; CAPACITY],
}

#[derive(Debug)]
struct RangeBranch<K, V> {
    // Implied Pivots
    // Cap - 2
    pivot: [M<K>; R_CAPACITY],
    links: [M<*mut Node<K, V>>; CAPACITY],
}

// When K: SliceIndex, allow Dense
// When K: Binary, allow Range.

#[derive(Debug)]
enum NodeTag<K, V> {
    SL(SparseLeaf<K, V>),
    DL(DenseLeaf<V>),
    RL(RangeLeaf<K, V>),
    RB(RangeBranch<K, V>),
}

#[derive(Debug)]
struct Node<K, V> {
    tid: u64,
    // checksum: u32,
    inner: NodeTag<K, V>,
}

#[derive(Debug, PartialEq)]
enum RangeInsertState<K, V> {
    Ok(Option<V>),
    Err(K, K, M<V>, InternalPivots), // lowerKey, upperKey, value, InternalPivots for insert
}

impl<K, V> fmt::Display for RangeLeaf<K, V>
where
    K: std::fmt::Display,
    V: std::fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = write!(f, "pivot =   [");

        if result.is_err() {
            return result;
        }

        for i in 0..R_CAPACITY {
            match &self.pivot[i] {
                M::Some(p) => result = write!(f, "{}, ", p),
                M::None => result = write!(f, "X, "),
            };

            if result.is_err() {
                return result;
            }
        }
        result = write!(f, "]\nvalue = [");
        if result.is_err() {
            return result;
        }
        for i in 0..CAPACITY {
            match &self.value[i] {
                M::Some(v) => result = write!(f, "{}, ", v),
                M::None => result = write!(f, "X, "),
            };

            if result.is_err() {
                return result;
            }
        }
        write!(f, "]\n\n")
    }
}

impl InternalPivots {
    // returns an index for where the upper pivot needs to be inserted in the pivots array
    pub fn get_upper_index(
        &self,
        insert_count: usize,
        insert_upper: bool,
        insert_lower: bool,
    ) -> usize {
        if self.lower_is_min {
            if insert_count == 0 {
                return 0;
            } else {
                if insert_upper {
                    return self.lower + insert_count - 1;
                }

                return self.lower + insert_count;
            }
        }

        if insert_count == 0 {
            return self.lower + 1;
        } else {
            if insert_lower {
                return self.lower + 2;
            }
            return self.lower + 1;
        }
    }

    pub fn get_upper(&mut self) -> usize {
        if self.upper_is_max {
            return R_CAPACITY;
        }

        return self.upper;
    }

    pub fn increase_lower(&mut self, mut amount: usize) {
        if self.lower_is_min && amount != 0 {
            self.lower_is_min = false;
            amount -= 1;
        }

        if amount == 0 {
            return;
        }

        self.lower += amount;
    }

    pub fn decrease_lower(&mut self, mut amount: usize) {
        if amount == 0 || self.lower_is_min {
            return;
        }

        if self.lower < amount {
            // makes self.lower = -1 which means lower is refering to min
            if self.lower == amount - 1 {
                self.lower = 0;
                self.lower_is_min = true;
                return;
            } else {
                //panic!("attempting to subtract {} from {}, will result in overflow! aborting.",
                //      amount, self.lower);
            }
        }

        self.lower -= amount;
    }

    pub fn increase_upper(&mut self, mut amount: usize) {
        if self.upper_is_max {
            return;
        }

        if self.upper + 1 == R_CAPACITY {
            self.upper = 0;
            self.upper_is_max = true;
            return;
        }

        self.upper += amount;
    }

    pub fn decrease_upper(&mut self, mut amount: usize) {
        if amount == 0 {
            return;
        }

        if self.upper_is_max {
            self.upper = R_CAPACITY - 1;
            self.upper_is_max = false;
            return;
        }

        if self.upper < amount {
            panic!(
                "attempting to subtract {} from {}, will result in overflow! aborting.",
                amount, self.lower
            );
        }

        self.upper -= amount;
    }
}

impl<K, V> RangeLeaf<K, V>
where
    K: Clone + PartialEq + PartialOrd + num::Num + Debug,
    V: Clone + PartialEq + Debug,
{
    pub fn new() -> Self {
        RangeLeaf {
            pivot: [
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
            ],
            value: [
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
            ],
        }
    }

    pub fn search(&mut self, pivot: &K, min: &K, max: &K) -> Option<V> {
        if pivot >= max || pivot < min {
            return None;
        }

        let (mut internal_pivots, _) =
            self.find_internal_pivots(pivot, &(pivot.clone() + K::one()));
        let upper_pivot = internal_pivots.get_upper();

        match &self.value[upper_pivot] {
            M::Some(v) => {
                return Some(v.clone());
            }
            M::None => {}
        }

        None
    }

    // inserts the range:value pair lower_pivot..upper_pivot and value into the node,
    // returns false if the node is full even if compacting will allow the insert
    // it is the cursors job too try compacting the node on failure
    //
    // returns true if the range is successfully inserted or false on faliure to insert
    pub fn insert_range(
        &mut self,
        lower_pivot: K,
        upper_pivot: K,
        value: V,
        min: &K,
        max: &K,
    ) -> RangeInsertState<K, V> {
        return self.internal_insert(lower_pivot, upper_pivot, M::Some(value), min, max);
    }

    /* A diagram showing the 4 different cases for ranges to be deleted
     * this diagram only shows the pivots
     *
     *   case 2        case 4        case 3
     *   ________   _____________   ________
     *   |    | |   | |       | |   | |    |
     *   |   --- --- --- --- --- --- ---   |
     *  min | A | B | C | D | E | F | G | max
     *   |  |___|___|___|___|___|___|___|  |
     *   |_________________________________|
     *                  case 1
     *
     * case 1: removes all values from the node
     *
     * case 2: the lower pivot of the range to be removed is equal to min and the upper
     *          pivot of the range to be removed is either a pivot in the pivots array or
     *          a pivot that isn't in the pivot array but is part of a valid range between two pivots
     *
     * case 3: the lower pivot of the range to be removed is either a pivot in the pivots array or
     *          a pivot that isn't in the pivot array but is part of a valid range between two pivots
     *          and the upper pivot is equal to max
     *
     * case 4: the lower pivot of the range to be removed is either a pivot in the pivots array or
     *          a pivot that isn't in the pivot array but is part of a valid range between two pivots
     *          and the upper pivot of the range to be removed is either a pivot in the pivots array or
     *          a pivot that isn't in the pivot array but is part of a valid range between two pivots
     *
     *  returns true on successfull delete of range or on failure returns false
     */

    pub fn delete_range(
        &mut self,
        lower_pivot: K,
        mut upper_pivot: K,
        min: &K,
        max: &K,
    ) -> RangeInsertState<K, V> {
        let mut result = self.internal_insert(lower_pivot, upper_pivot, M::None, min, max);

        if let RangeInsertState::Ok(v) = result {
            return RangeInsertState::Ok(None);
        }

        return result;
    }

    /* A diagram showing the 4 different cases for ranges being inserted/deleted
     * this diagram only shows the pivots
     *
     *   case 2        case 4        case 3
     *   ________   _____________ __________
     *   |    | |   | |       | | | |      |
     *   |   --- --- --- --- --- --- ---   |
     *  min | A | B | C | D | E | F | G | max
     *   |  |___|___|___|___|___|___|___|  |
     *   |_________________________________|
     *                  case 1
     *
     */

    fn internal_insert(
        &mut self,
        lower_pivot: K,
        upper_pivot: K,
        mut value: M<V>,
        min: &K,
        max: &K,
    ) -> RangeInsertState<K, V> {
        let last_value: V;
        let mut range_insert_state: RangeInsertState<K, V>;

        // case 1
        if &lower_pivot == min && &upper_pivot == max {
            self.clear_node();
            self.value[0] = value;
            return RangeInsertState::Ok(None);
        }

        let (mut internal_pivots, last_range) =
            self.find_internal_pivots(&lower_pivot, &upper_pivot);
        let mut distance = self.get_dist_between_pivots(&internal_pivots);
        let (insert_lower, insert_upper) = self.check_if_pivots_need_insert(
            &internal_pivots,
            &lower_pivot,
            &upper_pivot,
            min,
            max,
        );
        let insert_count = self.get_insert_count(insert_lower, insert_upper);
        let mut return_value: Option<V> = None;

        // make sure the value we are inserting is not the same as a value
        // that is already present
        if distance == 0 {
            if internal_pivots.upper_is_max {
                return_value = match self.value[CAPACITY - 1] {
                    M::Some(ref v) => Some(v.clone()),
                    M::None => None,
                };
            } else {
                return_value = match self.value[internal_pivots.upper] {
                    M::Some(ref v) => Some(v.clone()),
                    M::None => None,
                };
            }

            if self.value[internal_pivots.upper] == value {
                match value {
                    M::Some(v) => return RangeInsertState::Ok(Some(v)),
                    M::None => return RangeInsertState::Ok(None),
                };
            }
            if value == M::None && last_range {
                internal_pivots.increase_lower(1);
                if self.value[internal_pivots.lower] == M::None {
                    return RangeInsertState::Ok(None);
                }
                internal_pivots.decrease_lower(1);
            }
        }
        // case 2
        if &lower_pivot == min {
            if self.make_space_for_insert(
                distance,
                insert_count,
                &internal_pivots,
                max,
                insert_lower,
                insert_upper,
            ) {
                let upper_index =
                    internal_pivots.get_upper_index(insert_count, insert_upper, insert_lower);
                let valid_range_above = self.next_range_is_valid(&internal_pivots);

                if insert_upper {
                    self.insert_upper_pivot(
                        &upper_pivot,
                        &mut internal_pivots,
                        distance,
                        upper_index,
                        insert_count,
                        valid_range_above,
                        last_range,
                    );
                }

                if distance == 0 {
                    range_insert_state = RangeInsertState::Ok(return_value);
                } else {
                    range_insert_state = RangeInsertState::Ok(None);
                }

                self.value[upper_index] = value;

                return range_insert_state;
            }
        }
        // case 3
        else if &upper_pivot == max {
            if insert_lower {
                // if the internal_pivots.lower is the last pivot
                // in the array there is no room to insert another so
                // return false
                if internal_pivots.lower < R_CAPACITY - 1 {
                    // increment the lower pivot index to point to the new index
                    // for the pivot inserted so the for loop below will start
                    // at the right index when deleteing pivots
                    internal_pivots.increase_lower(1);
                    self.pivot[internal_pivots.lower] = M::Some(lower_pivot);
                } else {
                    return RangeInsertState::Err(lower_pivot, upper_pivot, value, internal_pivots);
                }
            }

            if distance == 0 {
                range_insert_state = RangeInsertState::Ok(return_value);
            } else {
                range_insert_state = RangeInsertState::Ok(None);
            }

            self.value[internal_pivots.lower + 1] = value;

            for i in internal_pivots.lower + 1..R_CAPACITY {
                self.pivot[i] = M::None;
            }

            for i in internal_pivots.lower + 2..CAPACITY {
                self.value[i] = M::None;
            }

            return range_insert_state;;
        }
        // case 4
        else {
            if self.make_space_for_insert(
                distance,
                insert_count,
                &internal_pivots,
                max,
                insert_lower,
                insert_upper,
            ) {
                let upper_index =
                    internal_pivots.get_upper_index(insert_count, insert_upper, insert_lower);
                let valid_range_above = self.next_range_is_valid(&internal_pivots);

                if insert_upper {
                    self.insert_upper_pivot(
                        &upper_pivot,
                        &mut internal_pivots,
                        distance,
                        upper_index,
                        insert_count,
                        valid_range_above,
                        last_range,
                    );
                }

                if insert_lower {
                    if !self.insert_lower_pivot(&lower_pivot, &mut internal_pivots) {
                        return RangeInsertState::Err(
                            lower_pivot,
                            upper_pivot,
                            value,
                            internal_pivots,
                        );
                    }
                }

                if distance == 0 {
                    range_insert_state = RangeInsertState::Ok(return_value);
                } else {
                    range_insert_state = RangeInsertState::Ok(None);
                }

                self.value[upper_index] = value;

                return range_insert_state;
            }
        }

        return RangeInsertState::Err(lower_pivot, upper_pivot, value, internal_pivots);
    }

    //when there is a range from a pivot in the node to max that
    // where the pivot isn't directly next to max the pivot next to it
    // will be M::None, to check if there is a valid value next to a pivot
    // we check the value
    //
    // min [10, 20, 30, X, X, X, X] max
    //   [a,   b,  c,  d, x, x, x, x]
    //
    // the range 30 to max has value d
    fn next_range_is_valid(&mut self, internal_pivots: &InternalPivots) -> bool {
        if internal_pivots.upper_is_max {
            return false;
        } else if self.value[internal_pivots.upper] != M::None {
            return true;
        }

        false
    }

    // inserts the upper pivot whilst makine sure that values are move or copied where required
    fn insert_upper_pivot(
        &mut self,
        upper_pivot: &K,
        internal_pivots: &mut InternalPivots,
        distance: usize,
        upper_index: usize,
        insert_count: usize,
        valid_range_above: bool,
        last_range: bool,
    ) {
        let mut value_holder: M<V> = M::None;

        // if we are here then insert count must be either 1 or 2
        //
        // when distance is 0 this means that we are inserting inbetween two
        // adjacent pivots and thus there must be a right shift of the pivots,
        //
        // [10 | 20] -> [10, p1, 20] or [10, p1, p2, 20]
        //     |           |   |           |   |   |
        //     a           a   X           a   b   X
        //
        // so we must copy the value a to where X is in both cases and
        // in the first case then b(the value to be inserted) goes
        // where a was, in the second case b is already inserted
        //
        if last_range {
            internal_pivots.increase_lower(1);
            if self.value[internal_pivots.lower] != M::None {
                value_holder = self.value[internal_pivots.lower].clone();
                self.value[upper_index + 1] = value_holder;
            }
            internal_pivots.decrease_lower(1);
        } else if distance == 0 {
            internal_pivots.increase_lower(1);
            value_holder = self.value[internal_pivots.lower].clone();
            self.value[upper_index + 1] = value_holder;
            internal_pivots.decrease_lower(1);
        }
        //when insert count == 2 and
        else if insert_count == 2 && upper_index != R_CAPACITY - 1 {
            if (valid_range_above || self.pivot[upper_index + 1] != M::None)
                && self.value[upper_index + 1] == M::None
            {
                value_holder = self.value[upper_index].clone();
                self.value[upper_index + 1] = value_holder;
            }
        }

        self.pivot[upper_index] = M::Some(upper_pivot.clone());
    }

    fn insert_lower_pivot(
        &mut self,
        lower_pivot: &K,
        internal_pivots: &mut InternalPivots,
    ) -> bool {
        // if the internal_pivots.lower is the last pivot
        // in the array there is no room to insert another so
        // return false
        if internal_pivots.lower < R_CAPACITY - 1 {
            // increment the lower pivot index to point to the new index
            // for the pivot inserted so the for loop below will start
            // at the right index when deleteing pivots
            internal_pivots.increase_lower(1);
            self.pivot[internal_pivots.lower] = M::Some(lower_pivot.clone());
            return true;
        }

        // NOTE: should I return the internal_pivots here?
        return false;
    }

    // checks if the upper_pivot and lower_pivot need to be inserted into the node then
    // returns a tuple containing two boolean values (insert_lower_pivot, insert_upper_pivot)
    // TODO: split into two functions and name get_insert_upper and get_insert_lower
    fn check_if_pivots_need_insert(
        &mut self,
        internal_pivots: &InternalPivots,
        lower_pivot: &K,
        upper_pivot: &K,
        min: &K,
        max: &K,
    ) -> (bool, bool) {
        let insert_upper_pivot;
        let insert_lower_pivot;

        if !internal_pivots.upper_is_max {
            insert_upper_pivot = match &self.pivot[internal_pivots.upper] {
                M::Some(p) => upper_pivot != p,
                M::None => true,
            };
        } else {
            insert_upper_pivot = upper_pivot != max;
        }

        if !internal_pivots.lower_is_min {
            insert_lower_pivot = match &self.pivot[internal_pivots.lower] {
                M::Some(p) => lower_pivot != p,
                M::None => false,
            };
        } else {
            insert_lower_pivot = lower_pivot != min;
        }

        return (insert_lower_pivot, insert_upper_pivot);
    }

    fn get_dist_between_pivots(&self, internal_pivots: &InternalPivots) -> usize {
        if internal_pivots.lower_is_min && internal_pivots.upper_is_max {
            return R_CAPACITY;
        } else if internal_pivots.lower_is_min {
            return internal_pivots.upper;
        } else if internal_pivots.upper_is_max {
            return R_CAPACITY - (internal_pivots.lower + 1);
        }

        return internal_pivots.upper - internal_pivots.lower - 1;
    }

    // loops over the values linearly to find if there are any contiguous values
    // that are the same. This is the first step in condensing a RangeLeaf node.
    //
    // for all sets of contiguous values that are the same, the start index of the pivot
    // for the first value and, the end pivot for the last value are added to an InternalPivots
    // variable and pushed onto the found_ranges vector which is then returned after all
    // values have been checked
    pub fn find_ranges_to_condense(&mut self) -> Vec<InternalPivots> {
        let mut range_start_index = 0;
        let mut matched = false;
        let mut found_ranges: Vec<InternalPivots> = Vec::new();

        {
            let mut previous_val: &M<V> = &self.value[0];

            for i in 1..CAPACITY {
                if &self.value[i] == previous_val {
                    if !matched {
                        matched = true;
                        range_start_index = i - 1;
                    }
                } else if matched {
                    found_ranges.push(InternalPivots {
                        lower: range_start_index,
                        upper: i - 1,
                        lower_is_min: false,
                        upper_is_max: (i - 1) == R_CAPACITY,
                    });
                    matched = false;
                }

                previous_val = &self.value[i];
            }
        }

        if matched {
            found_ranges.push(InternalPivots {
                lower: range_start_index,
                upper: 0,
                lower_is_min: false,
                upper_is_max: true,
            });
        }

        found_ranges
    }

    // compacts contiguous ranges that share a value.
    // any contiguous ranges that have the same value can be compacted
    // by making a new range with the lower pivot as the lower pivot of the first range
    // in the set of contiguous ranges and the upper pivot of the new range as the upper pivot of
    // the last range in the set of contiguous ranges
    //
    // i.e if we have a contiguous set of ranges ([1,5],[5,10],[10,16]) this is the
    // same as the range [1,16], within the rangeNode each range has a value so if a set of contiguous
    // ranges have the same value then we can compact them into one range
    //
    // returns true if the node was compacted or false if there was nothing to do
    pub fn compact_node(&mut self, max: &K) -> bool {
        let found_ranges: Vec<InternalPivots>;

        found_ranges = self.find_ranges_to_condense();
        if found_ranges.is_empty() {
            return false;
        }

        let mut pivot_index_to_move;
        let mut distance;

        for internal_pivots in found_ranges.iter().rev() {
            distance = self.get_dist_between_pivots(internal_pivots) + 1;

            pivot_index_to_move = if internal_pivots.upper_is_max {
                R_CAPACITY
            } else {
                internal_pivots.upper
            };

            self.move_pivots_left(pivot_index_to_move, distance, max);
        }

        true
    }

    // completely clears the node setting all pivots and values to M::None
    pub fn clear_node(&mut self) {
        for i in 0..R_CAPACITY {
            self.pivot[i] = M::None;
        }

        for i in 0..CAPACITY {
            self.value[i] = M::None;
        }
    }

    fn get_insert_count(&mut self, insert_lower: bool, insert_upper: bool) -> usize {
        if insert_lower && insert_upper {
            2
        } else if insert_lower || insert_upper {
            1
        } else {
            0
        }
    }

    // if possible moves the pivots and values within the rangeNode to accomodate
    // the new pivot/s and value to be inserted
    //
    // returns false if there is not enough space to make room for the new pivots
    // otherwise returns true on success
    pub fn make_space_for_insert(
        &mut self,
        distance_betweeen_pivots: usize,
        mut pivot_insert_count: usize,
        internal_pivots: &InternalPivots,
        max: &K,
        insert_lower_pivot: bool,
        insert_upper_pivot: bool,
    ) -> bool {
        let range_end = if internal_pivots.upper_is_max {
            R_CAPACITY
        } else {
            internal_pivots.upper
        };

        // start and end pivots of the range have pivots inbetween
        if distance_betweeen_pivots != 0 {
            let left_shift_amount: isize =
                distance_betweeen_pivots as isize - pivot_insert_count as isize;

            // pivot_insert_count can only ever be 2 so if distance_betweeen_pivots is 1
            // leftShift amount could be -1 so we need to right shift by 1
            if left_shift_amount == -1 {
                if self.move_pivots_right(range_end, 1) == false {
                    return false;
                }
            } else if internal_pivots.upper_is_max {
                if self.move_pivots_left(R_CAPACITY, left_shift_amount as usize, max) == false {
                    return false;
                }
            } else {
                if self.move_pivots_left(range_end, left_shift_amount as usize, max) == false {
                    return false;
                }
            }
        }
        // pivots are adjacent
        else {
            if internal_pivots.upper_is_max && (insert_lower_pivot || insert_upper_pivot) {
                return false;
            }

            if !internal_pivots.upper_is_max {
                if self.pivot[internal_pivots.upper] == M::None {
                    pivot_insert_count -= 1;
                }
            }

            if self.move_pivots_right(range_end, pivot_insert_count) == false {
                let mut clear_last_pivot = false;
                // special case when the pivots array looks like [p1, p2, p3, p4, p5, p6, max]
                match &self.pivot[R_CAPACITY - 1] {
                    M::Some(p) => {
                        if p == max && pivot_insert_count == 1 {
                            clear_last_pivot = true;
                        }
                    }
                    M::None => {}
                }

                if clear_last_pivot {
                    self.pivot[R_CAPACITY - 1] = M::None;
                    self.value[CAPACITY - 1] = self.value[CAPACITY - 2].clone();
                    self.move_pivots_right(range_end, 1);
                    return true;
                }

                //there isn't enough room in the node to insert
                return false;
            }
        }

        true
    }

    /*
     * Definitions for comparing ranges:
     *
     *  A = [a_1, a_2) Note: a_1 < a_2
     *  B = [b_1, b_2) Note: b_1 < b_2
     *  x, a_1, a_2, b_1 and b_2 all have '<', '>' and '=' defined
     *
     *  SINGLE VALUE DEFINITIONS:
     *
     *  (x < A)         := x < a_1
     *  (x > A)         := x >= a_2 (Note: a_2 is exclusive to the range so if x = a_2 it is larger than the range)
     *  (x \in A)       := a_1 < x < a_2
     *
     *  RANGE DEFINITIONS:
     *
     *  Note: if !(A > B) that doesn't neccesarily mean (B > A)
     *  (A < B)         := a_2 <= b_1
     *  (A > B)         := a_1 >= b_2
     *  (A \subset B)   := (a_1 >= b_1) & (a_2 < b_2
     *  ((A \intersect B) != {}) & (A \notSubset B))
     *
     */

    // TODO: re-write this description following the advice from programming/communication
    // returns the index's of the range that holds the pivot
    // Note: This can be over many ranges i.e from R1 - R6
    //       which would return InternalPivots{start: -1, end: 5} as the indexes
    //
    // Range A is the range to be inserted i.e range_lower..range_upper
    // Range B is the current range being assesed within the RangeLeaf node
    // Range C is the range that came before B Note: C's UpperBound is B's lowerBound
    //
    // rangeNumbers     -   [ R1  | R2 | R3 | R4 | R5 | R6 | R7 |  R8 ]
    // Pivots & min max - [min | P1 | P2 | P3 | P4 | P5 | P6 | P7 | max]
    //
    // this function assumes the following to be true:
    //      range_lower >= min
    //      range_upper <= max
    //
    // The algorithm bellow uses the following basic steps:
    //  1. Find where !(A > B) (This lets us know where the range A starts at)
    //      since a_1 < b_2 and we know a_1 > c_2 since C comes before B
    //  2. Check if (A \subset C) in which case we know that the Range is C
    //      return Range here if found
    //  3. Find where A < D where D is a range that is after B
    //
    pub fn find_internal_pivots(
        &mut self,
        range_lower: &K,
        range_upper: &K,
    ) -> (InternalPivots, bool) {
        // range number of B
        // i.e range number 1 [min, P1)
        let mut range_b: usize = 0;
        let mut index: usize = 0; // used to keep the index from the for loops
        let mut set = false;
        let mut is_none = false;

        // find range B  where !(A > B)
        // since we know that for all index's up to the current value of i
        // range_lower > self.pivot[i] when rangeLower < self.pivot[i] we know
        // that the range starts at the (i-1)th index
        for i in 0..R_CAPACITY {
            index = i;
            match &self.pivot[i] {
                M::Some(b2) => {
                    if range_lower < b2 {
                        range_b = i + 1;
                        set = true;
                        break;
                    }
                }
                M::None => {
                    is_none = true;
                    break;
                }
            }
        }

        // if we didn't find any pivot where range_lower < self.pivot[i]
        // then we either ran into a None value or since we don't check min or max
        // the range must be between pivots at the index's R_CAPACITY and R_CAPCITY-1
        if !set {
            if is_none {
                if index == 0 {
                    return (
                        InternalPivots {
                            lower: 0,
                            upper: 0,
                            lower_is_min: true,
                            upper_is_max: false,
                        },
                        !set,
                    );
                } else {
                    return (
                        InternalPivots {
                            lower: index - 1,
                            upper: index,
                            lower_is_min: false,
                            upper_is_max: false,
                        },
                        !set,
                    );
                }
            } else {
                return (
                    InternalPivots {
                        lower: R_CAPACITY - 1,
                        upper: 0,
                        lower_is_min: false,
                        upper_is_max: true,
                    },
                    !set,
                );
            }
        }

        // now we know that the range starts at index range_b-1 and we need to find
        // where the range ends.
        for i in (range_b - 1)..R_CAPACITY {
            match &self.pivot[i] {
                M::Some(b2) => {
                    if range_upper <= b2 {
                        if range_b == 1 {
                            return (
                                InternalPivots {
                                    lower: 0,
                                    upper: i,
                                    lower_is_min: true,
                                    upper_is_max: false,
                                },
                                !set,
                            );
                        } else {
                            return (
                                InternalPivots {
                                    lower: (range_b - 2),
                                    upper: i,
                                    lower_is_min: false,
                                    upper_is_max: false,
                                },
                                !set,
                            );
                        }
                    }
                }
                M::None => {
                    if range_b == 1 {
                        return (
                            InternalPivots {
                                lower: 0,
                                upper: i,
                                lower_is_min: true,
                                upper_is_max: false,
                            },
                            !set,
                        );
                    } else {
                        return (
                            InternalPivots {
                                lower: (range_b - 2),
                                upper: i,
                                lower_is_min: false,
                                upper_is_max: false,
                            },
                            !set,
                        );
                    }
                }
            }
        }
        if range_b == 1 {
            return (
                InternalPivots {
                    lower: 0,
                    upper: 0,
                    lower_is_min: true,
                    upper_is_max: true,
                },
                !set,
            );
        } else {
            return (
                InternalPivots {
                    lower: (range_b - 2),
                    upper: 0,
                    lower_is_min: false,
                    upper_is_max: true,
                },
                !set,
            );
        }
    }

    fn get_free_pivots(&mut self, startIndex: usize) -> usize {
        let mut free_pivots = 0;

        for i in (startIndex..R_CAPACITY).rev() {
            if let M::None = self.pivot[i] {
                free_pivots += 1;
            }
        }

        free_pivots
    }

    pub fn move_pivots_left(
        &mut self,
        mut pivot_index: usize,
        mut distance: usize,
        max: &K,
    ) -> bool {
        let mut last_range_in_use = false;

        if pivot_index > R_CAPACITY {
            return false;
        } else if distance == 0 {
            return true;
        }

        // the new index for the pivot at pivot_index is pivotIndex-distance
        // make sure this is not negative
        if (pivot_index as isize) - (distance as isize) < 0 {
            return false;
        }

        let mut pivot_holder: M<K> = M::None;
        let mut value_holder: M<V> = M::None;

        // move all of the pivots to the left
        for i in pivot_index..R_CAPACITY {
            mem::swap(&mut self.pivot[i], &mut pivot_holder);
            mem::swap(&mut self.pivot[i - distance], &mut pivot_holder);
            pivot_holder = M::None;
        }

        // move all of the values to the left
        for i in pivot_index..CAPACITY {
            mem::swap(&mut self.value[i], &mut value_holder);
            mem::swap(&mut self.value[i - distance], &mut value_holder);
            value_holder = M::None;
        }

        // set any pivots and values to M::None that should have
        // in some cases there will be some pivots and values that arne't set
        // to M::None when they should be, this happens when the distance that the pivots are being
        // shifted is larger than the amount of pivots between the pivot_index and the end of the
        // array including the pivot at pivot_index
        let places_to_end = R_CAPACITY - pivot_index;
        if distance > places_to_end {
            for i in R_CAPACITY - distance..R_CAPACITY {
                self.pivot[i] = M::None;
                self.value[i + 1] = M::None;
            }
        }
        return true;
    }

    // moves the pivot/s at pivot_index and above to the right x distance where x = distance variable
    // if the move would push values off the end then false is returned
    pub fn move_pivots_right(&mut self, pivot_index: usize, distance: usize) -> bool {
        if distance == 0 {
            return true;
        }

        let free_pivots = self.get_free_pivots(R_CAPACITY - distance);
        if free_pivots < distance {
            return false;
        }
        // make sure the pivot is greater than 0 and less than the index of the
        // last pivot as moving that would push it out of the array
        else if pivot_index >= R_CAPACITY - 1 {
            return false;
        } else if pivot_index + distance > R_CAPACITY {
            return false;
        }

        // must use a variable to hold the pivots and values between swaps as we can't have two
        // mutable borrows of the same array at the same time :(
        let mut pivot_holder: M<K> = M::None;
        let mut value_holder: M<V> = M::None;
        let last_index = (R_CAPACITY) - free_pivots;

        for i in (pivot_index..last_index).rev() {
            mem::swap(&mut pivot_holder, &mut self.pivot[i]);
            mem::swap(&mut pivot_holder, &mut self.pivot[i + distance]);
        }

        for i in ((pivot_index + 1)..last_index + 1).rev() {
            mem::swap(&mut value_holder, &mut self.value[i]);
            mem::swap(&mut value_holder, &mut self.value[i + distance]);
        }

        return true;
    }
}

impl<K, V> SparseLeaf<K, V>
where
    K: Clone + PartialEq + PartialOrd + Debug,
    V: Debug,
{
    pub fn new() -> Self {
        SparseLeaf {
            key: [
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
            ],
            value: [
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
                M::None,
            ],
        }
    }

    // insert takes a key and a value and tries to find a place to
    // insert it, if the sparseLeaf is already full and therefore
    // the key and value can't be inserted None is returned
    pub fn insert(&mut self, k: K, v: V) -> Option<()> {
        // Just find a free slot and insert
        for i in 0..CAPACITY {
            println!("insert: {:?}", self.key[i]);
            if !self.key[i].is_some() {
                // Make the new k: v here
                let mut nk = M::Some(k);
                let mut nv = M::Some(v);
                // swap them
                mem::swap(&mut self.key[i], &mut nk);
                mem::swap(&mut self.value[i], &mut nv);
                // return Some
                return Some(());
            }
        }

        None
    }

    // search tries to find a key that matches k within the sparseLeaf
    // if the key can't be found None is returned
    // if the key is found but the value associated to it is a None value
    // since this is not allowed we panick
    pub fn search(&mut self, k: &K) -> Option<&V> {
        for i in 0..CAPACITY {
            match &self.key[i] {
                M::Some(v) => {
                    if v == k {
                        match &self.value[i] {
                            M::Some(v) => {
                                return Some(v);
                            }
                            M::None => panic!("SparseLeaf - search failure. None value found associated to a valid key, aborting."),
                        }
                    }
                }

                M::None => {}
            }
        }

        None
    }

    // update attempts to change the associated value of a key with a new value
    // if the key isn't found then None is returned and nothing is updated,
    // if the value that was associated with the key before the update was a None
    //  value then we panick because that should not happen
    pub fn update(&mut self, k: K, v: V) -> Option<V> {
        for i in 0..CAPACITY {
            match &self.key[i] {
                M::Some(v) => {
                    if v != &k {
                        continue;
                    }
                }

                M::None => {
                    continue;
                }
            }

            let mut nv = M::Some(v);

            mem::swap(&mut self.value[i], &mut nv);

            match nv {
                M::Some(v) => return Some(v),
                M::None => panic!("SparseLeaf - update failure. None value found associated to a valid key, aborting."),
            }
        }

        None
    }

    // remove attempts to delete a key/value pair from the sparseLeaf
    // if the key is found and the value isn't None then the value is returned
    // if the key isn't found then None is returned
    // if the keys associated value is None then we panic because that shouldn't happen
    pub fn remove(&mut self, k: &K) -> Option<V> {
        for i in 0..CAPACITY {
            println!("remove: {:?}", self.key[i]);

            match &self.key[i] {
                M::Some(v) => {
                    if v != k {
                        continue;
                    }
                }
                M::None => {
                    continue;
                }
            }

            let mut nk = M::None;
            let mut nv = M::None;

            mem::swap(&mut self.key[i], &mut nk);
            mem::swap(&mut self.value[i], &mut nv);

            match nv {
                M::Some(v) => {
                    return Some(v);
                }
                M::None => panic!(
                    "SparseLeaf - remove() None value found associated to a valid key, aborting."
                ),
            }
        }

        None
    }

    // either returns some(k) holding the largest key in the node
    // or none if the node is empty
    pub fn get_max(&self) -> Option<&K> {
        let mut max: &M<K> = &self.key[0];
        let mut key_found: bool = false;

        for key in self.key.iter() {
            match key {
                M::Some(_) => {
                    if key_found == false {
                        max = key;
                        key_found = true;
                    } else if max < key {
                        max = key;
                    }
                }
                M::None => continue,
            }
        }

        match max {
            M::Some(v) => return Some(&v),
            M::None => return None,
        }
    }

    // either returns some(k) holding the smallest key in the node
    // or none if the node is empty
    pub fn get_min(&self) -> Option<&K> {
        let mut min: &M<K> = &self.key[0];
        let mut key_found: bool = false;

        for key in self.key.iter() {
            match key {
                M::Some(_k) => {
                    if key_found == false {
                        min = key;
                        key_found = true;
                    } else if min > key {
                        min = key;
                    }
                }
                M::None => {}
            }
        }

        match min {
            M::Some(k) => return Some(k),
            M::None => return None,
        }
    }

    // This function is used to help verify the validity of the entire tree
    // this function returns true if all keys within the SparseLeaf are within the bounds
    // min to max or equal to min or max or the SparseLeaf node is empty
    // otherwise this returns false
    pub fn check_bounds(&mut self, min_bound: &K, max_bound: &K) -> bool {
        let min = self.get_min();
        let max = self.get_max();

        // if either min or max is None they must both be None
        // if they are both None then the Node MUST be empty and
        // we can return true
        if min == None && max == None {
            return true;
        }

        if min >= Some(&min_bound) && max <= Some(&max_bound) {
            return true;
        }

        false
    }

    // We need to sort *just before* we split if required.
    // This function implements selection sort for a SparseLeaf
    // creates a new SparseLeaf struct, inserts the minimal values
    // one by one then overwrites the old struct with the new one
    //
    // we create the new node so we don't have to deal with the None values
    // being in-between values otherwise the code would be more complex to handle
    // compacting the values and then sorting or vice versa so there is no gaps
    // between actual keys in the underlying array
    pub fn sort(&mut self) {
        let mut smallest_key_index: usize;
        let mut sl: SparseLeaf<K, V> = SparseLeaf::new();
        let mut sl_index: usize = 0;

        // run once for every key in the sparseLeaf
        for _ in 0..CAPACITY {
            smallest_key_index = 0;

            // run a pass over the remaining items to be sorted to find the
            // entry with the smallest key and swap it for the item at currentIndex
            for i in 0..8 {
                match self.key[i] {
                    M::Some(_) => {
                        if self.key[i] < self.key[smallest_key_index] {
                            smallest_key_index = i;
                        }
                    }
                    M::None => continue,
                }
            }

            // swap the found element into the new SparseLeaf with the M::None
            // that is already in the SparseLeaf instead of just using the insert method
            // on the new SparseLeaf so the sorting function will keep working
            //
            // we could also just just insert the values into the new node and set the value of
            // the old node to M::None manually but that would require more code and I figured
            // this was a bit cleaner, thoughts?
            mem::swap(&mut self.key[smallest_key_index], &mut sl.key[sl_index]);
            mem::swap(&mut self.value[smallest_key_index], &mut sl.value[sl_index]);
            sl_index += 1;
        }

        *self = sl;
    }
}

#[cfg(test)]
mod tests {
    use super::InternalPivots;
    use super::RangeInsertState;
    use super::RangeLeaf;
    use super::SparseLeaf;
    use super::M;
    use collections::maple_tree::CAPACITY;
    use collections::maple_tree::R_CAPACITY;
    use std::ops;
    use std::usize::MAX;

    #[test]
    fn test_internal_pivots_increase_lower() {
        let mut ip = InternalPivots {
            lower: 0,
            upper: 0,
            lower_is_min: true,
            upper_is_max: true,
        };

        ip.increase_lower(1);
        assert!(ip.lower_is_min == false);
        assert!(ip.lower == 0);

        ip.increase_lower(3);
        assert!(ip.lower_is_min == false);
        assert!(ip.lower == 3);
    }

    #[test]
    #[should_panic]
    fn test_internal_pivots_decrease_lower() {
        let mut ip = InternalPivots {
            lower: 3,
            upper: 0,
            lower_is_min: false,
            upper_is_max: true,
        };

        ip.decrease_lower(4);
        assert!(ip.lower_is_min == true);
        assert!(ip.lower == 0);

        // this test panics so there is no assert and
        // it is at the very end of the functio
        ip.lower = 3;
        ip.lower_is_min = false;
        ip.decrease_lower(5)
    }

    #[test]
    fn test_internal_pivots_increase_upper() {
        let mut ip = InternalPivots {
            lower: 0,
            upper: R_CAPACITY - 1,
            lower_is_min: true,
            upper_is_max: false,
        };
        ip.increase_upper(1);
        assert!(ip.upper == 0);
        assert!(ip.upper_is_max == true);
    }

    #[test]
    fn test_internal_pivots_decrease_upper() {
        let mut ip = InternalPivots {
            lower: 0,
            upper: 0,
            lower_is_min: true,
            upper_is_max: true,
        };

        ip.decrease_upper(1);

        assert!(ip.upper == R_CAPACITY - 1);
        assert!(ip.upper_is_max == false);
    }

    #[test]
    fn test_sparse_leaf_get_max() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();
        let test_vals: [usize; 8] = [3, 8, 7, 4, 2, 1, 5, 6];

        for val in test_vals.iter() {
            sl.insert(*val, *val);
        }

        // check that get_max() works for a full node
        assert!(sl.get_max() == Some(&8));

        //check that get_max() works for a node with Nones inbetween values
        let del_vals: [usize; 4] = [3, 8, 2, 1];
        for val in del_vals.iter() {
            sl.remove(val);
        }

        assert!(sl.get_max() == Some(&7));

        // check that get_min() works for empty nodes
        let sl_empty: SparseLeaf<usize, usize> = SparseLeaf::new();
        assert!(sl_empty.get_max() == None);
    }

    #[test]
    fn test_sparse_leaf_get_min() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();
        let test_vals: [usize; 8] = [3, 8, 7, 4, 2, 1, 5, 6];

        for val in test_vals.iter() {
            sl.insert(*val, *val);
        }

        // check that get_min() works for a full node
        assert!(sl.get_min() == Some(&1));

        //check that get_min() works for a node with Nones inbetween values
        let del_vals: [usize; 4] = [3, 8, 2, 1];
        for val in del_vals.iter() {
            sl.remove(val);
        }

        assert!(sl.get_min() == Some(&4));

        // check that get_min() works for empty nodes
        let sl_empty: SparseLeaf<usize, usize> = SparseLeaf::new();
        assert!(sl_empty.get_min() == None);
    }

    #[test]
    fn test_sparse_leaf_search() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();

        // test valid search
        sl.insert(2, 2);

        assert!(sl.search(&2).is_some());

        // test invalid search
        assert!(sl.search(&3).is_none());

        sl = SparseLeaf::new();

        for i in 0..CAPACITY {
            sl.insert(i, i);
        }

        sl.remove(&3);

        assert!(sl.search(&4).is_some());
    }

    #[test]
    fn test_sparse_leaf_update() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();

        // Insert K:V pair
        sl.insert(2, 2);

        // update inplace.
        sl.update(2, 3);

        // check that the value was correctly changed
        assert!(sl.search(&2) == Some(&3));
    }

    #[test]
    fn test_sparse_leaf_insert() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();

        // insert
        sl.insert(2, 2);

        // test valid search
        assert!(sl.search(&2) == Some(&2));

        // test invalid search
        assert!(sl.search(&1).is_none());

        // test insert after node is already full

        for i in 1..CAPACITY {
            sl.insert(i, i);
        }

        assert!(sl.insert(8, 8).is_none())
    }

    #[test]
    fn test_sparse_leaf_remove() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();

        // check removing a non-existent value fails
        assert!(sl.remove(&0).is_none());

        // check removing a value that exists
        sl.insert(0, 0);
        assert!(sl.remove(&0).is_some());

        // check removing existing values out of order is successfull
        let remove_keys = [3, 7, 8, 1, 4];
        for i in 0..CAPACITY {
            sl.insert(i, i);
        }
        for i in 0..remove_keys.len() {
            assert!(sl.remove(&i).is_some());
        }
    }

    #[test]
    fn test_sparse_leaf_check_bounds() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();

        // test that check_min_max returns true when the sparseLeaf is empty
        assert!(sl.check_bounds(&0, &8));

        // insert 8 values from 0 - 7
        for i in 0..CAPACITY - 3 {
            sl.insert(i, i);
        }

        assert!(sl.check_bounds(&0, &8));

        // test that check_min_max returns some when the values are out of the range
        // and returns the first value that is found outside the range.

        sl.insert(10, 10);
        sl.insert(11, 11);
        sl.insert(12, 12);
        assert!(sl.check_bounds(&0, &8) == false);
    }

    #[test]
    fn test_sparse_leaf_sort() {
        // test sorting full node
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();
        let mut test_vals: [usize; 8] = [3, 8, 7, 4, 2, 1, 5, 6];
        let sorted_test_vals: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

        for val in test_vals.iter() {
            sl.insert(*val, *val);
        }
        for i in 0..CAPACITY {
            match sl.key[i] {
                M::Some(v) => {
                    println!("{0}", v);
                }
                M::None => println!("None"),
            }
        }
        sl.sort();

        for i in 0..CAPACITY {
            // the code inside match is usefull for debuging if a test fails
            match sl.key[i] {
                M::Some(v) => {
                    println!(
                        "(actualValue = {0}) - (sortedTestValue = {1})",
                        v, sorted_test_vals[i]
                    );
                }
                M::None => println!("None - {}", sorted_test_vals[i]),
            }
            assert!(sl.key[i] == M::Some(sorted_test_vals[i]));
        }

        // test sorting half full node with M::None's inbetween each value
        // i.e [3, None, 4, None, 2, None, 1, None]

        test_vals = [3, 8, 4, 6, 2, 7, 1, 5];
        let none_positions: [usize; 4] = [8, 6, 7, 5];
        sl = SparseLeaf::new();

        for val in test_vals.iter() {
            sl.insert(*val, *val);
        }

        // remove every second value from sl
        for val in none_positions.iter() {
            sl.remove(&val);
        }

        sl.sort();

        for i in 0..4 {
            println!("{} <-> ", sorted_test_vals[i]);
            assert!(sl.key[i] == M::Some(sorted_test_vals[i]));
        }
    }

    #[test]
    fn test_range_node_internal_insert() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();

        // set state for first test return value
        let mut return_value = RangeInsertState::<usize, usize>::Ok(None);

        //
        //#########  CASE 1 ###########
        //
        println!("###### CASE 1 ######");
        set_node_state(
            [5, 10, 20, 30, 40, 50, 60],
            [1, 1, 1, 4, 2, 2, 2, 8],
            &mut rn,
        );
        assert!(rn.internal_insert(0, MAX, M::Some(20), &0, &MAX) == return_value);
        check_node_state([0, 0, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0, 0, 0], &rn);

        println!("###### CASE 3 ######");
        //
        //#########  CASE 3 ###########
        //

        set_node_state(
            [5, 10, 20, 30, 40, 50, 60],
            [1, 1, 1, 4, 2, 2, 2, 8],
            &mut rn,
        );

        assert!(rn.internal_insert(30, MAX, M::Some(20), &0, &MAX) == return_value);
        check_node_state([5, 10, 20, 30, 0, 0, 0], [1, 1, 1, 4, 20, 0, 0, 0], &rn);

        set_node_state(
            [5, 10, 20, 30, 40, 50, 60],
            [1, 1, 1, 4, 2, 2, 2, 8],
            &mut rn,
        );
        assert!(rn.internal_insert(35, MAX, M::Some(20), &0, &MAX) == return_value);
        println!("{}", rn);
        check_node_state([5, 10, 20, 30, 35, 0, 0], [1, 1, 1, 4, 2, 20, 0, 0], &rn);

        //
        //#########  CASE 2 ###########
        //
        println!("###### CASE 2 ######");

        set_node_state(
            [10, 20, 30, 40, 50, 0, 0],
            [1, 1, 1, 4, 2, 2, 0, 0],
            &mut rn,
        );
        assert!(rn.internal_insert(0, 20, M::Some(20), &0, &MAX) == RangeInsertState::Ok(None));
        println!("{}", rn);
        check_node_state([20, 30, 40, 50, 0, 0, 0], [20, 1, 4, 2, 2, 0, 0, 0], &rn);

        set_node_state(
            [10, 20, 30, 40, 50, 0, 0],
            [1, 1, 1, 4, 2, 2, 0, 0],
            &mut rn,
        );
        assert!(rn.internal_insert(0, 5, M::Some(20), &0, &MAX) == RangeInsertState::Ok(Some(1)));
        println!("{}", rn);
        check_node_state([5, 10, 20, 30, 40, 50, 0], [20, 1, 1, 1, 4, 2, 2, 0], &rn);

        set_node_state(
            [10, 20, 30, 40, 50, 0, 0],
            [1, 1, 1, 4, 2, 2, 0, 0],
            &mut rn,
        );
        assert!(rn.internal_insert(0, 10, M::Some(20), &0, &MAX) == RangeInsertState::Ok(Some(1)));
        println!("{}", rn);
        check_node_state([10, 20, 30, 40, 50, 0, 0], [20, 1, 1, 4, 2, 2, 0, 0], &rn);

        set_node_state(
            [10, 20, 30, 40, 50, 0, 0],
            [1, 1, 1, 4, 2, 2, 0, 0],
            &mut rn,
        );
        assert!(rn.internal_insert(0, 55, M::Some(20), &0, &MAX) == RangeInsertState::Ok(None));
        println!("{}", rn);
        check_node_state([55, 0, 0, 0, 0, 0, 0], [20, 2, 0, 0, 0, 0, 0, 0], &rn);

        //
        //#########  CASE 4 ###########
        //
        println!("###### CASE 4 ######");

        // insert range inbetween two nodes directly next to each other
        // where the pivots to be inserted don't equal either the upper
        // or lower pivot in the internal_range
        set_node_state(
            [10, 20, 30, 40, 50, 0, 0],
            [1, 1, 1, 4, 2, 2, 0, 0],
            &mut rn,
        );
        assert!(rn.internal_insert(22, 28, M::Some(5), &0, &MAX) == RangeInsertState::Ok(Some(1)));
        check_node_state([10, 20, 22, 28, 30, 40, 50], [1, 1, 1, 5, 1, 4, 2, 2], &rn);

        // insert range where the distance between the upper and lower pivot is
        // greater than 0
        // where the pivots to be inserted don't exist in the node
        set_node_state(
            [10, 20, 30, 40, 50, 0, 0],
            [1, 1, 1, 4, 2, 2, 0, 0],
            &mut rn,
        );
        assert!(rn.internal_insert(25, 35, M::Some(5), &0, &MAX) == RangeInsertState::Ok(None));
        println!("{}", rn);
        check_node_state([10, 20, 25, 35, 40, 50, 0], [1, 1, 1, 5, 4, 2, 2, 0], &rn);

        // insert range where the distance between the upper and lower pivot is
        // greater than 0
        // where the pivots to be inserted don't exist in the node
        set_node_state([10, 20, 30, 0, 0, 0, 0], [1, 1, 1, 4, 0, 0, 0, 0], &mut rn);
        assert!(rn.internal_insert(25, 35, M::Some(5), &0, &MAX) == RangeInsertState::Ok(None));
        println!("{}", rn);
        check_node_state([10, 20, 25, 35, 0, 0, 0], [1, 1, 1, 5, 4, 0, 0, 0], &rn);

        // insert range where the lower pivot equals a pivot in the node and
        // the upper doesn't
        set_node_state(
            [10, 20, 30, 40, 50, 0, 0],
            [1, 1, 1, 4, 2, 2, 0, 0],
            &mut rn,
        );
        assert!(rn.internal_insert(20, 35, M::Some(5), &0, &MAX) == RangeInsertState::Ok(None));
        println!("{}", rn);
        check_node_state([10, 20, 35, 40, 50, 0, 0], [1, 1, 5, 4, 2, 2, 0, 0], &rn);

        // insert range where the lower pivot isn't in the node and the upper pivot
        // does equal another pivot already in the node
        set_node_state(
            [10, 20, 30, 40, 50, 0, 0],
            [1, 1, 1, 4, 2, 2, 0, 0],
            &mut rn,
        );
        assert!(rn.internal_insert(15, 30, M::Some(5), &0, &MAX) == RangeInsertState::Ok(None));
        println!("{}", rn);
        check_node_state([10, 15, 30, 40, 50, 0, 0], [1, 1, 5, 4, 2, 2, 0, 0], &rn);

        set_node_state(
            [10, 20, 30, 40, 50, 0, 0],
            [1, 1, 1, 4, 2, 2, 0, 0],
            &mut rn,
        );
        assert!(rn.internal_insert(5, 55, M::Some(5), &0, &MAX) == RangeInsertState::Ok(None));
        println!("{}", rn);
        check_node_state([5, 55, 0, 0, 0, 0, 0], [1, 5, 2, 0, 0, 0, 0, 0], &rn);

        set_node_state([0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], &mut rn);
        assert!(rn.internal_insert(5, 55, M::Some(5), &0, &MAX) == RangeInsertState::Ok(Some(1)));
        println!("{}", rn);
        check_node_state([5, 55, 0, 0, 0, 0, 0], [1, 5, 1, 0, 0, 0, 0, 0], &rn);

        set_node_state([10, 20, 30, 40, 0, 0, 0], [1, 2, 3, 4, 0, 0, 0, 0], &mut rn);
        assert!(rn.internal_insert(60, 80, M::Some(5), &0, &MAX) == RangeInsertState::Ok(None));
        println!("{}", rn);
        check_node_state([10, 20, 30, 40, 60, 80, 0], [1, 2, 3, 4, 0, 5, 0, 0], &rn);
    }

    #[test]
    fn test_range_node_find_ranges_to_condense() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();

        //pivot = min [5, 10, 20, 30, 40, 50 ,60] max
        //value =   [1,  1,  1,  4,  2,  2,  2,  8]
        set_node_state(
            [5, 10, 20, 30, 40, 50, 60],
            [1, 1, 1, 4, 2, 2, 2, 8],
            &mut rn,
        );

        let found_ranges = rn.find_ranges_to_condense();
        let mut test_range: Vec<InternalPivots> = Vec::new();
        test_range.push(InternalPivots {
            lower: 0,
            upper: 2,
            lower_is_min: false,
            upper_is_max: false,
        });
        test_range.push(InternalPivots {
            lower: 4,
            upper: 6,
            lower_is_min: false,
            upper_is_max: false,
        });
        assert!(found_ranges == test_range);
    }

    #[test]
    fn test_range_node_compact_node() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();
        let max = std::usize::MAX;

        //pivot = min [5, 10, 20, 30, 40, 50 ,60] max
        //value =   [1,  1,  1,  4,  2,  2,  2,  8]
        set_node_state(
            [5, 10, 20, 30, 40, 50, 60],
            [1, 1, 1, 4, 2, 2, 2, 8],
            &mut rn,
        );

        // after condense shold be
        //pivot = min [20, 30 ,60, max, X, X, X] max
        //value =   [1,  4,  2,  8, X, X, X, X]

        assert!(rn.compact_node(&max) == true);
        print!("{}", rn);
        check_node_state([20, 30, 60, 0, 0, 0, 0], [1, 4, 2, 8, 0, 0, 0, 0], &rn);

        //pivot = min [5, 10, 20, 30, 40, 50 ,60] max
        //value =   [1,  1,  1,  4,  2,  2,  2,  2]
        set_node_state(
            [5, 10, 20, 30, 40, 50, 60],
            [1, 1, 1, 4, 2, 2, 2, 2],
            &mut rn,
        );

        // after condense shold be
        //pivot = min [20, 30 ,max, X, X, X, X] max
        //value =   [1,  4,  2, X, X, X, X, X]
        assert!(rn.compact_node(&max) == true);
        println!("{}", rn);
        check_node_state([20, 30, 0, 0, 0, 0, 0], [1, 4, 2, 0, 0, 0, 0, 0], &rn);

        //pivot = min [5, 10, 20, 30, 40, 50 ,60] max
        //value =   [1, 1, 1, 1, 1, 1, 1, 1]
        set_node_state(
            [5, 10, 20, 30, 40, 50, 60],
            [1, 1, 1, 1, 1, 1, 1, 1],
            &mut rn,
        );

        // after condense shold be
        //pivot = min [max, X, X, X, X, X] max
        //value =   [1,  X,  X,  X, X, X, X, X]
        assert!(rn.compact_node(&max) == true);
        println!("{}", rn);
        check_node_state([0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], &rn);

        //pivot = min [5, 10, 20, 30, 40, 50 ,60] max
        //value =   [3, 4,  5,  6, 7,  8,  1,  1]
        set_node_state(
            [5, 10, 20, 30, 40, 50, 60],
            [3, 4, 5, 6, 7, 8, 1, 1],
            &mut rn,
        );

        // after condense should be
        //pivot = min [5, 10, 20, 30, 40, 50, max] max
        //value =   [3, 4,  5,  6, 7,  8,  1,  X]
        assert!(rn.compact_node(&max) == true);
        println!("{}", rn);
        check_node_state([5, 10, 20, 30, 40, 50, 0], [3, 4, 5, 6, 7, 8, 1, 0], &rn);
    }

    #[test]
    fn test_range_node_insert_range() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();
        let min = 0;
        let max = std::usize::MAX;

        assert!(rn.insert_range(10, 20, 1, &min, &max) == RangeInsertState::Ok(None));
        print!("{}", rn);
        check_node_state([10, 20, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(15, 17, 6, &min, &max) == RangeInsertState::Ok(Some(1)));
        print!("{}", rn);
        check_node_state([10, 15, 17, 20, 0, 0, 0], [0, 1, 6, 1, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(20, 40, 7, &min, &max) == RangeInsertState::Ok(None));
        print!("{}", rn);
        check_node_state([10, 15, 17, 20, 40, 0, 0], [0, 1, 6, 1, 7, 0, 0, 0], &rn);

        assert!(rn.insert_range(10, 40, 3, &min, &max) == RangeInsertState::Ok(None));
        print!("{}", rn);
        check_node_state([10, 40, 0, 0, 0, 0, 0], [0, 3, 0, 0, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(7, 8, 7, &min, &max) == RangeInsertState::Ok(None));
        print!("{}", rn);
        check_node_state([7, 8, 10, 40, 0, 0, 0], [0, 7, 0, 3, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(0, 7, 2, &min, &max) == RangeInsertState::Ok(None));
        print!("{}", rn);
        check_node_state([7, 8, 10, 40, 0, 0, 0], [2, 7, 0, 3, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(5, 7, 10, &min, &max) == RangeInsertState::Ok(Some(2)));
        print!("{}", rn);
        check_node_state([5, 7, 8, 10, 40, 0, 0], [2, 10, 7, 0, 3, 0, 0, 0], &rn);

        assert!(rn.insert_range(5, 7, 10, &min, &max) == RangeInsertState::Ok(Some(10)));
        print!("{}", rn);
        check_node_state([5, 7, 8, 10, 40, 0, 0], [2, 10, 7, 0, 3, 0, 0, 0], &rn);

        assert!(rn.insert_range(9, 30, 15, &min, &max) == RangeInsertState::Ok(None));
        print!("{}", rn);
        check_node_state([5, 7, 8, 9, 30, 40, 0], [2, 10, 7, 0, 15, 3, 0, 0], &rn);

        assert!(rn.insert_range(40, 60, 5, &min, &max) == RangeInsertState::Ok(None));
        print!("{}", rn);
        check_node_state([5, 7, 8, 9, 30, 40, 60], [2, 10, 7, 0, 15, 3, 5, 0], &rn);

        assert!(
            rn.insert_range(60, 70, 5, &min, &max)
                == RangeInsertState::Err(
                    60,
                    70,
                    M::Some(5),
                    InternalPivots {
                        lower: 6,
                        upper: 0,
                        lower_is_min: false,
                        upper_is_max: true
                    }
                )
        );
        let result = rn.insert_range(60, max, 9, &min, &max);
        println!("{:?}", result);
        assert!(result == RangeInsertState::Ok(None));
        print!("{}", rn);
        check_node_state([5, 7, 8, 9, 30, 40, 60], [2, 10, 7, 0, 15, 3, 5, 9], &rn);

        // get new rangeLeaf to clear state
        rn = RangeLeaf::new();
        assert!(rn.insert_range(50, max, 1, &min, &max) == RangeInsertState::Ok(None));
        print!("{}", rn);
        check_node_state([50, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(50, 75, 2, &min, &max) == RangeInsertState::Ok(Some(1)));
        print!("{}", rn);
        check_node_state([50, 75, 0, 0, 0, 0, 0], [0, 2, 1, 0, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(20, 40, 3, &min, &max) == RangeInsertState::Ok(None));
        print!("{}", rn);
        check_node_state([20, 40, 50, 75, 0, 0, 0], [0, 3, 0, 2, 1, 0, 0, 0], &rn);

        set_node_state(
            [10, 20, 30, 40, 50, 60, 0],
            [1, 2, 3, 4, 5, 6, 7, 0],
            &mut rn,
        );

        assert!(rn.insert_range(70, max, 8, &min, &max) == RangeInsertState::Ok(Some(7)));
        println!("{}", rn);
        check_node_state([10, 20, 30, 40, 50, 60, 70], [1, 2, 3, 4, 5, 6, 7, 8], &rn);

        set_node_state(
            [10, 20, 30, 40, 50, 60, 0],
            [1, 2, 3, 4, 5, 6, 7, 0],
            &mut rn,
        );

        assert!(rn.insert_range(30, 35, 8, &min, &max) == RangeInsertState::Ok(Some(4)));
        println!("{}", rn);
        check_node_state([10, 20, 30, 35, 40, 50, 60], [1, 2, 3, 8, 4, 5, 6, 7], &rn);
    }

    #[test]
    fn test_range_node_find_internal_pivots() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();

        assert!(
            rn.find_internal_pivots(&3, &4)
                == (
                    InternalPivots {
                        lower: 0,
                        upper: 0,
                        lower_is_min: true,
                        upper_is_max: false
                    },
                    true
                )
        );

        assert!(
            rn.find_internal_pivots(&10, &20)
                == (
                    InternalPivots {
                        lower: 0,
                        upper: 0,
                        lower_is_min: true,
                        upper_is_max: false
                    },
                    true
                )
        );
        set_node_state([7, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], &mut rn);

        assert!(
            rn.find_internal_pivots(&4, &6)
                == (
                    InternalPivots {
                        lower: 0,
                        upper: 0,
                        lower_is_min: true,
                        upper_is_max: false
                    },
                    false
                )
        );

        set_node_state(
            [7, 11, 20, 25, 35, 69, 85],
            [0, 0, 0, 0, 0, 0, 0, 0],
            &mut rn,
        );

        assert!(
            rn.find_internal_pivots(&20, &25)
                == (
                    InternalPivots {
                        lower: 2,
                        upper: 3,
                        lower_is_min: false,
                        upper_is_max: false
                    },
                    false
                )
        );
        assert!(
            rn.find_internal_pivots(&3, &4)
                == (InternalPivots {
                    lower: 0,
                    upper: 0,
                    lower_is_min: true,
                    upper_is_max: false
                }, false);
        );
        assert!(
            rn.find_internal_pivots(&5, &17)
                == (InternalPivots {
                    lower: 0,
                    upper: 2,
                    lower_is_min: true,
                    upper_is_max: false
                }, false);
        );
        assert!(
            rn.find_internal_pivots(&7, &25)
                == (
                    InternalPivots {
                        lower: 0,
                        upper: 3,
                        lower_is_min: false,
                        upper_is_max: false
                    },
                    false
                )
        );
        assert!(
            rn.find_internal_pivots(&3, &100)
                == (
                    InternalPivots {
                        lower: 0,
                        upper: 0,
                        lower_is_min: true,
                        upper_is_max: true
                    },
                    false
                )
        );
        assert!(
            rn.find_internal_pivots(&90, &95)
                == (
                    InternalPivots {
                        lower: 6,
                        upper: 0,
                        lower_is_min: false,
                        upper_is_max: true
                    },
                    true
                )
        );
        assert!(
            rn.find_internal_pivots(&20, &95)
                == (
                    InternalPivots {
                        lower: 2,
                        upper: 0,
                        lower_is_min: false,
                        upper_is_max: true
                    },
                    false
                )
        );
    }

    #[test]
    fn test_range_node_move_pivots_left() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();
        let max: usize = std::usize::MAX;

        set_node_state([1, 2, 6, 7, 17, 19, 0], [3, 5, 23, 5, 2, 0, 0, 0], &mut rn);
        assert!(rn.move_pivots_left(3, 3, &max) == true);
        check_node_state([7, 17, 19, 0, 0, 0, 0], [5, 2, 0, 0, 0, 0, 0, 0], &rn);

        assert!(rn.move_pivots_left(3, 4, &max) == false);

        assert!(rn.move_pivots_left(10, 3, &max) == false);

        assert!(rn.move_pivots_left(12, 15, &max) == false);

        assert!(rn.move_pivots_left(0, 1, &max) == false);

        assert!(rn.move_pivots_left(1, 1, &max) == true);
        check_node_state([17, 19, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0], &rn);

        set_node_state(
            [5, 10, 20, 30, 40, 50, 60],
            [1, 1, 1, 4, 2, 2, 2, 2],
            &mut rn,
        );
        assert!(rn.move_pivots_left(6, 3, &max) == true);
        println!("{}", rn);
        check_node_state([5, 10, 20, 60, 0, 0, 0], [1, 1, 1, 2, 2, 0, 0, 0], &rn);

        set_node_state(
            [5, 10, 20, 30, 40, 50, 60],
            [1, 1, 1, 4, 2, 2, 2, 2],
            &mut rn,
        );
        assert!(rn.move_pivots_left(5, 3, &max) == true);
        println!("{}", rn);
        check_node_state([5, 10, 50, 60, 0, 0, 0], [1, 1, 2, 2, 2, 0, 0, 0], &rn);

        assert!(rn.move_pivots_left(2, 2, &max) == true);
        println!("{}", rn);
        check_node_state([50, 60, 0, 0, 0, 0, 0], [2, 2, 2, 0, 0, 0, 0, 0], &rn);
    }

    // since RangeNode::move_pivots_right
    // is used in RangeNode::insert we will test this function by
    // inserting the pivots and values manually
    #[test]
    fn test_range_node_move_pivots_right() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();

        set_node_state([1, 2, 6, 7, 17, 0, 0], [1, 2, 3, 4, 5, 0, 0, 0], &mut rn);
        assert!(rn.move_pivots_right(3, 2) == true);
        println!("{}", rn);
        check_node_state([1, 2, 6, 0, 0, 7, 17], [1, 2, 3, 4, 0, 0, 5, 0], &rn);

        set_node_state([1, 2, 6, 7, 17, 0, 0], [1, 2, 3, 4, 5, 0, 0, 0], &mut rn);
        assert!(rn.move_pivots_right(3, 4) == false);

        set_node_state(
            [10, 20, 30, 40, 50, 0, 0],
            [1, 1, 1, 4, 2, 2, 0, 0],
            &mut rn,
        );
        assert!(rn.move_pivots_right(0, 1) == true);
        println!("{}", rn);
        check_node_state([0, 10, 20, 30, 40, 50, 0], [1, 0, 1, 1, 4, 2, 2, 0], &rn);

        set_node_state(
            [10, 20, 30, 40, 50, 60, 70],
            [1, 1, 1, 4, 2, 2, 0, 0],
            &mut rn,
        );
        assert!(rn.move_pivots_right(0, 1) == false);
    }

    /*
    #[test]
    fn test_range_node_find_valid_internal_pivots() {
        let mut internal_pivots: InternalPivots;
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();
        let max: usize = 80;

        set_node_state([10, 20, 30, 40, 0, 0, 0], [1, 2, 3, 4, 0, 0, 0, 0], &mut rn);

        //test 1
        let (internal_pivots_option, _) = rn.find_valid_internal_pivots(&40, &70);
        assert!(internal_pivots.lower == 3 && internal_pivots.upper == 4);

        // test 2
        let (internal_pivots_option, _) = rn.find_valid_internal_pivots(&30, &70);
        assert!(internal_pivots_option.is_some());
        internal_pivots = internal_pivots_option.unwrap();
        assert!(internal_pivots.lower == 2 && internal_pivots.upper == 3);

        //test 3
        let (internal_pivots_option, _) = rn.find_valid_internal_pivots(&60, &70);
        assert!(internal_pivots_option.is_none());

        //test 4
        let (internal_pivots_option, _) = rn.find_valid_internal_pivots(&40, &max);
        assert!(internal_pivots_option.is_none());

        //test 5
        let (internal_pivots_option, _) = rn.find_valid_internal_pivots(&30, &max);
        assert!(internal_pivots_option.is_some());
        internal_pivots = internal_pivots_option.unwrap();
        assert!(internal_pivots.lower == 2 && internal_pivots.upper == 3);
    }
    */

    #[test]
    fn test_range_node_search() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();
        let min: usize = 5;
        let max: usize = 80;

        set_node_state(
            [10, 20, 30, 40, 50, 60, 70],
            [1, 2, 3, 4, 5, 6, 7, 8],
            &mut rn,
        );
        assert!(rn.search(&5, &min, &max) == Some(1));
        assert!(rn.search(&80, &min, &max) == None);
        assert!(rn.search(&200, &min, &max) == None);

        set_node_state(
            [10, 20, 30, 40, 50, 0, 0],
            [1, 2, 0, 4, 5, 0, 0, 0],
            &mut rn,
        );
        assert!(rn.search(&55, &min, &max) == None);
        assert!(rn.search(&2, &min, &max) == None);
        assert!(rn.search(&25, &min, &max) == None);
    }

    #[test]
    fn test_range_node_delete_range() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();
        let min: usize = 0;
        let max: usize = MAX;

        // set state for first test return value
        let insert_state = RangeInsertState::<usize, usize>::Ok(None);

        // case 1
        set_node_state(
            [10, 20, 30, 40, 50, 60, 70],
            [1, 2, 3, 4, 5, 6, 7, 8],
            &mut rn,
        );
        println!("case 1");
        assert!(rn.delete_range(min, max, &min, &max) == insert_state);
        check_node_state([0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], &rn);

        // case 2 a
        set_node_state(
            [10, 20, 30, 40, 50, 60, 70],
            [1, 2, 3, 4, 5, 6, 7, 8],
            &mut rn,
        );
        println!("case 2a");
        assert!(rn.delete_range(min, 40, &min, &max) == insert_state);
        println!("{}", rn);
        check_node_state([40, 50, 60, 70, 0, 0, 0], [0, 5, 6, 7, 8, 0, 0, 0], &rn);

        // case 2 b
        set_node_state(
            [10, 20, 30, 40, 50, 60, 70],
            [1, 2, 3, 4, 5, 6, 7, 8],
            &mut rn,
        );
        println!("case 2b");
        assert!(rn.delete_range(min, 45, &min, &max) == insert_state);
        check_node_state([45, 50, 60, 70, 0, 0, 0], [0, 5, 6, 7, 8, 0, 0, 0], &rn);

        // case 3 a
        set_node_state(
            [10, 20, 30, 40, 50, 60, 70],
            [1, 2, 3, 4, 5, 6, 7, 8],
            &mut rn,
        );
        println!("case 3a");
        assert!(rn.delete_range(40, max, &min, &max) == insert_state);
        check_node_state([10, 20, 30, 40, 0, 0, 0], [1, 2, 3, 4, 0, 0, 0, 0], &rn);

        // case 3 b
        set_node_state(
            [10, 20, 30, 40, 50, 60, 70],
            [1, 2, 3, 4, 5, 6, 7, 8],
            &mut rn,
        );
        println!("case 3b");
        assert!(rn.delete_range(45, max, &min, &max) == insert_state);
        check_node_state([10, 20, 30, 40, 45, 0, 0], [1, 2, 3, 4, 5, 0, 0, 0], &rn);

        // case 4a
        set_node_state(
            [10, 20, 30, 40, 50, 60, 70],
            [1, 2, 3, 4, 5, 6, 7, 8],
            &mut rn,
        );
        println!("case 4a");
        assert!(rn.delete_range(10, 75, &min, &max) == insert_state);
        check_node_state([10, 75, 0, 0, 0, 0, 0], [1, 0, 8, 0, 0, 0, 0, 0], &rn);

        // case 4a.2
        set_node_state(
            [10, 20, 30, 40, 50, 60, 70],
            [1, 2, 3, 4, 5, 6, 7, 8],
            &mut rn,
        );
        assert!(rn.delete_range(10, 20, &min, &max) == insert_state);
        println!("{}", rn);
        check_node_state([10, 20, 30, 40, 50, 60, 70], [1, 0, 3, 4, 5, 6, 7, 8], &rn);

        // case 4b
        set_node_state(
            [10, 20, 30, 40, 50, 60, 70],
            [1, 2, 3, 4, 5, 6, 7, 8],
            &mut rn,
        );
        println!("case 4b");
        assert!(rn.delete_range(5, 70, &min, &max) == insert_state);
        print!("{}", rn);
        check_node_state([5, 70, 0, 0, 0, 0, 0], [1, 0, 8, 0, 0, 0, 0, 0], &rn);

        // case 4c
        set_node_state(
            [10, 20, 30, 40, 50, 60, 70],
            [1, 2, 3, 4, 5, 6, 7, 8],
            &mut rn,
        );
        println!("case 4c");
        assert!(rn.delete_range(15, 35, &min, &max) == insert_state);
        print!("{}", rn);
        check_node_state([10, 15, 35, 40, 50, 60, 70], [1, 2, 0, 4, 5, 6, 7, 8], &rn);

        // case 4d
        println!("case 4d");
        assert!(rn.delete_range(40, 60, &min, &max) == insert_state);
        check_node_state([10, 15, 35, 40, 60, 70, 0], [1, 2, 0, 4, 0, 7, 8, 0], &rn);

        // case 4e
        set_node_state(
            [10, 20, 30, 40, 50, 60, 70],
            [1, 2, 3, 4, 5, 6, 7, 8],
            &mut rn,
        );
        println!("case 4e");
        assert!(rn.delete_range(15, 40, &min, &max) == insert_state);
        println!("{}", rn);
        check_node_state([10, 15, 40, 50, 60, 70, 0], [1, 2, 0, 5, 6, 7, 8, 0], &rn);

        // case 4f
        set_node_state(
            [10, 20, 30, 40, 50, 60, 70],
            [1, 2, 3, 4, 5, 6, 7, 8],
            &mut rn,
        );
        println!("case 4f");
        assert!(rn.delete_range(10, 35, &min, &max) == insert_state);
        println!("{}", rn);
        check_node_state([10, 35, 40, 50, 60, 70, 0], [1, 0, 4, 5, 6, 7, 8, 0], &rn);

        // invalid range 1
        set_node_state([10, 20, 30, 40, 0, 0, 0], [1, 2, 3, 4, 0, 0, 0, 0], &mut rn);
        assert!(rn.delete_range(60, 80, &min, &max) == insert_state);
        println!("{}", rn);
        check_node_state([10, 20, 30, 40, 0, 0, 0], [1, 2, 3, 4, 0, 0, 0, 0], &rn);

        // invalid range 2
        // solve this!!!!
        set_node_state([10, 20, 30, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0, 0, 0], &mut rn);
        println!("invalid range 2 ");
        assert!(rn.delete_range(30, 60, &min, &max) == insert_state);
        println!("{}", rn);
        check_node_state([10, 20, 30, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0, 0, 0], &rn);

        // valid range special case
        set_node_state([10, 20, 30, 40, 0, 0, 0], [1, 2, 3, 4, 5, 0, 0, 0], &mut rn);
        assert!(rn.delete_range(60, 80, &min, &max) == insert_state);
        println!("{}", rn);
        check_node_state([10, 20, 30, 40, 60, 80, 0], [1, 2, 3, 4, 5, 0, 5, 0], &rn);
    }

    #[cfg(test)]
    fn set_node_state(
        pivots: [usize; R_CAPACITY],
        values: [usize; CAPACITY],
        node: &mut RangeLeaf<usize, usize>,
    ) {
        for i in 0..R_CAPACITY {
            if pivots[i] == 0 {
                node.pivot[i] = M::None;
            } else {
                node.pivot[i] = M::Some(pivots[i]);
            }
        }

        for i in 0..CAPACITY {
            if values[i] == 0 {
                node.value[i] = M::None;
            } else {
                node.value[i] = M::Some(values[i]);
            }
        }
    }

    #[cfg(test)]
    fn check_node_state(
        pivots: [usize; R_CAPACITY],
        values: [usize; CAPACITY],
        node: &RangeLeaf<usize, usize>,
    ) {
        println!("check_node_state() output:");
        print!("pivot = [");
        for i in 0..R_CAPACITY {
            if pivots[i] == 0 {
                match node.pivot[i] {
                    M::Some(p) => print!("{}, ", p),
                    _ => print!("X, "),
                }
                assert!(M::None == node.pivot[i]);
            } else {
                match node.pivot[i] {
                    M::Some(p) => print!("{}, ", p),
                    _ => print!("X, "),
                }
                assert!(M::Some(pivots[i]) == node.pivot[i]);
            }
        }

        print!("]\nvalue = [");
        for i in 0..CAPACITY {
            if values[i] == 0 {
                print!("X, ");
                assert!(M::None == node.value[i]);
            } else {
                match node.value[i] {
                    M::Some(p) => print!("{}, ", p),
                    _ => print!(", "),
                }

                assert!(M::Some(values[i]) == node.value[i]);
            }
        }
        print!("]\n\n");
    }
}
