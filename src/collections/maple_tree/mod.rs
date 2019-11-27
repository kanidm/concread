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
pub enum NodeTag<K, V> {
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
pub enum FoundRange {
    Min(usize),           // this range goes from min to the index at K
    Max(usize),           // this range goes from the index K to max
    MinMax,               // this range goes from min to max
    Pivots(usize, usize), // this range is from the pivots
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

    // TODO: remove this function and in move_pivots_right just count down from R_CAPACITY-1 untill
    // there isn't a None or untill the number of places to move has been reached and use that to
    // decide if the operation will succede or not
    // returns the number of used pivots
    pub fn len(&mut self) -> usize {
        let mut count: usize = 0;

        for pivot in self.pivot.iter() {
            match pivot {
                M::Some(_) => count += 1,
                // since the node is sorted with M::None's all being at the end of the node
                // when we encounter M::None there are never anymore pivots
                M::None => break,
            }
        }

        count
    }

    // loops over the values linearly to find if there are any contiguous values
    // that are the same. This is the first step in condensing a RangeLeaf node.
    //
    // for all sets of contiguous values that are the same, the start index of the pivot
    // for the first value and, the end pivot for the last value are added to an ops::Range
    // variable and pushed onto the foundRanges vector which is then returned after all
    // values have been checked
    pub fn find_ranges_to_condense(&mut self) -> Vec<ops::Range<usize>> {
        let mut rangeStartIndex = 0;
        let mut matched = false;
        let mut foundRanges: Vec<ops::Range<usize>> = Vec::new();

        {
            let mut previousVal: &M<V> = &self.value[0];

            for i in 1..CAPACITY {
                if &self.value[i] == previousVal {
                    if !matched {
                        matched = true;
                        rangeStartIndex = i - 1;
                    }
                } else if matched {
                    foundRanges.push(ops::Range {
                        start: rangeStartIndex,
                        end: i - 1,
                    });
                    matched = false;
                }

                previousVal = &self.value[i];
            }
        }

        if matched {
            foundRanges.push(ops::Range {
                start: rangeStartIndex,
                end: R_CAPACITY,
            });
        }

        foundRanges
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
        let mut foundRanges: Vec<ops::Range<usize>>;

        foundRanges = self.find_ranges_to_condense();
        if foundRanges.is_empty() {
            return false;
        }

        let mut lastRangeInUse = self.value[CAPACITY - 1] != M::None;
        let mut lastValue: M<V> = M::None;;
        let mut pivotIndexToMove = 0;
        let mut distance = 0;

        for range in foundRanges.iter().rev() {
            distance = range.end - range.start;
            pivotIndexToMove = range.end;

            // when the last pivot in the range to be compacted is located at R_CAPACITY
            // we know that this is the max value, we must copy the max value
            // into the last position in the array and then treat this as the last pivot of
            // the range
            if range.end == R_CAPACITY {
                self.pivot[R_CAPACITY - 1] = M::Some(max.clone());
                self.value[CAPACITY - 1] = M::None;

                distance -= 1;
                pivotIndexToMove = range.end - 1;

                lastRangeInUse = false;
            }

            // if the last range is used we have to save the value at the
            // end of the value's array so we can copy it to the pos
            if lastRangeInUse {
                lastValue = self.value[CAPACITY - 1].clone();
            }

            self.move_pivots_left(pivotIndexToMove, distance, max);
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
     */

    // returns true on successfull delete of range or on failure returns false
    pub fn delete_range(&mut self, lowerPivot: K, upperPivot: K, min: &K, max: &K) -> bool {
        let mut range = self.find_range(&lowerPivot, &upperPivot);

        let (insertLowerPivot, insertUpperPivot) =
            self.check_if_pivots_need_insert(&range, &lowerPivot, &upperPivot, min, max);

        // case 1
        // removing range lowerPivot..upperPivot(leaves the node empty)
        if &lowerPivot == min && &upperPivot == max {
            self.clear_node();
            return true;
        }
        // case 2
        else if &lowerPivot == min {
            // when upperPivot is not equal to the pivot at range.end, insert the
            // upperPivot at range.end-1
            if insertUpperPivot {
                if self.move_pivots_left(range.end as usize, range.end as usize - 1, max) {
                    self.pivot[0] = M::Some(upperPivot);
                    self.value[0] = M::None;
                    return true;
                }
            } else if self.move_pivots_left(range.end as usize, range.end as usize, max) {
                self.value[0] = M::None;
                return true;
            }

            return false;
        }
        // case 3
        else if &upperPivot == max {
            let mut rangeStart = range.start as usize + 1;
            if insertLowerPivot {
                self.pivot[rangeStart] = M::Some(lowerPivot);
                rangeStart += 1;
            }

            // set each pivot after the the first in the range to M::None
            for i in rangeStart..R_CAPACITY {
                self.pivot[i] = M::None;
            }

            for i in rangeStart..CAPACITY {
                self.value[i] = M::None;
            }

            return true;
        }
        // case 4
        else {
            let includesMax = range.end == R_CAPACITY as isize;
            let distanceBetweenPivots = if range.start != -1 {
                range.end - range.start - 1
            } else {
                range.end
            };

            let mut pivotInsertCount = 0;

            if insertLowerPivot {
                pivotInsertCount += 1;
                range.start += 1;
            }
            if insertUpperPivot {
                pivotInsertCount += 1;
            }

            if !self.make_space_for_insert(
                distanceBetweenPivots,
                pivotInsertCount,
                range.end,
                includesMax,
                max,
                insertLowerPivot,
                insertUpperPivot,
            ) {
                return false;
            }

            let mut end = self.calculate_end_pivot_index(
                distanceBetweenPivots,
                pivotInsertCount,
                range.end,
                includesMax,
                insertUpperPivot,
            );

            if !includesMax{
               end -= 1; 
            }

            if insertLowerPivot {
                self.pivot[range.start as usize] = M::Some(lowerPivot);
            }
            if insertUpperPivot {
                self.pivot[end] = M::Some(upperPivot);
            }

            self.value[range.start as usize+1] = M::None;

            println!("distanceBetweenPivots = {}", distanceBetweenPivots);
            println!("pivotInsertCount = {}", pivotInsertCount);
            println!("end = {}", end);

            return true;
        }

        false
    }

    // checks if the upperPivot and lowerPivot need to be inserted into the node then
    // returns a tuple containing two boolean values (insertLowerPivot, insertUpperPivot)
    fn check_if_pivots_need_insert(
        &mut self,
        range: &ops::Range<isize>,
        lowerPivot: &K,
        upperPivot: &K,
        min: &K,
        max: &K,
    ) -> (bool, bool) {
        let mut insertUpperPivot = false;
        let mut insertLowerPivot = false;

        if range.end as usize != R_CAPACITY {
            insertUpperPivot = match &self.pivot[range.end as usize] {
                M::Some(p) => upperPivot != p,
                M::None => false,
            };
        } else {
            insertUpperPivot = upperPivot != max;
        }

        if range.start > -1 {
            insertLowerPivot = match &self.pivot[range.start as usize] {
                M::Some(p) => lowerPivot != p,
                M::None => false,
            };
        } else {
            insertLowerPivot = lowerPivot != min;
        }

        return (insertLowerPivot, insertUpperPivot);
    }

    // inserts the range:value pair lowerPivot..upperPivot and value into the node,
    // returns false if the node is full even if compacting will allow the insert
    // it is the cursors job too try compacting the node on failure
    //
    // returns true if the range is successfully inserted or false on faliure to insert
    pub fn insert_range(
        &mut self,
        lowerPivot: K,
        upperPivot: K,
        value: V,
        min: &K,
        max: &K,
    ) -> bool {
        let range = self.find_range(&lowerPivot, &upperPivot);
        let mut start: usize = range.start as usize;
        let mut end: usize = range.end as usize;

        let distanceBetweenPivots = range.end - range.start - 1; //i.e adjacent pivots will make this 0
        let mut pivotInsertCount = 0;

        let mut insertLowerPivot = false;
        let mut insertUpperPivot = false;
        let mut lastPivotIsNone = false;

        let includesMin = range.start == -1;
        let includesMax = range.end == R_CAPACITY as isize;

        //discover if the end of the range is a valid value or M::None
        if includesMax == false {
            lastPivotIsNone = match &self.pivot[range.end as usize] {
                M::Some(p) => false,
                M::None => true,
            };
        }

        if includesMin {
            insertLowerPivot = &lowerPivot != min;
        } else {
            insertLowerPivot = match &self.pivot[range.start as usize] {
                M::Some(b1) => &lowerPivot != b1,
                M::None => false,
            };
        }

        if includesMax {
            // need to make room for upper pivot to be inserted but
            // the node is full since
            if &upperPivot != max {
                return false;
            }
        } else {
            insertUpperPivot = match &self.pivot[range.end as usize] {
                M::Some(b2) => &upperPivot != b2,
                M::None => false,
            };
        }

        if insertLowerPivot {
            start = (range.start + 1) as usize;
            pivotInsertCount += 1;
        } else if !includesMin {
            start = range.start as usize;
        }
        if insertUpperPivot {
            pivotInsertCount += 1;
        }

        if !self.make_space_for_insert(
            distanceBetweenPivots,
            pivotInsertCount,
            range.end,
            includesMax,
            max,
            insertLowerPivot,
            insertUpperPivot,
        ) {
            return false;
        }

        end = self.calculate_end_pivot_index(
            distanceBetweenPivots,
            pivotInsertCount,
            range.end,
            includesMax,
            insertUpperPivot,
        );

        if !insertLowerPivot && !insertUpperPivot && !lastPivotIsNone {
            self.value[end] = M::Some(value);
            return true;
        }

        if insertLowerPivot {
            self.pivot[start] = M::Some(lowerPivot);
        }
        if insertUpperPivot && !lastPivotIsNone || lastPivotIsNone {
            self.pivot[end] = M::Some(upperPivot);
        }

        if insertUpperPivot && insertLowerPivot && distanceBetweenPivots == 0 {
            self.value[start] = self.value[end + 1].clone();
        } else if insertLowerPivot && !insertUpperPivot && distanceBetweenPivots == 0 {
            self.value[start] = self.value[end].clone();
        }

        self.value[end] = M::Some(value);

        true
    }

    // returns the correct end pivot index within the pivot array
    // for the range that is to be inserted
    pub fn calculate_end_pivot_index(
        &mut self,
        distanceBetweenPivots: isize,
        pivotInsertCount: usize,
        rangeEndPivotIndex: isize,
        includesMax: bool,
        insertUpperPivot: bool,
    ) -> usize {
        let mut end: usize;
        let leftShiftAmount: isize = distanceBetweenPivots as isize - pivotInsertCount as isize;

        if distanceBetweenPivots != 0 {
            if leftShiftAmount == -1 {
                end = rangeEndPivotIndex as usize;
            } else if includesMax {
                end = R_CAPACITY - 1 - leftShiftAmount as usize;
            } else {
                end = rangeEndPivotIndex as usize - leftShiftAmount as usize;
            }
        } else {
            end = rangeEndPivotIndex as usize + pivotInsertCount as usize;

            if insertUpperPivot {
                end -= 1;
            }
        }

        end
    }

    // if possible moves the pivots and values within the rangeNode to accomodate
    // the new pivot/s and value to be inserted
    //
    // returns false if there is not enough space to make room for the new pivots
    // otherwise returns true on success
    pub fn make_space_for_insert(
        &mut self,
        distanceBetweenPivots: isize,
        pivotInsertCount: usize,
        rangeEnd: isize,
        includesMax: bool,
        max: &K,
        insertLowerPivot: bool,
        insertUpperPivot: bool,
    ) -> bool {
        // start and end pivots of the range have pivots inbetween
        if distanceBetweenPivots != 0 {
            let leftShiftAmount: isize = distanceBetweenPivots as isize - pivotInsertCount as isize;
            
            // pivotInsertCount can only ever be 2 so if distanceBetweenPivots is 1
            // leftShift amount could be -1 so we need to right shift by 1
            if leftShiftAmount == -1 {
                if self.move_pivots_right(rangeEnd as usize, 1) == false {
                    return false;
                }
            } else if includesMax {
                println!("{}", leftShiftAmount);
                if self.move_pivots_left(R_CAPACITY, leftShiftAmount as usize, max) == false {
                    return false;
                }
            } else {
                if self.move_pivots_left(rangeEnd as usize, leftShiftAmount as usize, max) == false
                {
                    return false;
                }
            }
        }
        // pivots are adjacent
        else {
            if includesMax && (insertLowerPivot || insertUpperPivot) {
                return false;
            }
            if self.move_pivots_right(rangeEnd as usize, pivotInsertCount as usize) == false {
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
    //       which would return ops::Range{start: -1, end: 5} as the indexes
    //
    // Range A is the range to be inserted i.e rangeLower..rangeUpper
    // Range B is the current range being assesed within the RangeLeaf node
    // Range C is the range that came before B Note: C's UpperBound is B's lowerBound
    //
    // rangeNumbers     -   [ R1  | R2 | R3 | R4 | R5 | R6 | R7 |  R8 ]
    // Pivots & min max - [min | P1 | P2 | P3 | P4 | P5 | P6 | P7 | max]
    //
    // this function assumes the following to be true:
    //      rangeLower >= min
    //      rangeUpper <= max
    //
    // The algorithm bellow uses the following basic steps:
    //  1. Find where !(A > B) (This lets us know where the range A starts at)
    //      since a_1 < b_2 and we know a_1 > c_2 since C comes before B
    //  2. Check if (A \subset C) in which case we know that the Range is C
    //      return Range here if found
    //  3. Find where A < D where D is a range that is after B
    //
    pub fn find_range(&mut self, rangeLower: &K, rangeUpper: &K) -> ops::Range<isize> {
        // range number of B
        // i.e range number 1 [min, P1)
        let mut rangeB: usize = 0;
        let mut index: isize = 0; // used to keep the index from the for loops
        let mut set = false;
        let mut isNone = false;

        // find range B  where !(A > B)
        // since we know that for all index's up to the current value of i
        // rangeLower > self.pivot[i] when rangeLower < self.pivot[i] we know
        // that the range starts at the (i-1)th index
        for i in 0..R_CAPACITY {
            index = i as isize;
            match &self.pivot[i] {
                M::Some(b2) => {
                    if rangeLower < b2 {
                        rangeB = i + 1;
                        set = true;
                        break;
                    }
                }
                M::None => {
                    isNone = true;
                    break;
                }
            }
        }

        // if we didn't find any pivot where rangeLower < self.pivot[i]
        // then we either ran into a None value or since we don't check min or max
        // the range must be between pivots at the index's R_CAPACITY and R_CAPCITY-1
        if !set {
            if isNone {
                return ops::Range {
                    start: (index - 1) as isize,
                    end: index as isize,
                };
            } else {
                return ops::Range {
                    start: (R_CAPACITY - 1) as isize,
                    end: R_CAPACITY as isize,
                };
            }
        }

        // now we know that the range starts at index rangeB-1 and we need to find
        // where the range ends.
        for i in (rangeB - 1)..R_CAPACITY {
            match &self.pivot[i] {
                M::Some(b2) => {
                    if rangeUpper <= b2 {
                        if rangeB == 1 {
                            return ops::Range {
                                start: -1,
                                end: i as isize,
                            };
                        }
                        return ops::Range {
                            start: (rangeB - 2) as isize,
                            end: i as isize,
                        };
                    }
                }
                M::None => {
                    return ops::Range {
                        start: (rangeB - 2) as isize,
                        end: i as isize,
                    };
                }
            }
        }

        if rangeB == 1 {
            return ops::Range {
                start: -1,
                end: R_CAPACITY as isize,
            };
        } else {
            return ops::Range {
                start: (rangeB - 2) as isize,
                end: R_CAPACITY as isize,
            };
        }
    }

    pub fn move_pivots_left(&mut self, mut pivotIndex: usize, mut distance: usize, max: &K) -> bool {
        let mut lastRangeInUse = false;

        if pivotIndex > R_CAPACITY || pivotIndex < 0{
            return false;
        }
        else if distance == 0{
            return true;
        }

        // the new index for the pivot at pivotIndex is pivotIndex-distance
        // make sure this is not negative
        if (pivotIndex as isize) - (distance as isize) < 0 {
            return false;
        }

        // to shift the max value left, first copy the pivot into the
        // last place in the array and the value in the second last place
        // in the array
        if pivotIndex == R_CAPACITY {
            let mut tmpVal = M::None;
            mem::swap(&mut tmpVal, &mut self.value[CAPACITY - 1]);
            mem::swap(&mut tmpVal, &mut self.value[CAPACITY - 2]);

            self.pivot[R_CAPACITY - 1] = M::Some(max.clone());
            pivotIndex -= 1;
            distance -= 1;
        }

        // befor we move the values check if the last range between the end pivot
        // and max is in use so we know if we need to copy max into one of the pivots later
        if self.value[CAPACITY - 1] != M::None {
            lastRangeInUse = true;
        }

        let mut pivotHolder: M<K> = M::None;
        let mut valueHolder: M<V> = M::None;

        // move all of the pivots to the left
        for i in pivotIndex..R_CAPACITY {
            mem::swap(&mut self.pivot[i], &mut pivotHolder);
            mem::swap(&mut self.pivot[i - distance], &mut pivotHolder);
            pivotHolder = M::None;
        }

        // move all of the values to the left
        for i in pivotIndex..CAPACITY {
            mem::swap(&mut self.value[i], &mut valueHolder);
            mem::swap(&mut self.value[i - distance], &mut valueHolder);
            valueHolder = M::None;
        }

        // set any pivots and values to M::None that should have
        // in some cases there will be some pivots and values that arne't set
        // to M::None when they should be, this happens when the distance that the pivots are being
        // shifted is larger than the amount of pivots between the pivotIndex and the end of the
        // array including the pivot at pivotIndex
        let placesToEnd = R_CAPACITY - pivotIndex;
        if distance > placesToEnd {
            for i in R_CAPACITY - distance..R_CAPACITY {
                self.pivot[i] = M::None;
                self.value[i + 1] = M::None;
            }
        }

        if lastRangeInUse {
            self.pivot[R_CAPACITY - distance] = M::Some(max.clone());
        }
        return true;
    }

    // moves the pivot/s at pivotIndex and above to the right x distance where x = distance variable
    // if the move would push values off the end then false is returned
    pub fn move_pivots_right(&mut self, pivotIndex: usize, distance: usize) -> bool {
        if distance == 0 {
            return true;
        }
        let freePivots = R_CAPACITY - self.len();

        // make sure there is enough room to move the pivots into
        if freePivots < distance {
            return false;
        }
        // make sure the pivot is greater than 0 and less than the index of the
        // last pivot as moving that would push it out of the array
        else if pivotIndex < 0 || pivotIndex >= R_CAPACITY - 1 {
            return false;
        } else if pivotIndex + distance > R_CAPACITY {
            return false;
        }

        // must use a variable to hold the pivots and values between swaps as we can't have two
        // mutable borrows of the same array at the same time :(
        let mut pivotHolder: M<K> = M::None;
        let mut valueHolder: M<V> = M::None;
        let lastIndex = (R_CAPACITY) - freePivots;

        // starts at lastPivotIndex and goes until reaching the pivot at pivotIndex
        // so moving pivots and values from righ to left so each pivot and value is moved
        // into a free place in the arrays
        for i in 0..lastIndex + 1 {
            if lastIndex - i >= pivotIndex {
                match &self.pivot[lastIndex - i] {
                    M::None => {
                        continue;
                    }
                    M::Some(_) => {
                        mem::swap(&mut pivotHolder, &mut self.pivot[lastIndex - i]);
                        mem::swap(&mut pivotHolder, &mut self.pivot[lastIndex - i + distance]);

                        mem::swap(&mut valueHolder, &mut self.value[lastIndex - i]);
                        mem::swap(&mut valueHolder, &mut self.value[lastIndex - i + distance]);
                    }
                }
            }
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

    // returns the number of keys in a node
    // should we store a lenth value on a node instead of calculating it on the fly?
    pub fn get_len(&self) -> usize {
        let mut size: usize = 0;
        for key in self.key.iter() {
            match key {
                M::Some(_) => size += 1,
                M::None => continue,
            }
        }

        size
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
    pub fn check_bounds(&mut self, minBound: &K, maxBound: &K) -> bool {
        let min = self.get_min();
        let max = self.get_max();

        // if either min or max is None they must both be None
        // if they are both None then the Node MUST be empty and
        // we can return true
        if min == None && max == None {
            return true;
        }

        if min >= Some(&minBound) && max <= Some(&maxBound) {
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
        for current_index in 0..CAPACITY {
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
    use super::RangeLeaf;
    use super::SparseLeaf;
    use super::M;
    use collections::maple_tree::CAPACITY;
    use collections::maple_tree::R_CAPACITY;
    use std::ops;

    #[test]
    fn test_sparse_leaf_get_len() {
        let mut sl: SparseLeaf<usize, usize> = SparseLeaf::new();

        assert!(sl.get_len() == 0);

        let test_vals: [usize; 8] = [3, 8, 7, 4, 2, 1, 5, 6];

        for val in test_vals.iter() {
            sl.insert(*val, *val);
        }

        assert!(sl.get_len() == 8);

        let del_vals: [usize; 4] = [3, 8, 2, 1];
        for val in del_vals.iter() {
            sl.remove(val);
        }

        assert!(sl.get_len() == 4);
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
        let slEmpty: SparseLeaf<usize, usize> = SparseLeaf::new();
        assert!(slEmpty.get_max() == None);
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
        let slEmpty: SparseLeaf<usize, usize> = SparseLeaf::new();
        assert!(slEmpty.get_min() == None);
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
    fn test_range_node_find_ranges_to_condense() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();

        //pivot = min [5, 10, 20, 30, 40, 50 ,60] max
        //value =   [1,  1,  1,  4,  2,  2,  2,  8]
        rn.pivot[0] = M::Some(5);
        rn.pivot[1] = M::Some(10);
        rn.pivot[2] = M::Some(20);
        rn.pivot[3] = M::Some(30);
        rn.pivot[4] = M::Some(40);
        rn.pivot[5] = M::Some(50);
        rn.pivot[6] = M::Some(60);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(1);
        rn.value[2] = M::Some(1);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(2);
        rn.value[5] = M::Some(2);
        rn.value[6] = M::Some(2);
        rn.value[7] = M::Some(8);

        let foundRanges = rn.find_ranges_to_condense();
        let mut testRange: Vec<ops::Range<usize>> = Vec::new();
        testRange.push(ops::Range { start: 0, end: 2 });
        testRange.push(ops::Range { start: 4, end: 6 });
        assert!(foundRanges == testRange);
    }

    #[test]
    fn test_range_node_compact_node() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();
        let min = 0;
        let max = std::usize::MAX;

        //pivot = min [5, 10, 20, 30, 40, 50 ,60] max
        //value =   [1,  1,  1,  4,  2,  2,  2,  8]
        rn.pivot[0] = M::Some(5);
        rn.pivot[1] = M::Some(10);
        rn.pivot[2] = M::Some(20);
        rn.pivot[3] = M::Some(30);
        rn.pivot[4] = M::Some(40);
        rn.pivot[5] = M::Some(50);
        rn.pivot[6] = M::Some(60);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(1);
        rn.value[2] = M::Some(1);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(2);
        rn.value[5] = M::Some(2);
        rn.value[6] = M::Some(2);
        rn.value[7] = M::Some(8);

        // after condense shold be
        //pivot = min [20, 30 ,60, max, X, X, X] max
        //value =   [1,  4,  2,  8, X, X, X, X]

        assert!(rn.compact_node(&max) == true);
        println!("{}", rn);
        check_node_state([20, 30, 60, max, 0, 0, 0], [1, 4, 2, 8, 0, 0, 0, 0], &rn);

        //pivot = min [5, 10, 20, 30, 40, 50 ,60] max
        //value =   [1,  1,  1,  4,  2,  2,  2,  2]
        rn.pivot[0] = M::Some(5);
        rn.pivot[1] = M::Some(10);
        rn.pivot[2] = M::Some(20);
        rn.pivot[3] = M::Some(30);
        rn.pivot[4] = M::Some(40);
        rn.pivot[5] = M::Some(50);
        rn.pivot[6] = M::Some(60);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(1);
        rn.value[2] = M::Some(1);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(2);
        rn.value[5] = M::Some(2);
        rn.value[6] = M::Some(2);
        rn.value[7] = M::Some(2);

        // after condense shold be
        //pivot = min [20, 30 ,max, X, X, X, X] max
        //value =   [1,  4,  2, X, X, X, X, X]
        assert!(rn.compact_node(&max) == true);
        println!("{}", rn);
        check_node_state([20, 30, max, 0, 0, 0, 0], [1, 4, 2, 0, 0, 0, 0, 0], &rn);

        //pivot = min [5, 10, 20, 30, 40, 50 ,60] max
        //value =   [1, 1, 1, 1, 1, 1, 1, 1]
        rn.pivot[0] = M::Some(5);
        rn.pivot[1] = M::Some(10);
        rn.pivot[2] = M::Some(20);
        rn.pivot[3] = M::Some(30);
        rn.pivot[4] = M::Some(40);
        rn.pivot[5] = M::Some(50);
        rn.pivot[6] = M::Some(60);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(1);
        rn.value[2] = M::Some(1);
        rn.value[3] = M::Some(1);
        rn.value[4] = M::Some(1);
        rn.value[5] = M::Some(1);
        rn.value[6] = M::Some(1);
        rn.value[7] = M::Some(1);

        // after condense shold be
        //pivot = min [max, X, X, X, X, X] max
        //value =   [1,  X,  X,  X, X, X, X, X]
        assert!(rn.compact_node(&max) == true);
        println!("{}", rn);
        check_node_state([max, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], &rn);

        //pivot = min [5, 10, 20, 30, 40, 50 ,60] max
        //value =   [3, 4,  5,  6, 7,  8,  1,  1]
        rn.pivot[0] = M::Some(5);
        rn.pivot[1] = M::Some(10);
        rn.pivot[2] = M::Some(20);
        rn.pivot[3] = M::Some(30);
        rn.pivot[4] = M::Some(40);
        rn.pivot[5] = M::Some(50);
        rn.pivot[6] = M::Some(60);

        rn.value[0] = M::Some(3);
        rn.value[1] = M::Some(4);
        rn.value[2] = M::Some(5);
        rn.value[3] = M::Some(6);
        rn.value[4] = M::Some(7);
        rn.value[5] = M::Some(8);
        rn.value[6] = M::Some(1);
        rn.value[7] = M::Some(1);

        // after condense should be
        //pivot = min [5, 10, 20, 30, 40, 50, max] max
        //value =   [3, 4,  5,  6, 7,  8,  1,  X]
        assert!(rn.compact_node(&max) == true);
        println!("{}", rn);
        check_node_state([5, 10, 20, 30, 40, 50, max], [3, 4, 5, 6, 7, 8, 1, 0], &rn);
    }

    #[test]
    fn test_range_node_insert_range() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();
        let min = 0;
        let max = std::usize::MAX;

        assert!(rn.insert_range(10, 20, 1, &min, &max) == true);
        print!("{}", rn);
        check_node_state([10, 20, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(15, 17, 6, &min, &max) == true);
        print!("{}", rn);
        check_node_state([10, 15, 17, 20, 0, 0, 0], [0, 1, 6, 1, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(20, 40, 7, &min, &max) == true);
        print!("{}", rn);
        check_node_state([10, 15, 17, 20, 40, 0, 0], [0, 1, 6, 1, 7, 0, 0, 0], &rn);

        assert!(rn.insert_range(10, 40, 3, &min, &max) == true);
        print!("{}", rn);
        check_node_state([10, 40, 0, 0, 0, 0, 0], [0, 3, 0, 0, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(7, 8, 7, &min, &max) == true);
        print!("{}", rn);
        check_node_state([7, 8, 10, 40, 0, 0, 0], [0, 7, 0, 3, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(0, 7, 2, &min, &max) == true);
        print!("{}", rn);
        check_node_state([7, 8, 10, 40, 0, 0, 0], [2, 7, 0, 3, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(5, 7, 10, &min, &max) == true);
        print!("{}", rn);
        check_node_state([5, 7, 8, 10, 40, 0, 0], [2, 10, 7, 0, 3, 0, 0, 0], &rn);

        assert!(rn.insert_range(5, 7, 10, &min, &max) == true);
        print!("{}", rn);
        check_node_state([5, 7, 8, 10, 40, 0, 0], [2, 10, 7, 0, 3, 0, 0, 0], &rn);

        assert!(rn.insert_range(9, 30, 15, &min, &max) == true);
        print!("{}", rn);
        check_node_state([5, 7, 8, 9, 30, 40, 0], [2, 10, 7, 0, 15, 3, 0, 0], &rn);

        assert!(rn.insert_range(40, 60, 5, &min, &max) == true);
        print!("{}", rn);
        check_node_state([5, 7, 8, 9, 30, 40, 60], [2, 10, 7, 0, 15, 3, 5, 0], &rn);

        assert!(rn.insert_range(60, 70, 5, &min, &max) == false);

        assert!(rn.insert_range(60, max, 9, &min, &max) == true);
        print!("{}", rn);
        check_node_state([5, 7, 8, 9, 30, 40, 60], [2, 10, 7, 0, 15, 3, 5, 9], &rn);

        // get new rangeLeaf to clear state
        rn = RangeLeaf::new();
        assert!(rn.insert_range(50, max, 1, &min, &max) == true);
        print!("{}", rn);
        check_node_state([50, max, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(50, 75, 2, &min, &max) == true);
        print!("{}", rn);
        check_node_state([50, 75, max, 0, 0, 0, 0], [0, 2, 1, 0, 0, 0, 0, 0], &rn);

        assert!(rn.insert_range(20, 40, 3, &min, &max) == true);
        print!("{}", rn);
        check_node_state([20, 40, 50, 75, max, 0, 0], [0, 3, 0, 2, 1, 0, 0, 0], &rn);

        /*
         * test what happens when we insert a range x..Max when i.e
         * pivot = [x, Max, 0, 0, 0, 0, 0] what happens when the node
         * becomes
         * pivot = [p1, p2, p3. p4, p5, x, MAX] and we are inserting a
         * range that needes to shift over one? the node will say its full
         * but [x, MAX] can be shifted up one and make space
         */
    }

    #[test]
    fn test_range_node_find_range() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();
        let min: usize = 3;
        let max: usize = 100;

        //println!("{:?}", rn.find_range(&3, &4));
        assert!(rn.find_range(&3, &4) == ops::Range { start: -1, end: 0 });

        assert!(rn.find_range(&10, &20) == ops::Range { start: -1, end: 0 });
        rn.pivot[0] = M::Some(7);

        assert!(rn.find_range(&4, &6) == ops::Range { start: -1, end: 0 });

        rn.pivot[1] = M::Some(11);
        rn.pivot[2] = M::Some(20);
        rn.pivot[3] = M::Some(25);
        rn.pivot[4] = M::Some(35);
        rn.pivot[5] = M::Some(69);
        rn.pivot[6] = M::Some(85);

        assert!(rn.find_range(&20, &25) == ops::Range { start: 2, end: 3 });
        assert!(rn.find_range(&3, &4) == ops::Range { start: -1, end: 0 });
        assert!(rn.find_range(&5, &17) == ops::Range { start: -1, end: 2 });
        assert!(rn.find_range(&7, &25) == ops::Range { start: 0, end: 3 });
        assert!(
            rn.find_range(&3, &100)
                == ops::Range {
                    start: -1,
                    end: R_CAPACITY as isize
                }
        );
        assert!(
            rn.find_range(&90, &95)
                == ops::Range {
                    start: 6,
                    end: R_CAPACITY as isize
                }
        );
        assert!(
            rn.find_range(&20, &95)
                == ops::Range {
                    start: 2,
                    end: R_CAPACITY as isize
                }
        );
    }

    #[test]
    fn test_range_node_move_pivots_left() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();
        let max: usize = std::usize::MAX;
        rn.pivot[0] = M::Some(1);
        rn.value[0] = M::Some(3);

        rn.pivot[1] = M::Some(2);
        rn.value[1] = M::Some(5);

        rn.pivot[2] = M::Some(6);
        rn.value[2] = M::Some(23);

        rn.pivot[3] = M::Some(7);
        rn.value[3] = M::Some(5);

        rn.pivot[4] = M::Some(17);
        rn.value[4] = M::Some(2);

        rn.pivot[5] = M::Some(19);
        rn.pivot[6] = M::None;

        assert!(rn.move_pivots_left(3, 3, &max) == true);
        check_node_state([7, 17, 19, 0, 0, 0, 0], [5, 2, 0, 0, 0, 0, 0, 0], &rn);

        assert!(rn.move_pivots_left(3, 4, &max) == false);

        assert!(rn.move_pivots_left(10, 3, &max) == false);

        assert!(rn.move_pivots_left(12, 15, &max) == false);

        assert!(rn.move_pivots_left(0, 1, &max) == false);

        assert!(rn.move_pivots_left(1, 1, &max) == true);
        check_node_state([17, 19, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0], &rn);

        rn.pivot[0] = M::Some(5);
        rn.pivot[1] = M::Some(10);
        rn.pivot[2] = M::Some(20);
        rn.pivot[3] = M::Some(30);
        rn.pivot[4] = M::Some(40);
        rn.pivot[5] = M::Some(50);
        rn.pivot[6] = M::Some(60);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(1);
        rn.value[2] = M::Some(1);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(2);
        rn.value[5] = M::Some(2);
        rn.value[6] = M::Some(2);
        rn.value[7] = M::Some(2);

        assert!(rn.move_pivots_left(6, 3, &max) == true);
        println!("{}", rn);
        check_node_state([5, 10, 20, 60, max, 0, 0], [1, 1, 1, 2, 2, 0, 0, 0], &rn);

        rn.pivot[0] = M::Some(5);
        rn.pivot[1] = M::Some(10);
        rn.pivot[2] = M::Some(20);
        rn.pivot[3] = M::Some(30);
        rn.pivot[4] = M::Some(40);
        rn.pivot[5] = M::Some(50);
        rn.pivot[6] = M::Some(60);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(1);
        rn.value[2] = M::Some(1);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(2);
        rn.value[5] = M::Some(2);
        rn.value[6] = M::Some(2);
        rn.value[7] = M::Some(2);

        assert!(rn.move_pivots_left(5, 3, &max) == true);
        println!("{}", rn);
        check_node_state([5, 10, 50, 60, max, 0, 0], [1, 1, 2, 2, 2, 0, 0, 0], &rn);

        assert!(rn.move_pivots_left(2, 2, &max) == true);
        println!("{}", rn);
        check_node_state([50, 60, max, 0, 0, 0, 0], [2, 2, 2, 0, 0, 0, 0, 0], &rn);
    }

    // since RangeNode::move_pivots_right
    // is used in RangeNode::insert we will test this function by
    // inserting the pivots and values manually
    #[test]
    fn test_range_node_move_pivots_right() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();
        rn.pivot[0] = M::Some(6);

        assert!(rn.move_pivots_right(0, 2) == true);
        for i in 0..R_CAPACITY {
            if i == 2 {
                assert!(rn.pivot[2] == M::Some(6));
            } else {
                assert!(rn.pivot[i] == M::None);
            }
        }

        let testOne: [usize; R_CAPACITY] = [1, 2, 6, 7, 0, 17, 19];

        rn.pivot[0] = M::Some(1);
        rn.pivot[1] = M::Some(2);
        rn.pivot[2] = M::Some(6);
        rn.pivot[3] = M::Some(7);
        rn.pivot[4] = M::Some(17);
        rn.pivot[5] = M::Some(19);
        rn.pivot[6] = M::None;

        assert!(rn.move_pivots_right(4, 1) == true);

        for i in 0..R_CAPACITY {
            if testOne[i] == 0 {
                assert!(rn.pivot[i] == M::None);
            } else {
                println!("{}", i);
                assert!(rn.pivot[i] == M::Some(testOne[i]));
            }
        }

        let testTwo: [usize; R_CAPACITY] = [1, 2, 6, 7, 10, 15, 20];

        for i in 0..R_CAPACITY {
            rn.pivot[i] = M::Some(testTwo[i]);
        }

        assert!(rn.move_pivots_right(4, 1) == false);

        let testThree: [usize; R_CAPACITY] = [1, 2, 6, 7, 10, 15, 0];

        for i in 0..R_CAPACITY - 1 {
            if testThree[i] == 0 {
                rn.pivot[i] = M::None;
            } else {
                rn.pivot[i] = M::Some(testThree[i]);
            }
        }

        assert!(rn.move_pivots_right(2, 3) == false);
    }

    #[test]
    fn test_range_node_delete_range() {
        let mut rn: RangeLeaf<usize, usize> = RangeLeaf::new();
        let min: usize = 0;
        let max: usize = 80;

        rn.pivot[0] = M::Some(10);
        rn.pivot[1] = M::Some(20);
        rn.pivot[2] = M::Some(30);
        rn.pivot[3] = M::Some(40);
        rn.pivot[4] = M::Some(50);
        rn.pivot[5] = M::Some(60);
        rn.pivot[6] = M::Some(70);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(2);
        rn.value[2] = M::Some(3);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(5);
        rn.value[5] = M::Some(6);
        rn.value[6] = M::Some(7);
        rn.value[7] = M::Some(8);
        
        // case 1
        println!("case 1");
        assert!(rn.delete_range(min, max, &min, &max) == true);
        check_node_state([0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], &rn);

        rn.pivot[0] = M::Some(10);
        rn.pivot[1] = M::Some(20);
        rn.pivot[2] = M::Some(30);
        rn.pivot[3] = M::Some(40);
        rn.pivot[4] = M::Some(50);
        rn.pivot[5] = M::Some(60);
        rn.pivot[6] = M::Some(70);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(2);
        rn.value[2] = M::Some(3);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(5);
        rn.value[5] = M::Some(6);
        rn.value[6] = M::Some(7);
        rn.value[7] = M::Some(8);
        
        // case 2 a
        println!("case 2a");
        assert!(rn.delete_range(min, 40, &min, &max) == true);
        check_node_state([40, 50, 60, 70, max, 0, 0], [0, 5, 6, 7, 8, 0, 0, 0], &rn);

        rn.pivot[0] = M::Some(10);
        rn.pivot[1] = M::Some(20);
        rn.pivot[2] = M::Some(30);
        rn.pivot[3] = M::Some(40);
        rn.pivot[4] = M::Some(50);
        rn.pivot[5] = M::Some(60);
        rn.pivot[6] = M::Some(70);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(2);
        rn.value[2] = M::Some(3);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(5);
        rn.value[5] = M::Some(6);
        rn.value[6] = M::Some(7);
        rn.value[7] = M::Some(8);
        
        // case 2 b
        println!("case 2b");
        assert!(rn.delete_range(min, 45, &min, &max) == true);
        check_node_state([45, 50, 60, 70, max, 0, 0], [0, 5, 6, 7, 8, 0, 0, 0], &rn);

        rn.pivot[0] = M::Some(10);
        rn.pivot[1] = M::Some(20);
        rn.pivot[2] = M::Some(30);
        rn.pivot[3] = M::Some(40);
        rn.pivot[4] = M::Some(50);
        rn.pivot[5] = M::Some(60);
        rn.pivot[6] = M::Some(70);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(2);
        rn.value[2] = M::Some(3);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(5);
        rn.value[5] = M::Some(6);
        rn.value[6] = M::Some(7);
        rn.value[7] = M::Some(8);
        
        // case 3 a
        println!("case 3a");
        assert!(rn.delete_range(40, max, &min, &max) == true);
        check_node_state([10, 20, 30, 40, 0, 0, 0], [1, 2, 3, 4, 0, 0, 0, 0], &rn);
        
        rn.pivot[0] = M::Some(10);
        rn.pivot[1] = M::Some(20);
        rn.pivot[2] = M::Some(30);
        rn.pivot[3] = M::Some(40);
        rn.pivot[4] = M::Some(50);
        rn.pivot[5] = M::Some(60);
        rn.pivot[6] = M::Some(70);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(2);
        rn.value[2] = M::Some(3);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(5);
        rn.value[5] = M::Some(6);
        rn.value[6] = M::Some(7);
        rn.value[7] = M::Some(8);
        
        // case 3 b
        println!("case 3b");
        assert!(rn.delete_range(45, max, &min, &max) == true);
        check_node_state([10, 20, 30, 40, 45, 0, 0], [1, 2, 3, 4, 5, 0, 0, 0], &rn);

        rn.pivot[0] = M::Some(10);
        rn.pivot[1] = M::Some(20);
        rn.pivot[2] = M::Some(30);
        rn.pivot[3] = M::Some(40);
        rn.pivot[4] = M::Some(50);
        rn.pivot[5] = M::Some(60);
        rn.pivot[6] = M::Some(70);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(2);
        rn.value[2] = M::Some(3);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(5);
        rn.value[5] = M::Some(6);
        rn.value[6] = M::Some(7);
        rn.value[7] = M::Some(8);

        // case 4a
        println!("case 4a");
        assert!(rn.delete_range(10, 75, &min, &max) == true);
        check_node_state([10, 75, max, 0, 0, 0, 0], [1, 0, 8, 0, 0, 0, 0, 0], &rn);
        
        rn.pivot[0] = M::Some(10);
        rn.pivot[1] = M::Some(20);
        rn.pivot[2] = M::Some(30);
        rn.pivot[3] = M::Some(40);
        rn.pivot[4] = M::Some(50);
        rn.pivot[5] = M::Some(60);
        rn.pivot[6] = M::Some(70);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(2);
        rn.value[2] = M::Some(3);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(5);
        rn.value[5] = M::Some(6);
        rn.value[6] = M::Some(7);
        rn.value[7] = M::Some(8);

        // case 4b
        println!("case 4b");
        assert!(rn.delete_range(5, 70, &min, &max) == true);
        check_node_state([5, 70, max, 0, 0, 0, 0], [1, 0, 8, 0, 0, 0, 0, 0], &rn);


        rn.pivot[0] = M::Some(10);
        rn.pivot[1] = M::Some(20);
        rn.pivot[2] = M::Some(30);
        rn.pivot[3] = M::Some(40);
        rn.pivot[4] = M::Some(50);
        rn.pivot[5] = M::Some(60);
        rn.pivot[6] = M::Some(70);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(2);
        rn.value[2] = M::Some(3);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(5);
        rn.value[5] = M::Some(6);
        rn.value[6] = M::Some(7);
        rn.value[7] = M::Some(8);

        // case 4c
        println!("case 4c");
        assert!(rn.delete_range(15, 35, &min, &max) == true);
        check_node_state([10, 15, 35, 40, 50, 60, 70], [1, 2, 0, 4, 5, 6, 7, 8], &rn);

        // case 4d
        println!("case 4d");
        assert!(rn.delete_range(40, 60, &min, &max) == true);
        check_node_state([10, 15, 35, 40, 60, 70, max], [1, 2, 0, 4, 0, 7, 8, 0], &rn);

        rn.pivot[0] = M::Some(10);
        rn.pivot[1] = M::Some(20);
        rn.pivot[2] = M::Some(30);
        rn.pivot[3] = M::Some(40);
        rn.pivot[4] = M::Some(50);
        rn.pivot[5] = M::Some(60);
        rn.pivot[6] = M::Some(70);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(2);
        rn.value[2] = M::Some(3);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(5);
        rn.value[5] = M::Some(6);
        rn.value[6] = M::Some(7);
        rn.value[7] = M::Some(8);

        // case 4e
        println!("case 4e");
        assert!(rn.delete_range(15, 40, &min, &max) == true);
        check_node_state([10, 15, 40, 50, 60, 70, max], [1, 2, 0, 5, 6, 7, 8, 0], &rn);
 
        rn.pivot[0] = M::Some(10);
        rn.pivot[1] = M::Some(20);
        rn.pivot[2] = M::Some(30);
        rn.pivot[3] = M::Some(40);
        rn.pivot[4] = M::Some(50);
        rn.pivot[5] = M::Some(60);
        rn.pivot[6] = M::Some(70);

        rn.value[0] = M::Some(1);
        rn.value[1] = M::Some(2);
        rn.value[2] = M::Some(3);
        rn.value[3] = M::Some(4);
        rn.value[4] = M::Some(5);
        rn.value[5] = M::Some(6);
        rn.value[6] = M::Some(7);
        rn.value[7] = M::Some(8);

        // case 4f
        println!("case 4f");
        assert!(rn.delete_range(10, 35, &min, &max) == true);
        check_node_state([10, 35, 40, 50, 60, 70, max], [1, 0, 4, 5, 6, 7, 8, 0], &rn);
    }

    #[cfg(test)]
    fn check_node_state(
        pivots: [usize; R_CAPACITY],
        values: [usize; CAPACITY],
        node: &RangeLeaf<usize, usize>,
    ) {
        println!("check_node_state output");
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
