// The benchmarks aim to only measure times of the operations in their names.
// That's why all use Bencher::iter_batched which enables non-benchmarked
// preparation before running the measured function.
// Insert (which doesn't completely avoid updates, but makes them unlikely) and
// remove both have benchmarks with empty values and with custom structs of 42
// 64-bit integers.
// (as a sidenote, the performance really differs; in the case of remove, the
// remove function itself returns original value - the benchmark doesn't use
// this value, but performance is significantly worse - about twice on my
// machine - than the empty value remove; it might be interesting to see if a
// remove function returning void would perform better, ie. if the returns
// are optimized - omitted in this case).
// The counts of inserted/removed elements are chosen at random from constant
// ranges in an attempt to avoid a single count performing better because of
// specific HW features of computers the code is benchmarked with.

extern crate criterion;
extern crate rand;
extern crate concread;

use criterion::{BatchSize, criterion_group, criterion_main, Criterion};
use rand::{Rng, thread_rng};
use concread::hashmap::*;
use std::mem;

// ranges of counts for different benchmarks:
const INSERT_COUNT : (usize, usize) = (120, 140);
const INSERT_COUNT_FOR_REMOVE : (usize, usize) = (340, 360);
const REMOVE_COUNT : (usize, usize) = (120, 140);

pub fn insert_empty_value(c: &mut Criterion) {
    c.bench_function("insert_empty_value", |b| b.iter_batched(
        || {
            let mut rng = thread_rng();
            let count = rng.gen_range(INSERT_COUNT.0, INSERT_COUNT.1);
            let mut list = Vec::with_capacity(count);
            for _ in 0..count {
                list.push(rng.gen_range(0, INSERT_COUNT.1<<8) as u32)
            }
            (HashMap::new(), list)
        },
        |mut data| insert_vec(&mut data),
        BatchSize::SmallInput
    ));
}

pub fn insert_struct_value(c: &mut Criterion) {
    c.bench_function("insert_struct_value", |b| b.iter_batched(
        || {
            let mut rng = thread_rng();
            let count = rng.gen_range(INSERT_COUNT.0, INSERT_COUNT.1);
            let mut list = Vec::with_capacity(count);
            for _ in 0..count {
                list.push((rng.gen_range(0, INSERT_COUNT.1<<8) as u32, default_struct()))
            }
            (HashMap::<u32, Struct>::new(), list)
        },
        |mut data| insert_struct_vec(&mut data),
        BatchSize::SmallInput
    ));
}

pub fn remove_empty_value(c: &mut Criterion) {
    c.bench_function("remove_empty_value", |b| b.iter_batched(
        || prepare_remove(()),
        |mut data| remove_vec(&mut data),
        BatchSize::SmallInput
    ));
}

pub fn remove_struct_value_no_read(c: &mut Criterion) {
    c.bench_function("remove_struct_value_no_read", |b| b.iter_batched(
        || prepare_remove(Struct::default()),
        |mut data| remove_vec(&mut data),
        BatchSize::SmallInput
    ));
}

criterion_group!(insert, insert_empty_value, insert_struct_value);
criterion_group!(remove, remove_empty_value, remove_struct_value_no_read);
criterion_main!(insert, remove);


fn insert_vec(pair: &mut (HashMap<u32, ()>, Vec<u32>)) {
    let mut write_txn = pair.0.write();
    for i in pair.1.iter() {
        write_txn.insert(*i, ());
    }
}

fn insert_struct_vec(pair: &mut (HashMap<u32, Struct>, Vec<(u32, StructValue)>)) {
    let mut write_txn = pair.0.write();
    for i in pair.1.iter_mut() {
        write_txn.insert(i.0, mem::replace(&mut i.1, None).unwrap());
    }
}

fn remove_vec<V: Clone>(pair: &mut (HashMap<u32, V>, Vec<u32>)) {
    let mut write_txn = pair.0.write();
    for i in pair.1.iter() {
        write_txn.remove(i);
    }
}


type StructValue = Option<Struct>;

#[derive(Default, Clone)]
struct Struct {
    var1: i64, var2: i64, var3: i64, var4: i64, var5: i64, var6: i64, var7: i64, var8: i64,
    var9: i64, var10: i64, var11: i64, var12: i64, var13: i64, var14: i64, var15: i64, var16: i64,
    var17: i64, var18: i64, var19: i64, var20: i64, var21: i64, var22: i64, var23: i64, var24: i64,
    var25: i64, var26: i64, var27: i64, var28: i64, var29: i64, var30: i64, var31: i64, var32: i64,
    var33: i64, var34: i64, var35: i64, var36: i64, var37: i64, var38: i64, var39: i64, var40: i64,
    var41: i64, var42: i64
}

fn default_struct() -> StructValue {
    Some(Default::default())
}

/// Prepares a remove benchmark with values in the HashMap being clones of the 'values' parameter
fn prepare_remove<V: Clone>(values: V) -> (HashMap<u32, V>, Vec<u32>) {
    let mut rng = thread_rng();
    let insert_count = rng.gen_range(INSERT_COUNT_FOR_REMOVE.0, INSERT_COUNT_FOR_REMOVE.1);
    let remove_count = rng.gen_range(REMOVE_COUNT.0, REMOVE_COUNT.1);
    let map = HashMap::new();
    let mut write_txn = map.write();
    for i in random_order(insert_count, insert_count).iter() {
        // We could count on the hash function alone to make the order random, but it seems
        // better to count on every possible implementation.
        write_txn.insert(i.clone(), values.clone());
    }
    write_txn.commit();
    (map, random_order(insert_count, remove_count))
}

/// Returns a Vec of n elements from the range [0,up_to) in random order without repetition
fn random_order(up_to: usize, n: usize) -> Vec<u32> {
    let mut rng = thread_rng();
    let mut order = Vec::with_capacity(n);
    let mut generated = vec![false; up_to];
    let mut remaining = n;
    let mut remaining_elems = up_to;
    while remaining > 0 {
        let mut r = rng.gen_range(0, remaining_elems);
        // find the r-th yet nongenerated number:
        for i in 0..up_to {
            if generated[i] { continue; }
            if r == 0 {
                order.push(i as u32);
                generated[i] = true;
                break;
            }
            r -= 1;
        }
        remaining -= 1;
        remaining_elems -= 1;
    }
    order
}
