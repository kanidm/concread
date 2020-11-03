use criterion::{BatchSize, criterion_group, criterion_main, Criterion};
use rand::{Rng, thread_rng};
use concread::hashmap::*;

// ranges of counts for different benchmarks:
const INSERT_COUNT : (u32, u32) = (120, 140);

pub fn insert_empty_value(c: &mut Criterion) {
    c.bench_function("insert_empty_value", |b| b.iter_batched(
        || {
            let mut rng = thread_rng();
            let count = rng.gen_range(INSERT_COUNT.0, INSERT_COUNT.1);
            let mut list = Vec::with_capacity(count);
            for _ in 0..count {
                list.push((rng.gen_range(0, INSERT_COUNT.1<<8), ()))
            }
            (HashMap::new().write(), list)
        },
        |mut data| insert_vec(&mut data),
        BatchSize::SmallInput
    ));
}
/*
pub fn insert_struct_value(c: &mut Criterion) {

}

pub fn remove_empty_value(c: &mut Criterion) {

}

pub fn remove_struct_value_no_read(c: &mut Criterion) {

}*/

criterion_group!(insert, insert_empty_value);//, insert_struct_value);
//criterion_group!(remove, remove_empty_value, remove_struct_value_no_read);
criterion_main!(insert);
//criterion_main!(insert, remove);


fn insert_vec<V>(pair: &mut (HashMapWriteTxn<u32, V>, Vec<(u32, V)>)) {
    for i in pair.1.iter() {
        pair.0.insert(i.0, i.1);
    }
}

