use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::time::{Duration, Instant};

use concread::arcache::ARCache;

/*
 * A fixed dataset size, with various % of cache pressure (5, 10, 20, 40, 60, 80, 110)
 * ^
 * \--- then vary the dataset sizes. Inclusions are always processed here.
 * -- vary the miss time/penalty as well?
 *
 * -- this measures time to complete.
 *
 * As above but could we measure hit rate as well?
 *
 */

#[derive(Debug)]
struct DataPoint {
    elapsed: Duration,
    hit_count: usize,
}

fn run_single_thread_test<K, V>(
    // Number of iterations
    iters: u64,
    // Number of iters to warm the cache.
    warm_iters: u64,
    // pct of backing set size to configure into the cache.
    cache_size_pct: u64,
    // backing set.
    backing_set: HashMap<K, V>,
    // backing set access delay on miss
    backing_set_delay: Option<Duration>,
    // How to lookup keys during each iter.
    // access_pattern: (),
) -> DataPoint
where
    K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static,
    V: Clone + Debug + Sync + Send + 'static,
{
    println!(
        "iters, size, pct -> {:?}, {:?}, {}",
        iters,
        backing_set.len(),
        cache_size_pct
    );

    let mut csize = (backing_set.len() / 100) * (cache_size_pct as usize);
    if csize == 0 {
        csize = 1;
    }

    let arc: ARCache<K, V> = ARCache::new_size_watermark(csize, 0, 0);

    let mut elapsed = Duration::from_secs(0);
    let mut hit_count = 0;

    for _i in 0..iters {
        let start = Instant::now();
        let mut wr_txn = arc.write();
        // hit/miss process.
        hit_count += 1;
        wr_txn.commit();
        elapsed = elapsed.checked_add(start.elapsed()).unwrap();
    }

    DataPoint { elapsed, hit_count }
}

pub fn basic_single_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_single_thread");
    for size in &[100, 200] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_custom(|iters| {
                let backing_set: HashMap<usize, usize> = HashMap::new();
                let data = run_single_thread_test(iters, iters / 10, 10, backing_set, None);
                println!("{:?}", data);
                data.elapsed
            })
        });
    }
    group.finish();
}

criterion_group!(
    name = basic;
    config = Criterion::default()
        // .measurement_time(Duration::from_secs(15))
        .with_plots();
    targets = basic_single_thread
);

criterion_main!(basic);
