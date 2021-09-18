use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use function_name::named;
use rand::distributions::uniform::SampleUniform;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::time::{Duration, Instant};
use uuid::Uuid;

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

enum AccessPattern<T>
where
    T: SampleUniform + PartialOrd + Clone,
{
    Random(T, T),
}

#[derive(Debug)]
struct DataPoint {
    elapsed: Duration,
    csize: usize,
    hit_count: usize,
    hit_pct: f64,
}

impl<T> AccessPattern<T>
where
    T: SampleUniform + PartialOrd + Clone,
{
    fn next(&self) -> T {
        match self {
            AccessPattern::Random(min, max) => {
                let mut rng = thread_rng();
                rng.gen_range((min.clone()..max.clone()))
            }
        }
    }
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
    access_pattern: AccessPattern<K>,
) -> DataPoint
where
    K: Hash + Eq + Ord + Clone + Debug + Sync + Send + 'static + SampleUniform + PartialOrd,
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
        let k = access_pattern.next();
        let v = backing_set.get(&k).cloned().unwrap();

        let start = Instant::now();
        let mut wr_txn = arc.write();
        // hit/miss process.
        if wr_txn.contains_key(&k) {
            hit_count += 1;
        } else {
            if let Some(delay) = backing_set_delay {
                std::thread::sleep(delay);
            }
            wr_txn.insert(k, v);
        }
        wr_txn.commit();
        elapsed = elapsed.checked_add(start.elapsed()).unwrap();
    }

    let hit_pct = (f64::from(hit_count as u32) / f64::from(iters as u32)) * 100.0;
    DataPoint {
        elapsed,
        csize,
        hit_count,
        hit_pct,
    }
}

#[named]
pub fn basic_single_thread_2048_small(c: &mut Criterion) {
    let max = 2048;
    let mut group = c.benchmark_group(function_name!());
    for pct in &[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110] {
        group.bench_with_input(BenchmarkId::from_parameter(pct), &max, |b, &max| {
            b.iter_custom(|iters| {
                let mut backing_set: HashMap<usize, usize> = HashMap::with_capacity(max);
                (0..max).for_each(|i| {
                    backing_set.insert(i, i);
                });
                let data = run_single_thread_test(
                    iters,
                    iters / 10,
                    *pct,
                    backing_set,
                    None,
                    AccessPattern::Random(0, max),
                );
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
    targets = basic_single_thread_2048_small
);

criterion_main!(basic);
