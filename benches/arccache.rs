use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use function_name::named;
use rand::distributions::uniform::SampleUniform;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::time::{Duration, Instant};
// use uuid::Uuid;

use concread::arcache::ARCache;
use criterion::measurement::{Measurement, ValueFormatter};

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
    csize: usize,
    hit_count: u32,
    attempt: u32,
    hit_pct: f64,
}

enum AccessPattern<T>
where
    T: SampleUniform + PartialOrd + Clone,
{
    Random(T, T),
}

impl<T> AccessPattern<T>
where
    T: SampleUniform + PartialOrd + Clone,
{
    fn next(&self) -> T {
        match self {
            AccessPattern::Random(min, max) => {
                let mut rng = thread_rng();
                rng.gen_range(min.clone()..max.clone())
            }
        }
    }
}

pub struct HitPercentageFormatter;

impl ValueFormatter for HitPercentageFormatter {
    fn format_value(&self, value: f64) -> String {
        // eprintln!("⚠️  format_value -> {:?}", value);
        format!("{}%", value)
    }

    fn format_throughput(&self, _throughput: &Throughput, value: f64) -> String {
        // eprintln!("⚠️  format_throughput -> {:?}", value);
        format!("{}%", value)
    }

    fn scale_values(&self, typical_value: f64, values: &mut [f64]) -> &'static str {
        // eprintln!("⚠️  scale_values -> typ {:?} : {:?}", typical_value, values);
        // panic!();
        "%"
    }

    fn scale_throughputs(
        &self,
        _typical_value: f64,
        _throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        // eprintln!("⚠️  scale_throughputs -> {:?}", values);
        "%"
    }

    fn scale_for_machines(&self, values: &mut [f64]) -> &'static str {
        // eprintln!("⚠️  scale_machines -> {:?}", values);
        "%"
    }
}

pub struct HitPercentage;

impl Measurement for HitPercentage {
    type Intermediate = (u32, u32);
    type Value = (u32, u32);

    fn start(&self) -> Self::Intermediate {
        unreachable!("HitPercentage requires the use of iter_custom");
    }

    fn end(&self, i: Self::Intermediate) -> Self::Value {
        // eprintln!("⚠️  end -> {:?}", i);
        i
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        // eprintln!("⚠️  add -> {:?} + {:?}", v1, v2);
        (v1.0 + v2.0, v1.1 + v2.1)
    }

    fn zero(&self) -> Self::Value {
        // eprintln!("⚠️  zero -> (0,0)");
        (0, 0)
    }

    fn to_f64(&self, val: &Self::Value) -> f64 {
        let x = (f64::from(val.0) / f64::from(val.1)) * 100.0;
        // eprintln!("⚠️  to_f64 -> {:?} -> {:?}", val, x);
        x
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &HitPercentageFormatter
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
    let mut attempt = 0;

    for _i in 0..warm_iters {
        let k = access_pattern.next();
        let v = backing_set.get(&k).cloned().unwrap();

        let mut wr_txn = arc.write();
        // hit/miss process.
        if !wr_txn.contains_key(&k) {
            wr_txn.insert(k, v);
        }
        wr_txn.commit();
    }

    for _i in 0..iters {
        attempt += 1;
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
        attempt,
        hit_pct,
    }
}

macro_rules! basic_single_thread_x_small_latency {
    ($c:expr, $max:expr, $measure:expr) => {
        let mut group = $c.benchmark_group(function_name!());
        for pct in &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110] {
            group.bench_with_input(BenchmarkId::from_parameter(pct), &$max, |b, &max| {
                b.iter_custom(|iters| {
                    let mut backing_set: HashMap<usize, usize> = HashMap::with_capacity(max);
                    (0..$max).for_each(|i| {
                        backing_set.insert(i, i);
                    });
                    let data = run_single_thread_test(
                        iters,
                        iters / 10,
                        *pct,
                        backing_set,
                        Some(Duration::from_nanos(5)),
                        AccessPattern::Random(0, max),
                    );
                    println!("{:?}", data);
                    data.elapsed
                })
            });
        }
        group.finish();
    };
}

macro_rules! basic_single_thread_x_small_pct {
    ($c:expr, $max:expr, $measure:expr) => {
        let mut group = $c.benchmark_group(function_name!());
        for pct in &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110] {
            group.bench_with_input(BenchmarkId::from_parameter(pct), &$max, |b, &max| {
                b.iter_custom(|iters| {
                    let mut backing_set: HashMap<usize, usize> = HashMap::with_capacity(max);
                    (0..$max).for_each(|i| {
                        backing_set.insert(i, i);
                    });
                    let data = run_single_thread_test(
                        iters,
                        iters / 10,
                        *pct,
                        backing_set,
                        Some(Duration::from_nanos(5)),
                        AccessPattern::Random(0, max),
                    );
                    println!("{:?}", data);
                    (data.hit_count, data.attempt)
                })
            });
        }
        group.finish();
    };
}

#[named]
pub fn basic_single_thread_2048_small_latency(c: &mut Criterion) {
    basic_single_thread_x_small_latency!(c, 2048, MeasureType::Latency);
}

#[named]
pub fn basic_single_thread_2048_small_pct(c: &mut Criterion<HitPercentage>) {
    basic_single_thread_x_small_pct!(c, 2048, MeasureType::HitPct);
}

criterion_group!(
    name = latency;
    config = Criterion::default()
        // .measurement_time(Duration::from_secs(15))
        .with_plots();
    targets = basic_single_thread_2048_small_latency
);

criterion_group!(
    name = hit_percent;
    config = Criterion::default().with_measurement(HitPercentage);
    targets = basic_single_thread_2048_small_pct
);

criterion_main!(latency, hit_percent);
