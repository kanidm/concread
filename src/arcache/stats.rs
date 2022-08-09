//! Doc

use std::fmt::Debug;

/// Trait!
pub trait ARCacheWriteStat<K> {
    // RW phase trackers
    /// _
    fn cache_clear(&mut self) {}

    /// _
    fn cache_read(&mut self) {}

    /// _
    fn cache_hit(&mut self) {}

    // Commit phase trackers

    /// _
    fn include(&mut self, _k: &K) {}

    /// _
    fn include_haunted(&mut self, k: &K) {
        self.include(k)
    }

    /// _
    fn modify(&mut self, _k: &K) {}

    /// _
    fn ghost_frequent_revive(&mut self, _k: &K) {}

    /// _
    fn ghost_recent_revive(&mut self, _k: &K) {}

    /// _
    fn evict_from_recent(&mut self, _k: &K) {}

    /// _
    fn evict_from_frequent(&mut self, _k: &K) {}

    // Size of things after the operation.

    /// _
    fn p_weight(&mut self, _p: u64) {}

    /// _
    fn shared_max(&mut self, _i: u64) {}

    /// _
    fn freq(&mut self, _i: u64) {}

    /// _
    fn recent(&mut self, _i: u64) {}

    /// _
    fn all_seen_keys(&mut self, _i: u64) {}
}

/// _
pub trait ARCacheReadStat {
    /// _
    fn cache_read(&mut self) {}

    /// _
    fn cache_local_hit(&mut self) {}

    /// _
    fn cache_main_hit(&mut self) {}

    /// _
    fn include(&mut self) {}

    /// _
    fn failed_include(&mut self) {}

    /// _
    fn local_include(&mut self) {}
}

impl<K> ARCacheWriteStat<K> for () {}

impl ARCacheReadStat for () {}

#[derive(Debug)]
/// A stat collector that allows tracing the keys of items that are changed
/// during writes and quiesce phases.
pub struct TraceStat {}

impl<K> ARCacheWriteStat<K> for TraceStat
where
    K: Debug,
{
    /// _
    fn include(&mut self, k: &K) {
        tracing::trace!(?k, "include");
    }

    /// _
    fn include_haunted(&mut self, k: &K) {
        tracing::trace!(?k, "include_haunted");
    }

    /// _
    fn modify(&mut self, k: &K) {
        tracing::trace!(?k, "modify");
    }

    /// _
    fn ghost_frequent_revive(&mut self, k: &K) {
        tracing::trace!(?k, "ghost_frequent_revive");
    }

    /// _
    fn ghost_recent_revive(&mut self, k: &K) {
        tracing::trace!(?k, "ghost_recent_revive");
    }

    /// _
    fn evict_from_recent(&mut self, k: &K) {
        tracing::trace!(?k, "evict_from_recent");
    }

    /// _
    fn evict_from_frequent(&mut self, k: &K) {
        tracing::trace!(?k, "evict_from_frequent");
    }
}

/// A simple track of counters from the cache
#[derive(Debug, Default)]
pub struct WriteCountStat {
    /// The number of attempts to read from the cache
    pub read_ops: u64,
    /// The number of cache hits during this operation
    pub read_hits: u64,

    /// The current cache weight between recent and frequent.
    pub p_weight: u64,

    /// The maximum number of items in the shared cache.
    pub shared_max: u64,
    /// The number of items in the frequent set at this point in time.
    pub freq: u64,
    /// The number of items in the recent set at this point in time.
    pub recent: u64,

    /// The number of total keys seen through the cache's lifetime.
    pub all_seen_keys: u64,
}

impl<K> ARCacheWriteStat<K> for WriteCountStat {
    /// _
    fn cache_clear(&mut self) {
        self.read_ops = 0;
        self.read_hits = 0;
    }

    /// _
    fn cache_read(&mut self) {
        self.read_ops += 1;
    }

    /// _
    fn cache_hit(&mut self) {
        self.read_hits += 1;
    }

    /// _
    fn p_weight(&mut self, p: u64) {
        self.p_weight = p;
    }

    /// _
    fn shared_max(&mut self, i: u64) {
        self.shared_max = i;
    }

    /// _
    fn freq(&mut self, i: u64) {
        self.freq = i;
    }

    /// _
    fn recent(&mut self, i: u64) {
        self.recent = i;
    }

    /// _
    fn all_seen_keys(&mut self, i: u64) {
        self.all_seen_keys = i;
    }
}

/// A simple track of counters from the cache
#[derive(Debug, Default, Clone)]
pub struct ReadCountStat {
    /// The number of attempts to read from the cache
    pub read_ops: u64,
    /// The number of cache hits on the thread local cache
    pub local_hit: u64,
    /// The number of cache hits on the main cache
    pub main_hit: u64,
    /// The number of queued inclusions to the main cache
    pub include: u64,
    /// The number of failed inclusions to the main cache
    pub failed_include: u64,
    /// The number of inclusions to the thread local cache
    pub local_include: u64,
}

/// _
impl ARCacheReadStat for ReadCountStat {
    /// _
    fn cache_read(&mut self) {
        self.read_ops += 1;
    }

    /// _
    fn cache_local_hit(&mut self) {
        self.local_hit += 1;
    }

    /// _
    fn cache_main_hit(&mut self) {
        self.main_hit += 1;
    }

    /// _
    fn include(&mut self) {
        self.include += 1;
    }

    /// _
    fn failed_include(&mut self) {
        self.failed_include += 1;
    }

    /// _
    fn local_include(&mut self) {
        self.local_include += 1;
    }
}
