use super::node::Node;

struct BptreeMap<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    root: Node<K, V>,
}

impl<K: Clone + Ord, V: Clone> BptreeMap<K, V> {
    // new

    // clear

    // get

    // contains_key

    // get_mut

    // insert (or update)

    // remove

    // split_off

    // ADVANCED
    // append (join two sets)

    // range/range_mut

    // entry

    // iter

    // iter_mut

    // keys

    // values

    // values_mut

    // len

    // is_empty
}
