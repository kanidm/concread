
Maple Tree (v2)
===============

The Maple Tree, is a modified B+Tree variant, designed by Matthew Wilcox and Liam R. Howlett. The
major modifications are removal of leaf-links, implied min-max of a branch node, and leaf node
variants. Concurrency of the tree is "built in", no external locking required. The tree is designed
to be reactive and adaptive to data inputs, self-optimising itself to gain "better" performance
compared to more generic structures.

This document is a description of the tree as understood by myself (William Brown), and may not
accurately reflect the design of the original authors, nor their terminology. Consult their work
for the authorative source. Some changes are made and I will attempt to highlight these where they
are present.

Why do I want this? (Quick Version)
-----------------------------------

If you use Mutex<HashMap/BTreeMap> or RwLock<HashMap/BTreeMap>, you will likely have a faster
and better experience with MapleTree (no mutex/rwlock required).

Behaviours
----------

The tree always has a fixed height from all leaves to the root (as per B+Tree).

Each node may be unbalanced with regard to key numbers compared to neighbours, and there is no
requirement that neighbour leaves are even of the same type. Each node implements it's own data
storage and retrieval mechanism, as well as sorting. They only assert from a branch that some data
range may exist within the node present.

The original author's tree only supports 32bit integer keys. This version supports any key which
supports PartialOrd and PartialEq, with values and keys implementing Clone. Each node will list
it's requirements.

Concurrency
-----------

Maple Trees are designed to be concurrent. Differing from the original authors work, this implementation
will aim for "concurrent readability" (rather than lock free-ish). This is because the Linux Kernel
often only requires operational consistency, where databases require transactional consistency.

An example of this to explain is:

::

    T0          T1
    read(x) -> 0
                write(x, v)
    read(x) -> v
    read(x) -> v

Note that the second read percieves the change of the write, but the read and write themselves
don't imply locking or ordering. Only that the read/write are guaranteed consistent structures.

Contrast to concurrent readability, which provides:

::

    T0          T1
    start_read()
                start_write()
    read(x) -> 0
                write(x, v)
    read(x) -> 0
    end_read()
                commit_write()
    start_read()
    read(x) -> v
    end_read()

Notice that the read will read x to 0 until the end of the read, despite concurrent writes. This
gives database style transactions and operation batching semantics which is quite important in a
database project (which I tend to work on).

Node Types
----------

The Maple Tree consists of 4 node variants (original authors describe 3). These are:

* RangeBranch
* SparseLeaf
* RangeLeaf
* DenseLeaf

Details:

* RangeBranch

The RangeBranch node is a subtle variation of the class B+Tree branch node. It contains N pointers
and N-1 pivots or keys. An important element of the RangeBranch type is that there is an implied
pivot/key at min and max. For example, if you have a node laid out as follows:

::

    ------------------------------------------------------
    |  |  4  |  8  |  12  |  16  |  20  |  24  |  28  |  |
    |  X  |  X  |  X  |  X   |   X  |   X  |   X  |   X  |
    ------------------------------------------------------

It is implied from the value at index 0 and index N-1 that min and max values must exist either
side.

::

        ------------------------------------------------------
     | (0) |  4  |  8  |  12  |  16  |  20  |  24  |  28  | (TYPE_MAX) |
        |  X  |  X  |  X  |  X   |   X  |   X  |   X  |   X  |
        ------------------------------------------------------

This allows expression of ranges: That the values between 28 -> MAX, must exist in the far right
pointer at value[N].

In the root node, this seems reasonable, as does in the branch of a tree that values between the two
pivots/keys must be bounded by them. (NOTE: I believe the authors design allows for a value to exist
in a leaf that is larger than the right parent pivot).

However, the importance of this range behaviour is shown with a subsequent RangeBranch or RangeLeaf.
This is that MIN/MAX are bounded by the pivots relative to the pointer. In an example such as:

::

        ------------------------------------------------------
     | (0) |  4  |  8  |  12  |  16  |  20  |  24  |  28  | (TYPE_MAX) |
        |  X  |  X  |  X  |  X   |   X  |   X  |   X  |   X  |
        ------------------------------------------------------
                                     \
                                      V
                ------------------------------------------------------
             | (16) |  17  |  19  |  END  |   -  |   -  |   -  |   -  | (20) |
                |  X  |  ZZ   |  X    |  X   |   X  |   X  |   X  |   X  |
                ------------------------------------------------------

It can be derived from this, that the values 17, 18, 19 must have the value ZZ. This will be covered
further in RangeLeaf as this is where the MIN/MAX behaviour has further usage.

* SparseLeaf

SparseLeaves are the "classic" B+Tree style leaf, containing a set of diverse key/value pairs. Some
minor changes are made from the classic implementation.

First, content of the SparseLeaf is un-sorted. This is because in a 128bit structure, this fills
two cachelines and it is faster to search the entire content, than the overhead of maintaining
the sorted operation of the leaf.

Second, the content of a SparseLeaf may be highly-unbalanced compared to neighbour leaves. Traditional
B+Tree construction is a "build up" process (which seems counter intuitive given a tree appears to
grow down). This process involves split and rebalance. For example, a B+tree with key storage
of three keys would grow like the following.

::

    [ 1, 2, 3 ]

    inesrt 4

    [ 1, 2, NULL ] -> [ 3, 4, NULL ]

            [ 3, NULL, NULL ]
              /         \
    [ 1, 2, NULL ] -> [ 3, 4, NULL ]

Important is to now note the presence of 4 NULL slots in the structure. This means we have 3 nodes
allocated, where 4/9 (almost 50%) of the space is now empty.

The Maple Tree sparse nodes attempt *not* to rebalance to prevent this. Given that values either are
highly randomised, or they are sequential, this turns out pretty well and keys nodes well used. An
example insert (note, I'm using a 64byte node, not 128 for brevity) would be:

::

    [ 1, 2, 3 ]

    inesrt 4

    [ 1, 2, 3 ] -> [ 4, NULL, NULL ]

            [ 4, NULL, NULL ]
              /         \
    [ 1, 2, 3 ] -> [ 4, NULL, NULL ]

Note the NULL's are now weighted to the right of the tree? This has an impact on the following operations
of insert, 5 and 6, where the tree now does *not* require an additional split, but the B+Tree would grow
to another level of height, and contain 3 leaf nodes.

By remaining dense and well packed, this improves tree compression, and compression then allows faster
search behaviours.

* DenseLeaf

A DenseLeaf is similar to a SparseLeaf in being a 128byte structure, however it omits keys. In a
sparseleaf, 64bytes of the capacity is consumed by the storage of these keys. A DenseLeaf omits keys
relying on implied keying of the values. In Rust, this would require the struct to be of the form:

::

    let tree: MapleTree<usize, _> = MapleTree::new();

In C, this is more loose, and could validly be any int type (C basically allows any int to index an array
even though the spec says it should use size_t, which is platform width unsigned).

An assumption is that if a DenseLeaf is the root node, we essentially have an array where we assume
starts at 0. Fast quick arrays anyone?

An example of this structure in a different scenario is assuming we have a RangeBranch as a parent
with the left and right pivots as 32 and 48.

::

    [ ...., 32, 48, .... ]
            |
            v
    [a, b, c, d, ... ]

This would mean that the value of a, residing at index 0, is (32, a). Similar, c must then be pivot + index
yielding (34, c).

DenseLeaves are essentially a trade that if your set is highly sequential and dense, you save keyspace
in exchange for a few extra CPU cycles.

* RangeLeaf

Finally, the we have the RangeLeaf. This is similar to the RangeBranch, storing keys and values with
implied MIN/MAX values from  the parent pivots. However, an important change is that ranges are implied
between the pivots of the RangeLeaf, allowing a single value to be stored for a dense set of keys.

In Rust this will likely only be possible for values where a defined MAX is known, such as integer
types. There is no "max" string, so ranges are not possible by definition.

As an example:

::

        ------------------------------------------------------
     | (16) |  17  |  19  |  END  |   -  |   -  |   -  |   -  | (20) |
        |  X  |  ZZ   |  X  |  X   |   X  |   X  |   X  |   X  |
        ------------------------------------------------------

In this case (as above), the value of 16 -> 17 (aka, 16) is X, where 17 -> 19 is ZZ. This works
effectively as a generator, where iter() would count and yield ZZ repeatedly until we step past the end of 19.

This also allows effective representation of huge sets where the values are sparse or not present.
For example:

        ------------------------------------------------------
     | (0) |  17  |  18  |   -   |   -  |   -  |   -  |   -  | (20) |
        |  X  |  ZZ   |  AA   |  X   |   X  |   X  |   X  |   X  |
        ------------------------------------------------------

Note the lack of the "END" terminator. This implies that 18 -> MAX, is the value AA (it's probably a
bad idea to iter() on this though ..., but at least it's compressed in memory).

Considerations
--------------

To remain fast, and behave correctly, the correct type of node should be allocated during the operation
of the tree. To determine the correct node is an exercise of great fun. Unlike C, due to the Rust
type system, we can immediately make a number of assupmtions about the correct node to use based on
types. I will reference this in the next two suggestions.


For all MapleTree<int , _> types
we would consider dense, range OR sparse nodes.

Most types implementing PartialOrd/PartialEq
would use sparse nodes.

We can also provide a wrapper to a HashMapleTree<K: Hash, V>, which would then internally use
MapleTree<usize, _> as sparse nodes.


A self optimising behaviour with no hint mechanism could be as such: Given an inserted key-value,
if the type has a MIN-MAX bound possible (int types), if that value is outside of +-16 of the min/max
we allocate sparse. If the value is min *or* max, or +-1 of neighbour value, we use dense. If the
insert is a range, we allocate the range (and subseqent values are put into the range node and on
split we then start to go back to dense/sparse types).

This system would work because insert patterns tend to 0 onwards (IE array style usage), sparse
usage (databases indexes, hash maps), and ranges are a requested insert with values either side.



While the type system gives us a few hints (and eleminates some possibilities), we can then use
hinting mechanisms such as insert_rang, or insert_hint(k, v, flag), where flag is a hint to the
future use of the structure. The human may know better than us in some cases!

This is important because with a usize type, it's valid to want to use range, dense or sparse, but
until a pattern is established, we don't know what to use. As a result, we could implement node
inplace conversion, but it would be better to know that we have highly sequential, ranged, or randomised
data to inform our choice, rather than rely on heuristics.

I would propose an insert interface such as:

::

    pub enum MapleUsage {
        Linear,
        Ranged,
        Randomised,
    }

    let t: MapleTree<K, V> = MapleTree::new();
    t.insert_hint(k, v, MapleUsage::...)

This way we can hint to the allocator what is the correct type to use per insert. IE consider
insert of k,v of 1 through 32 with unique values, a range insert of 32 to 64 of null, and then 64 to
80 of unique values. IN this case, we want to allocate 2 dense nodes, 1 range, then 1 dense. Now
imagine if we inserted 1 through 32 out of order? Would sparse be right, when we really want dense?

Suggestions are welcome to refine these ideas!
