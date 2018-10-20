Concread
========

Concurrently readable datastructures for Rust.

Concurrently readable is often referred to as Copy-On-Write, Multi-Version-Concurrency-Control.

These structures allow multiple readers with transactions
to proceed while single writers can operate. A reader is guaranteed the content
will remain the same for the duration of the read, and readers do not block writers.
Writers are serialised, just like a mutex.


When do I want to use these?
----------------------------

You can use these in place of a RwLock, and will likely see improvements in
parallel throughput.

The best use is in place of mutex/rwlock, where the reader exists for a
non-trivial amount of time.

For example, if you have a RwLock where the lock is taken, data changed or read, and dropped
immediately, this probably won't help you.

However, if you have a RwLock where you hold the read lock for any amount of time,
writers will begin to stall - or inversely, the writer will cause readers to block
and wait as the writer proceeds.

In the future, a concurrent BTree and HashTree will be added, that can be used inplace
of a `RwLock<BTreeMap>` or `RwLock<HashMap>`. Stay tuned!


What is concurrently readable?
------------------------------

In a multithread application, data is commonly needed to be shared between threads.
In sharing this there are multiple policies for this - Atomics for single integer
reads, Mutexs for single thread access, RwLock for many readers or one writer,
all the way to Lock Free which allows multiple read and writes of queues.

Lock Free however has the limitation of being built on Atomics. This means it can
really only update small amounts of data at a time consistently. It also means
that you don't have transactional behaviours. While this is great for queues,
it's not so good for a tree or hashmap where you want the state to be consistent
from the state to the end of an operation.

Mutexs and RwLock on the other hand allow much more complex structures to be protected,
but they cause stalls on other threads waiting to access them. RwLock for example
can see large delays if a reader won't yield!

Concurrently readable structures sit in between these two points. They provide
multiple concurrent readers, with transactional behaviour, while allowing single
writers to proceed simultaneously.

This is achieved by having writers copy the internal data before they modify
it. This allows readers to access old data, without modification, and allows
the writer to change the data inplace before commiting. Once the new data is
stored, old readers continue to access their old data - new readers will
see the new data.

This is a space-time trade off, using more memory to achieve better parallel
behaviour.

Contributing
------------

Please open an issue, pr or contact me directly by email (see github)

