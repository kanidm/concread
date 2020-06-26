#!/bin/sh

RUSTC_BOOTSTRAP=1  RUSTFLAGS="-Z sanitizer=address" cargo test $@

