use crate::internals::bptree::cursor::{CursorRead, CursorWrite, SuperBlock};
use crate::internals::lincowcell::{LinCowCellRaw, LinCowCellCapable};

struct TestStruct {
    bptree_map_a: SuperBlock<u32, u32>,
    bptree_map_b: SuperBlock<u32, u32>,
}

struct TestStructRead {
    bptree_map_a: CursorRead<u32, u32, parking_lot::RawMutex>,
    bptree_map_b: CursorRead<u32, u32, parking_lot::RawMutex>,
}

struct TestStructWrite {
    bptree_map_a: CursorWrite<u32, u32>,
    bptree_map_b: CursorWrite<u32, u32>,
}

impl LinCowCellCapable<TestStructRead, TestStructWrite> for TestStruct {
    fn create_reader(&self) -> TestStructRead {
        // This sets up the first reader.
        TestStructRead {
            bptree_map_a: self.bptree_map_a.create_reader(),
            bptree_map_b: self.bptree_map_b.create_reader(),
        }
    }

    fn create_writer(&self) -> TestStructWrite {
        // This sets up the first writer.
        TestStructWrite {
            bptree_map_a: <SuperBlock<u32, u32> as LinCowCellCapable<
                CursorRead<u32, u32, parking_lot::RawMutex>,
                CursorWrite<u32, u32>,
            >>::create_writer(&self.bptree_map_a),
            bptree_map_b: <SuperBlock<u32, u32> as LinCowCellCapable<
                CursorRead<u32, u32, parking_lot::RawMutex>,
                CursorWrite<u32, u32>,
            >>::create_writer(&self.bptree_map_b),
        }
    }

    fn pre_commit(&mut self, new: TestStructWrite, prev: &TestStructRead) -> TestStructRead {
        let TestStructWrite {
            bptree_map_a,
            bptree_map_b,
        } = new;

        let bptree_map_a = self
            .bptree_map_a
            .pre_commit(bptree_map_a, &prev.bptree_map_a);

        let bptree_map_b = self
            .bptree_map_b
            .pre_commit(bptree_map_b, &prev.bptree_map_b);

        TestStructRead {
            bptree_map_a,
            bptree_map_b,
        }
    }
}

#[test]
fn test_lc_basic() {
    let lcc: LinCowCellRaw<TestStruct, TestStructRead, TestStructWrite, parking_lot::RawMutex> =
        LinCowCellRaw::new(TestStruct {
            bptree_map_a: unsafe { SuperBlock::new() },
            bptree_map_b: unsafe { SuperBlock::new() },
        });

    let x = lcc.write();

    x.commit();

    let _r = lcc.read();
}
