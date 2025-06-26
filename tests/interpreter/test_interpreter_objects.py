from queue import SimpleQueue
import pyckpt.interpreter.objects as objs

def test_io_simple_queue():
    sq = SimpleQueue()
    sq.put(1)
    sq.put(2)
    sq.put(3)

    lst = objs.snapshot_simple_queue(sq)
    assert isinstance(lst, list)
    assert len(lst) == 3

    assert sq.get() == 1
    lst = objs.snapshot_simple_queue(sq)
    assert isinstance(lst, list)
    assert len(lst) == 2
