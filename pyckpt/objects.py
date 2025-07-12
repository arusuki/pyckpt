import copyreg
import multiprocessing
from io import BytesIO
from multiprocessing import Process
from queue import SimpleQueue
from threading import Thread
from types import FrameType, NoneType
from typing import Any, Callable, Dict, Generator, List, Set, Tuple, Type, TypeVar

import dill

from pyckpt.analyzer import analyze_stack_top
from pyckpt.interpreter.frame import NullObject, NullObjectType
from pyckpt.interpreter.generator import (
    get_generator_type,
    make_generator,
    snapshot_generator,
    snapshot_generator_frame,
)
from pyckpt.interpreter.objects import snapshot_simple_queue

None_PID = 0

Mapping = Callable[[Any], int]
Registry = Dict[Type, Callable[[Any], Tuple[Callable[[Tuple], Any], Tuple]]]


def reduce_as_none(_):
    return (NoneType, ())


def reduce_generator(gen: Generator):
    return (
        make_generator,
        (
            snapshot_generator(gen),
            snapshot_generator_frame(gen, analyze_stack_top),
        ),
    )


def _ret_null_object():
    return NullObject


def reduce_null_object(_obj: NullObjectType):
    return _ret_null_object, ()



def reduce_simple_queue(sq: SimpleQueue):
    def make_simple_queue(lst: List):
        q = SimpleQueue()
        for elem in lst:
            q.put(elem)
        return q
    return make_simple_queue, (snapshot_simple_queue(sq),)

class FakeQueue():
    def __init__(self):
        pass

    def __getattribute__(self, _name):
        raise NotImplementedError(
            "accessing multiprocessing.Queue after respawn"
        )



def reduce_multiprocessing_queue(q: multiprocessing.Queue):
    return FakeQueue, ()


def dispatch_table() -> Registry:
    return {
        FrameType: reduce_as_none,  # TODO: support frame types
        get_generator_type(): reduce_generator,
        NullObjectType: reduce_null_object,
        SimpleQueue: reduce_simple_queue,
        type(multiprocessing.Queue()): reduce_multiprocessing_queue,
    }

PersistType = (Thread, Process)
PersistedObjects = Dict[Type, Set[int]]

class Pickler(dill.Pickler):
    def __init__(self, file, *args, **kwds):
        super().__init__(file, *args, **kwds)
        self._persisted: PersistedObjects = {}
        self.dispatch_table = copyreg.dispatch_table.copy()
        self.dispatch_table.update(dispatch_table())
        for tp in PersistType:
            self._persisted[tp] = set()

    def persistent_id(self, obj):
        obj_type = type(obj)
        if obj_type in PersistType:
            obj_id = id(obj)
            self._persisted[obj_type].add(obj_id)
            return obj_id
        return None
    
    def consume_persisted(self) -> PersistedObjects:
        self._persisted, persisted = None, self._persisted
        return persisted

class Unpickler(dill.Unpickler):
    def __init__(
        self, 
        file,
        objects: PersistedObjects,
        *args,
        **kwds,
    ):
        super().__init__(file, *args, **kwds)
        pool: Dict[int, Any] = {}
        for tp, obj_ids in objects.items():
            for obj_id in obj_ids:
                pool[obj_id] = object.__new__(tp)
        self._pool = pool

    def persistent_load(self, pid: int):
        return self._pool[pid]

    def object_pool(self) -> Dict[int, Any]:
        return self._pool

def dump(file, cocoon) -> PersistedObjects:
    pickler = Pickler(file)
    pickler.dump(cocoon)
    return pickler.consume_persisted()

def load(file, objects: PersistedObjects):
    unpickler = Unpickler(file, objects)
    return unpickler.load(), unpickler.object_pool()

T = TypeVar("T")

def copy(cocoon: T) -> Tuple[T, Dict[int, Any]]:
    buffer = BytesIO()
    persist = dump(buffer, cocoon)
    buffer.seek(0)
    return load(buffer, persist)
