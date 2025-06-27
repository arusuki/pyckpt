import copyreg
from io import BytesIO
from queue import SimpleQueue
from types import FrameType, NoneType
from typing import IO, Any, Callable, Dict, Generator, List, Optional, Tuple, Type

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


def new_snapshot_registry() -> Registry:
    return {
        FrameType: reduce_as_none,  # TODO: support frame types
        get_generator_type(): reduce_generator,
        NullObjectType: reduce_null_object,
        SimpleQueue: reduce_simple_queue,
    }


class Pickler(dill.Pickler):
    def __init__(self, file, persistent_mapping: Dict[Type, Mapping], *args, **kwds):
        super().__init__(file, *args, **kwds)
        self._pm = persistent_mapping if persistent_mapping else {}

    def persistent_id(self, obj):
        obj_type = type(obj)
        if obj_type in self._pm:
            return self._pm[obj_type](obj)
        return None

    def persist_mapping(self):
        return self._pm


def create_pickler(
    file: IO[bytes],
    persist_mapping: Optional[Dict[Type, Mapping]] = None,
) -> Pickler:
    return Pickler(file=file, persistent_mapping=persist_mapping)


def dump(
    pickler: Pickler,
    cocoon: Any,
):
    dispatch_table = copyreg.dispatch_table.copy()
    registry = new_snapshot_registry()
    for _type, _entry in registry.items():
        dispatch_table[_type] = _entry
    pickler.dispatch_table = dispatch_table
    pickler.dump(cocoon)


class Unpickler(dill.Unpickler):
    def __init__(self, file, objects, *args, **kwds):
        super().__init__(file, *args, **kwds)
        self._objects = objects

    def persistent_load(self, pid: int):
        return self._objects[pid]


def load(file: IO[bytes], objects: Dict) -> Any:
    unpickler = Unpickler(file, objects=objects)
    return unpickler.load()


def copy(
    cocoon: Any,
    objects: Optional[Dict] = None,
    persist_mapping: Optional[Dict[Type, Mapping]] = None,
):
    buffer = BytesIO()
    pickler = create_pickler(buffer, persist_mapping)
    dump(pickler, cocoon)
    buffer.seek(0)
    objects = objects if objects else {}
    return load(buffer, objects)
