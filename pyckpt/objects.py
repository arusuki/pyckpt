import copyreg
import multiprocessing
from io import BytesIO
from multiprocessing import Process
from queue import SimpleQueue
from threading import Thread
from types import FrameType, NoneType
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import dill
import numpy
import torch
from torch.serialization import default_restore_location as restore_location
from torch.serialization import location_tag

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

def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
    if isinstance(bytes_str, bytes):
        return bytes_str.decode("ascii")
    return bytes_str

def normalize_storage_type(storage_type):
    return getattr(torch, storage_type.__name__)


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

def save_untyped_storages(file: IO[bytes], storages: dict[str, torch.storage.UntypedStorage]):
    dill.Pickler(file).dump(
        {
            key: torch.TypedStorage(
                wrap_storage=storage, dtype=torch.uint8, _internal=True
            )
            for key, storage in storages.items() 
        }
    )

def load_untyped_storages(file: IO[bytes]):
    storages: dict[str, torch.storage.TypedStorage] = dill.Unpickler(file).load()
    for key, store in storages.items():
        storages[key] = store._untyped_storage
    return storages


def get_leaf_base(array: numpy.ndarray) -> numpy.ndarray | torch.Tensor:
    while isinstance(array, numpy.ndarray) \
        and array.base is not None:
        array = array.base
    return array

def _create_array(t: torch.Tensor):
    return t.numpy()

def save_numpy_with_tensor_base(pickler: "Pickler", np_array: numpy.ndarray, base: torch.Tensor):
    if np_array.__array_interface__['data'][0] != \
      base.untyped_storage().data_ptr():
        raise NotImplementedError("numpy&tensor shared storage with offset")
    pickler.save_reduce(_create_array, (base,), obj=np_array)

class StorageID(Tuple): ...

class Pickler(dill.Pickler):
    def __init__(self, file, *args, **kwds):
        super().__init__(file, *args, **kwds)
        self._persisted: PersistedObjects = {}
        for tp in PersistType:
            self._persisted[tp] = set()
        self.dispatch_table = copyreg.dispatch_table.copy()
        self.dispatch_table.update(dispatch_table())
        self.serialized_storages: dict[str, torch.storage.UntypedStorage] = {}
        self.storage_dtypes: dict[int, torch.dtype] = {}
        self.id_map: dict[int, str] = {}


    def persistent_id(self, obj):
        if isinstance(obj, torch.storage.TypedStorage) \
          or torch.is_storage(obj):
            return self._persistent_id_storage(obj)
        obj_type = type(obj)
        if obj_type in PersistType:
            obj_id = id(obj)
            self._persisted[obj_type].add(obj_id)
            return obj_id
        return None
    
    def consume_persisted(self) -> PersistedObjects:
        self._persisted, persisted = {}, self._persisted
        self.serialized_storages, storages = {}, self.serialized_storages
        self.storage_dtypes.clear()
        self.id_map.clear()
        persisted["storage"] = storages
        return persisted

    def save(self, obj, save_persistent_id=True):
        if not isinstance(obj, numpy.ndarray):
            return super().save(obj, save_persistent_id)
        base = get_leaf_base(obj)
        if not torch.is_tensor(base):
            return super().save(obj, save_persistent_id)
        assert str(base.device) == "cpu"
        return save_numpy_with_tensor_base(self, obj, base)


    def _persistent_id_storage(self, obj: torch.storage.TypedStorage):
        """
        Persist tensor storage, adopted from\n
        https://github.com/pytorch/pytorch/torch/serialization.py
        """
        if isinstance(obj, torch.storage.TypedStorage):
            storage = obj._untyped_storage
            storage_dtype = obj.dtype
            storage_type_str = obj._pickle_storage_type()
            storage_type = getattr(torch, storage_type_str)
            storage_numel = obj._size()

        else:
            storage = obj
            storage_dtype = torch.uint8
            storage_type = normalize_storage_type(type(obj))
            storage_numel = storage.nbytes()

        if str(storage.device) != "meta" and storage.data_ptr() != 0:
            if storage.data_ptr() in self.storage_dtypes:
                if storage_dtype != self.storage_dtypes[storage.data_ptr()]:
                    raise RuntimeError(
                        "Cannot save multiple tensors or storages that "
                        "view the same data as different types"
                    )
            else:
                self.storage_dtypes[storage.data_ptr()] = storage_dtype

        storage_key = self.id_map.setdefault(storage._cdata, str(len(self.id_map)))
        if hasattr(obj, "_fake_device") and obj._fake_device is not None:
            location = str(obj._fake_device)
        else:
            location = location_tag(storage)
        self.serialized_storages[storage_key] = storage
        return StorageID(("storage", storage_type, storage_key, location, storage_numel))

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
        assert "storage" in objects
        self.loaded_storages: dict[str, torch.storage.UntypedStorage] = objects.pop("storage")
        self._typed_storages: dict[str, torch.storage.TypedStorage] = {}

        for tp, obj_ids in objects.items():
            for obj_id in obj_ids:
                pool[obj_id] = object.__new__(tp)
        self._pool = pool

    def persistent_load(self, pid: int | StorageID):
        if isinstance(pid, StorageID):
            return self._persistent_load_storage(pid)
        return self._pool[pid]

    def object_pool(self) -> Dict[int, Any]:
        return self._pool

    def _persistent_load_storage(self, pid: StorageID):
        typename = _maybe_decode_ascii(pid[0])
        assert typename == "storage", (
            f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        )
        storage_type, storage_key, location= pid[1], pid[2], pid[3]
        if storage_key in self._typed_storages:
            return self._typed_storages[storage_key]

        assert storage_key in self.loaded_storages
        storage = self.loaded_storages[storage_key]

        if storage_type is torch.UntypedStorage:
            dtype = torch.uint8
        else:
            dtype = storage_type._dtype

        if torch._guards.detect_fake_mode(None) is None:
            wrap_storage = restore_location(storage, location)
        else:
            storage._fake_device = location
            wrap_storage = storage

        typed_storage = \
            torch.TypedStorage(
                wrap_storage=wrap_storage, dtype=dtype, _internal=True
            )
        self._typed_storages[storage_key] = typed_storage
        return typed_storage


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

