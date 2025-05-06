# cython: embedsignature=True, embedsignature.format=python
import sys

from cpython.ref cimport Py_INCREF, PyTypeObject
from typing import Dict, Generator, Type
from pyckpt.interpreter.cpython cimport *
from pyckpt.interpreter.cpython cimport _Py_CODEUNIT
from pyckpt.interpreter.frame cimport init_locals_and_stack, _snapshot_frame


if (3, 11) <= sys.version_info <= (3, 12):
    from pyckpt.interpreter.cpython cimport PyInterpreterFrame_311 as _PyInterpreterFrame
    from pyckpt.interpreter.cpython cimport PyFrame_InitializeSpecials_311 as _PyFrame_InitializeSpecials
    from pyckpt.interpreter.cpython cimport get_frame_cleared_311 as get_frame_cleared
    from pyckpt.interpreter.cpython cimport get_frame_created_311 as get_frame_created
    from pyckpt.interpreter.cpython cimport get_frame_suspended_311 as get_frame_suspended
    from pyckpt.interpreter.cpython cimport get_frame_owned_by_generator_311 as get_frame_owned_by_generator
else:
    raise SystemError(f"unsupported python version: {sys.version_info}")

cdef extern from "Python.h":
    """
    # define New_Gen(slots) PyObject_GC_NewVar(PyGenObject, &PyGen_Type, slots)
    """

    ctypedef struct _PyErr_StackItem:
        PyObject* exc_value
        _PyErr_StackItem *previous_item

    ctypedef struct PyGenObject:
        PyCodeObject *gi_code;
        PyObject *gi_weakreflist;
        PyObject *gi_name;
        PyObject *gi_qualname;
        PyObject *gi_origin_or_finalizer;
        char gi_hooks_inited;
        char gi_closed;
        char gi_running_async;
        _PyErr_StackItem gi_exc_state;
        int gi_frame_state;
        _PyInterpreterFrame *gi_iframe;

    cdef PyGenObject* New_Gen(int slots)

    cdef PyTypeObject PyGen_Type

cdef PyGenObject* new_generator(PyFunctionObject *func):
    cdef PyCodeObject* code = <PyCodeObject*> func.func_code
    cdef int slots = code.co_nlocalsplus + code.co_stacksize;
    cdef PyGenObject* gen = New_Gen(slots)
    if gen == NULL:
        return NULL
    if func.func_name == NULL or func.func_qualname == NULL:
        return NULL
    gen.gi_frame_state = get_frame_cleared()
    gen.gi_code = code
    gen.gi_exc_state.previous_item = NULL
    gen.gi_exc_state.exc_value = NULL
    gen.gi_weakreflist = NULL
    gen.gi_name = func.func_name
    gen.gi_qualname = func.func_qualname
    Py_INCREF(<object> gen.gi_code)
    Py_INCREF(<object> gen.gi_name)
    Py_INCREF(<object> gen.gi_qualname)
    PyObject_GC_Track(<PyObject*> gen)
    return gen


cdef inline int no_exception(PyObject* exc_value):
    return exc_value == NULL or exc_value == Py_None

def snapshot_generator(generator: Generator):
    cdef PyGenObject* gen = <PyGenObject*> generator
    # if gen.gi_exc_state.exc_value != NULL:
    if not no_exception(gen.gi_exc_state.exc_value):
        raise NotImplementedError("exception stack for generators")

    Py_INCREF(<object> gen.gi_code)
    Py_INCREF(<object> gen.gi_name)
    Py_INCREF(<object> gen.gi_qualname)
    Py_INCREF(<object> gen.gi_frame_state)

    return {
        "gi_code"        : <object> gen.gi_code,
        "gi_name"        : <object> gen.gi_name,
        "gi_qualname"    : <object> gen.gi_qualname,
        "gi_frame_state" : <int>    gen.gi_frame_state,
        "suspended": gen.gi_frame_state == get_frame_suspended(),
    }

cpdef snapshot_generator_frame(object generator, object analyzer):
    cdef PyGenObject* gen = <PyGenObject*> generator
    if gen.gi_frame_state != get_frame_suspended():
        raise ValueError("snapshot non-suspended generator is not supported")
    return _snapshot_frame(gen.gi_iframe, False, analyzer)

def make_generator(gen_states: Dict, frame_states: Dict):
    cdef PyFunctionObject* func = <PyFunctionObject*> frame_states["func"]
    # N.B. The generator somehow should own a strong reference to the funcobject.
    # This is different from frame evaluation apis where the caller owns the reference.
    # Logically, the generator should own everything it could `touch`,and it can touch
    # the funcobject through the _PyInterpreterFrame C struct (which is not a PyObject!)
    # So just make the GC happy :)
    Py_INCREF(<object> func)
    cdef PyGenObject* gen = new_generator(func)
    cdef _PyInterpreterFrame* frame = gen.gi_iframe
    cdef PyCodeObject* code = <PyCodeObject*> func.func_code

    nlocals = frame_states["nlocals"]
    prev_instr_offset = frame_states["prev_instr_offset"]
    _PyFrame_InitializeSpecials(
        frame, func, <PyObject*> nlocals, code.co_nlocalsplus
    )
    frame.prev_instr = &(<_Py_CODEUNIT*>(code.co_code_adaptive))[prev_instr_offset]
    frame.owner = get_frame_owned_by_generator()
    gen.gi_frame_state = gen_states["gi_frame_state"]
    stack = frame_states["stack"]
    init_locals_and_stack(frame, nlocals, stack, code.co_nlocalsplus)
    if gen == NULL:
        raise RuntimeError("failed to make new generator")
    return <object> gen

def is_suspended(generator: Generator):
    return (<PyGenObject*> generator).gi_frame_state == get_frame_suspended()

def get_generator_type() -> Type:
    # `type` objects are not dynamically allocated
    # hence no incref here
    return <object> &PyGen_Type
