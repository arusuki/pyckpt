# cython: embedsignature=True, embedsignature.format=python

from types import CodeType, FrameType, FunctionType, TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, NamedTuple
from libc.stdlib cimport malloc, free
from cpython.ref cimport PyObject, Py_INCREF
from threading import Thread
from itertools import chain
from pyckpt.interpreter.cpython cimport *
from pyckpt.interpreter.cpython cimport (
    _Py_CODEUNIT,
    _PyErr_GetHandledException,
    _PyErr_SetHandledException,
    _PyEval_EvalFrameDefault,
)
from pyckpt.util import (
    CodePosition,
    BytecodeParseError,
    dump_code_and_offset,
)

import dis
import inspect
import logging
import sys
import bytecode

logger = logging.getLogger(__name__)

class NullObjectType:

    def __reduce__(self) -> str | tuple[Any, ...]:
        raise NotImplementedError("pickle NullObjectType is not allowed")

NullObject = NullObjectType()

cdef extern from "Python.h":
    int CO_GENERATOR
    int CO_COROUTINE
    int CO_ASYNC_GENERATOR
    int WAIT_LOCK
    int PyTrace_C_CALL
    int PyTrace_CALL

    ctypedef int (*Py_tracefunc)(PyObject *, PyFrameObject *, int, PyObject *)
    cdef int PyThread_acquire_lock(PyThread_type_lock lock, int wait_flag)
    cdef int PyEval_SetProfile(Py_tracefunc func, PyObject *arg)
    cdef void PyThread_release_lock(PyThread_type_lock lock)
    cdef int _PyEval_SetProfile(PyThreadState *tstate, Py_tracefunc func, PyObject *arg)
    cdef int _PyEval_SetTrace(PyThreadState *tstate, Py_tracefunc func, PyObject *arg)
    cdef PyObject *PyCode_GetCode(PyCodeObject *co)
    cdef char *PyBytes_AS_STRING(PyObject *string)


if (3, 11) <= sys.version_info <= (3, 12):
    from pyckpt.interpreter.cpython cimport PyInterpreterFrame_311 as _PyInterpreterFrame
    from pyckpt.interpreter.cpython cimport GET_FRAME_311 as GET_FRAME
    from pyckpt.interpreter.cpython cimport PyFrame_InitializeSpecials_311 as _PyFrame_InitializeSpecials
    from pyckpt.interpreter.cpython cimport get_frame_owned_by_generator_311 as get_frame_owned_by_generator
    from pyckpt.interpreter.cpython cimport  _PyRuntimeState_311 as _PyRuntimeState
else:
    raise SystemError(f"unsupported python version: {sys.version_info}")

from pyckpt.interpreter.cpython cimport get_python_py_runtime

logger = logging.getLogger(__name__)


Analyzer = Callable[[FunctionType, int, bool], int]
ExceptionStates=Tuple[Type, Exception, TracebackType]

cdef extern from *:
    """
    #define OPCODE(x) (x >> 8)
    #define ARG(x) (x & 0xff)
    #include <stddef.h>
    #define container_of(ptr, type, member)                                        \
    ((type *)((char *)(ptr) - offsetof(type, member)))
    #define generator_of(ptr) (container_of(ptr, PyGenObject, gi_iframe))
    """

    cdef int OPCODE(int code_unit)
    cdef int ARG(int code_unit)
    cdef PyObject* generator_of(_PyInterpreterFrame* frame)

cdef PyObject* __PyErr_Occurred(PyThreadState* tstate):
    return tstate.curexc_type if tstate != NULL else NULL

def frame_specials_size():
    return (sizeof(_PyInterpreterFrame) - 1) // sizeof(PyObject*)


cdef PyThreadState* _get_thread_state_by_id(unsigned long thread_id):
    cdef PyThreadState *tstate = NULL;
    cdef _PyRuntimeState* runtime = <_PyRuntimeState*> get_python_py_runtime()
    if runtime == NULL:
        return NULL
    PyThread_acquire_lock(runtime.interpreters.mutex, WAIT_LOCK)
    tstate = PyInterpreterState_ThreadHead(PyThreadState_GET().interp)
    PyThread_release_lock(runtime.interpreters.mutex)
    while tstate != NULL:
        if tstate.thread_id != thread_id:
            tstate = PyThreadState_Next(tstate)
            continue
        return  tstate
    return tstate

cdef PyThreadState* get_thread_by_id(unsigned long thread_id):
    cdef PyThreadState* tstate = NULL
    if PyThread_get_thread_ident() == thread_id:
        return PyThreadState_GET()
    return _get_thread_state_by_id(thread_id)


cdef void init_locals_and_stack(
    void* frame_,
    list nlocals,
    list stack,
    int co_nlocalsplus,
):
    cdef _PyInterpreterFrame* frame = <_PyInterpreterFrame*> frame_
    # n local vars + 1 sentinel(NULL)
    for i in range(co_nlocalsplus + 1):
        frame.localsplus[i] = NULL
    assert len(nlocals) <= co_nlocalsplus
    for i, py_obj in enumerate(nlocals):
        if py_obj is not NullObject:
            Py_INCREF(py_obj)
            obj = <PyObject*>py_obj
            frame.localsplus[i] = obj
        else:
            frame.localsplus[i] = NULL
    cdef PyObject** stack_base = <PyObject**> &frame.localsplus[co_nlocalsplus]
    for i, py_obj in enumerate(stack):
        if py_obj is not NullObject:
            Py_INCREF(py_obj)
            obj = <PyObject*>py_obj
            stack_base[i] = obj
        else:
            stack_base[i] = NULL
    frame.stacktop = co_nlocalsplus + len(stack)


def fetch_exception() -> ExceptionStates:
    cdef PyObject* p_type  = NULL
    cdef PyObject* p_value = NULL
    cdef PyObject* p_traceback = NULL

    PyErr_Fetch(&p_type, &p_value, &p_traceback)
    return (<object> p_type, <object> p_value, <object> p_traceback)

def restore_exception(states: ExceptionStates):
    PyErr_Restore(<PyObject*> states[0], <PyObject*> states[1], <PyObject*> states[2])

class EvaluateResult(NamedTuple):
    ret: Any
    exception_states: Optional[ExceptionStates]


cdef int _check_generator(_PyInterpreterFrame* frame):
    cdef PyCodeObject* code = <PyCodeObject*> frame.f_func.func_code
    cdef int co_flags = code.co_flags
    cdef int mask = (CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR)
    cdef int flag = co_flags & mask
    if flag == 0:
        return 0
    if flag == inspect.CO_GENERATOR:
        assert frame.owner == get_frame_owned_by_generator()
        return 1
    else:
        return -1

def get_generator(frame: FrameType):
    cdef _PyInterpreterFrame* _frame = GET_FRAME(<PyFrameObject*> frame)
    if _check_generator(_frame) <= 0:
        return None
    gen = <object> generator_of(_frame)
    Py_INCREF(gen)
    return <object> gen


def save_thread_state(thread: Thread):
    cdef PyThreadState* tstate
    cdef PyObject* exc
    tstate = get_thread_by_id(<unsigned long> thread.ident)
    if tstate == NULL:
        raise RuntimeError("fail to fetch python runtime information")
    if __PyErr_Occurred(tstate) != NULL:
        raise RuntimeError("capture a thread while exception is happening")
    exc = _PyErr_GetHandledException(tstate)
    exc_obj = <object> exc if exc != NULL else None
    # FIXME: exception states should be a linked list of stacked exception_info
    return {
        "exception": exc_obj
    }

def restore_thread_state(state: Dict):
    exc = state["exception"]
    if exc is None:
        _PyErr_SetHandledException(PyThreadState_GET(), NULL)
        return
    if PyErr_Occurred() != NULL or PyErr_GetHandledException() != NULL:
        raise RuntimeError("set thread state when there is exception pending")
    _PyErr_SetHandledException(PyThreadState_GET(), <PyObject *> exc)
    state["exception"] = None


cdef int trace_trampoline(PyObject *callback, PyFrameObject *frame, int what, PyObject *arg) noexcept:
    assert frame != NULL
    (<object> callback)(
        <object> frame,
        what,
        <object> arg if arg != NULL else NullObject,
    )
    return 0

def set_profile(func: Optional[FunctionType]):
    if not func:
        PyEval_SetProfile(NULL, NULL)
        return
    cdef _PyRuntimeState* runtime = <_PyRuntimeState*> get_python_py_runtime()
    cdef PyThreadState* tstate
    if runtime == NULL:
        raise RuntimeError("filed to fetch python runtime")
    PyEval_SetProfile(trace_trampoline, <PyObject*> func)

def set_profile_all_threads(func: Optional[FunctionType]):
    if not func:
        PyEval_SetProfile(NULL, NULL)
        return
    cdef _PyRuntimeState* runtime = <_PyRuntimeState*> get_python_py_runtime()
    cdef PyThreadState* tstate
    if runtime == NULL:
        raise RuntimeError("filed to fetch python runtime")

    PyThread_acquire_lock(runtime.interpreters.mutex, WAIT_LOCK)
    tstate = PyInterpreterState_ThreadHead(PyThreadState_GET().interp)
    PyThread_release_lock(runtime.interpreters.mutex)

    while tstate != NULL:
        _PyEval_SetProfile(tstate, trace_trampoline, <PyObject*> func)
        tstate = PyThreadState_Next(tstate)

cdef object _do_snapshot_frame(void* frame_ptr, int stack_size_hint):
    cdef _PyInterpreterFrame* frame = <_PyInterpreterFrame*> frame_ptr
    cdef PyFunctionObject* func = <PyFunctionObject*>(frame.f_func)
    cdef PyCodeObject* code = <PyCodeObject*> func.func_code
    cdef int nlocalsplus = code.co_nlocalsplus
    cdef _Py_CODEUNIT* code_start = <_Py_CODEUNIT*> code.co_code_adaptive
    cdef int stack_size = stack_size_hint
    if stack_size < 0:
        stack_size = frame.stacktop
    elif frame.stacktop >= 0 and stack_size > frame.stacktop:
        raise ValueError(f"invalid stack size hint: {stack_size_hint}, actual size: {frame.stacktop}") 
    if stack_size < 0:
        raise ValueError("require a hint for stack size")

    CALL = dis.opmap["CALL"]
    instr_offset = frame.prev_instr - code_start

    captured = {
        "func": <object> func,
        "nlocals": [
            <object> frame.localsplus[i]
                if frame.localsplus[i] != NULL else NullObject
            for i in range(nlocalsplus)
        ],
        "stack": [
            <object> frame.localsplus[nlocalsplus + i]
                if frame.localsplus[nlocalsplus + i] != NULL else NullObject
            for i in range(stack_size)
        ],
        "instr_offset": instr_offset,
    }
    for obj in chain(captured["nlocals"], captured["stack"]):
        Py_INCREF(obj)
    Py_INCREF(<object> func)
    return captured

def snapshot_frame(frame: FrameType, stack_size_hint: int = -1):
    cdef _PyInterpreterFrame* _frame = GET_FRAME(<PyFrameObject*>frame)
    return _do_snapshot_frame(_frame, stack_size_hint)

def eval_frame(
    func: FunctionType,
    nlocals: List[Any],
    stack: List[Any],
    instr_offset: int, 
    exc_states: Optional[ExceptionStates] = None,
) -> EvaluateResult:
    cdef PyFunctionObject* func_obj = <PyFunctionObject*>func
    cdef PyThreadState* state = PyThreadState_GET()
    cdef PyCodeObject * code = <PyCodeObject*> func_obj.func_code;
    cdef size_t size = code.co_nlocalsplus + code.co_stacksize + frame_specials_size();
    cdef _PyInterpreterFrame *frame = <_PyInterpreterFrame*> malloc(sizeof(PyObject*) * size)

    _PyFrame_InitializeSpecials(
        frame,
        <PyFunctionObject*> func_obj,
        func_obj.func_globals,
        code.co_nlocalsplus
    )
    frame.prev_instr = &(<_Py_CODEUNIT*>(code.co_code_adaptive))[instr_offset]
    init_locals_and_stack(frame, nlocals, stack, code.co_nlocalsplus)
    do_exc = 0
    if exc_states is not None:
        if PyErr_Occurred() != NULL:
            _, exc_prev, _ = fetch_exception()
            PyErr_Clear()
            raise RuntimeError("eval a frame when exception has been raised")\
                from exc_prev
        restore_exception(exc_states)
        do_exc = 1
    cdef PyObject* result = _PyEval_EvalFrameDefault(state, frame, do_exc)
    free(frame)
    if result == NULL:
        exc_states = fetch_exception()
        return EvaluateResult(NullObject, exc_states)
    return EvaluateResult(<object> result, None)


cpdef int frame_lasti_opcode(object frame):
    cdef PyCodeObject* code = <PyCodeObject*> frame.f_code
    cdef PyObject* code_bytes = PyCode_GetCode(code)
    cdef unsigned char* codes = <unsigned char*> PyBytes_AS_STRING(code_bytes)
    cdef int last_i = frame.f_lasti
    if last_i < 0:
        return 0
    assert sizeof(_Py_CODEUNIT) == 2
    while codes[last_i] == 0:
        last_i -= sizeof(_Py_CODEUNIT)
        if last_i < 0:
            return 0
    return codes[last_i]
