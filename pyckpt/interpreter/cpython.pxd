from libc.stdint cimport uint16_t

cdef extern from "Python.h":
    ctypedef struct PyObject:
        pass

    cdef void PyObject_GC_Track(PyObject *obj)

    cdef int Py_SIZE(PyObject*)

    ctypedef struct PyCodeObject:
        int co_stacksize
        int co_nlocalsplus
        int co_flags
        char[1] co_code_adaptive

    ctypedef int _Py_CODEUNIT

    ctypedef struct PyFunctionObject:
        PyObject *func_code
        PyObject *func_globals
        PyObject *func_name
        PyObject *func_qualname

    ctypedef struct PyFrameObject:
        pass

    cdef extern PyObject* _PyEval_EvalFrameDefault(PyThreadState *tstate, void *frame, int throwflag)

    cdef PyObject* Py_None


cdef extern from *:
    """
    #include <stdbool.h>

    struct PyFunctionObject;

    struct PyInterpreterFrame_311;

    typedef struct PyInterpreterFrame_311 {
        PyFunctionObject *f_func;
        PyObject *f_globals;
        PyObject *f_builtins;
        PyObject *f_locals;
        PyCodeObject *f_code;
        PyFrameObject *frame_obj;
        struct PyInterpreterFrame_311 *previous;
        _Py_CODEUNIT *prev_instr;
        int stacktop;
        bool is_entry;
        char owner;
        PyObject *localsplus[1];
    } PyInterpreterFrame_311;

    typedef struct PyFrameObject_311 {
        PyObject_HEAD
        PyFrameObject *f_back;
        struct PyInterpreterFrame_311 *f_frame;
        PyObject *f_trace;
        int f_lineno;
        char f_trace_lines;
        char f_trace_opcodes;
        char f_fast_as_locals;
        PyObject *_f_frame_data[1];
    } PyFrameObject_311;

    #define GET_FRAME_311(f) (((PyFrameObject_311*)f)->f_frame)

    enum frame_owner_311 {
        FRAME_OWNED_BY_THREAD_311       = 0,
        FRAME_OWNED_BY_GENERATOR_311    = 1,
        FRAME_OWNED_BY_FRAME_OBJECT_311 = 2
    };

    static inline int get_frame_owned_by_generator_311() {return FRAME_OWNED_BY_GENERATOR_311; }

    static inline void
    PyFrame_InitializeSpecials_311(
        PyInterpreterFrame_311 *frame, PyFunctionObject *func,
        PyObject *locals, int nlocalsplus)
    {
        frame->f_func = func;
        frame->f_code = (PyCodeObject *)Py_NewRef(func->func_code);
        frame->f_builtins = func->func_builtins;
        frame->f_globals = func->func_globals;
        frame->f_locals = Py_XNewRef(locals);
        frame->stacktop = nlocalsplus;
        frame->frame_obj = NULL;
        frame->prev_instr = _PyCode_CODE(frame->f_code) - 1;
        frame->is_entry = false;
        frame->owner = FRAME_OWNED_BY_THREAD_311;
    }

    typedef enum {
        FRAME_CREATED_311   = -2,
        FRAME_SUSPENDED_311 = -1,
        FRAME_EXECUTING_311 =  0,
        FRAME_COMPLETED_311 =  1,
        FRAME_CLEARED_311   =  4
    } PyFrameState_311;

    static inline int get_frame_cleared_311() { return FRAME_CLEARED_311; }
    static inline int get_frame_created_311() { return FRAME_CREATED_311; }
    static inline int get_frame_suspended_311() { return FRAME_SUSPENDED_311; }
    static inline int get_frame_executing_311() { return FRAME_EXECUTING_311; }

    #include <stdatomic.h>

    typedef struct {
        PyThread_type_lock mutex;
        PyInterpreterState *head;
        PyInterpreterState *main;
        int64_t next_id;
    } pyinterpreters_311;

    typedef struct {
        int _initialized;
        int preinitializing;
        int preinitialized;
        int core_initialized;
        int initialized;
        struct {atomic_uintptr_t ptr;} _finalizing;
        pyinterpreters_311 interpreters;
    } _PyRuntimeState_311;

    extern __attribute__((weak)) char _PyRuntime;

    static inline void* get_python_py_runtime() {
        return &_PyRuntime;
    }

    """

    # https://github.com/python/cpython/blob/3.11/Include/internal/pycore_frame.h
    ctypedef struct PyInterpreterFrame_311:
        PyFunctionObject *f_func
        _Py_CODEUNIT* prev_instr
        int stacktop
        int owner
        PyObject* localsplus[1]
        PyInterpreterFrame_311* previous
        PyFrameObject* frame_obj

    cdef PyInterpreterFrame_311* GET_FRAME_311(PyFrameObject* obj)
    cdef void PyFrame_InitializeSpecials_311(PyInterpreterFrame_311 *frame, PyFunctionObject *func, PyObject *locals, int nlocalsplus)

    int FRAME_CREATED
    int FRAME_SUSPENDED
    int FRAME_EXECUTING
    int FRAME_COMPLETED
    int FRAME_CLEARED

    cdef inline int get_frame_owned_by_generator_311()
    cdef inline int get_frame_cleared_311()
    cdef inline int get_frame_created_311()
    cdef inline int get_frame_suspended_311()
    cdef inline int get_frame_executing_311()

    ctypedef struct PyThread_type_lock:
        pass
    ctypedef struct pyinterpreters_311:
        PyThread_type_lock mutex
    ctypedef struct _PyRuntimeState_311:
        pyinterpreters_311 interpreters

    cdef void *get_python_py_runtime()


cdef extern from "pyerrors.h":
    cdef PyObject* _PyErr_GetHandledException(PyThreadState* tstate)
    cdef PyObject* _PyErr_SetHandledException(PyThreadState* tstate, PyObject *exc)
    cdef PyObject* PyErr_Occurred()
    cdef PyObject* PyErr_GetHandledException()

    cdef void PyErr_Clear()
    cdef PyObject* PyErr_SetObject(PyObject* type, PyObject* value)
    cdef void PyErr_Fetch(PyObject** p_type, PyObject** p_value, PyObject** p_traceback)
    cdef void PyErr_Restore(PyObject *type, PyObject *value, PyObject *traceback)

cdef extern from "pystate.h":
    ctypedef struct PyInterpreterState:
        pass

    ctypedef struct _PyCFrame:
        void* current_frame
        _PyCFrame *previous

    ctypedef struct PyThreadState:
        _PyCFrame* cframe
        PyObject *curexc_type
        PyInterpreterState* interp
        unsigned long thread_id
        void *exc_info

    ctypedef struct PyGILState_STATE:
        pass

    cdef PyGILState_STATE PyGILState_Ensure()

    cdef void PyGILState_Release(PyGILState_STATE)

    cdef PyThreadState * PyThreadState_GET()

    cdef PyThreadState *PyInterpreterState_ThreadHead(PyInterpreterState* interp)

    cdef PyThreadState *PyThreadState_Next(PyThreadState *tstate)

cdef extern from "pythread.h":
    cdef unsigned long PyThread_get_thread_ident()
