from cpython.ref cimport PyObject

cdef extern from "Python.h":
    ctypedef Py_ssize_t int

cdef extern from *:
    """
    typedef struct {
        PyObject_HEAD
        PyThread_type_lock lock;
        int locked;
        PyObject *lst;
        Py_ssize_t lst_pos;
        PyObject *weakreflist;
    } simplequeueobject_311;
    """
    
    ctypedef struct simplequeueobject_311:
        PyObject *lst;
        Py_ssize_t lst_pos;

