cdef void init_locals_and_stack(void* f, list nlocals, list stack, int co_nlocalsplus)
cdef object _snapshot_frame(void* frame_ptr, int is_leaf, object analyzer)