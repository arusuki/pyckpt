#define Py_BUILD_CORE

#define OPCODE(x) (x >> 8)
#define ARG(x) (x & 0xff)