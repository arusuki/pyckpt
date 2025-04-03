from setuptools import setup
from Cython.Build import cythonize

DEBUGGABLE_OBJECT = False
if DEBUGGABLE_OBJECT:
    import os
    os.environ['CFLAGS'] = '-g -O0'


setup(
    name="pyckpt",
    version="0.0.0",
    ext_modules=cythonize(
        "pyckpt/*.pyx",
        compiler_directives={"language_level": "3"},
    ),
    packages=["pyckpt"],
    requires=["Cython"],
    python_requires=">=3.11, <3.12",
)
