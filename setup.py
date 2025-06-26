import sys
from typing import List, Union

from Cython.Build import cythonize
from setuptools import Extension, setup

DEBUGGABLE_OBJECT = False
if DEBUGGABLE_OBJECT:
    import os

    os.environ["CFLAGS"] = "-g -O0"


def interpreter_module() -> List[Union[Extension, str]]:
    version = f"{sys.version_info.major}_{sys.version_info.minor}"
    supported_versions = ("3_11",)
    if version not in supported_versions:
        raise RuntimeError(
            f"unsupported CPython version:\
                {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

    return [
        "pyckpt/interpreter/frame.pyx",
        "pyckpt/interpreter/generator.pyx",
        "pyckpt/interpreter/objects.pyx",
    ]


cython_modules = []
cython_modules += interpreter_module()


setup(
    name="pyckpt",
    version="0.0.0",
    ext_modules=cythonize(
        cython_modules,
        compiler_directives={
            "language_level": "3",
        },
    ),
    packages=["pyckpt"],
    requires=["Cython"],
    python_requires=">=3.11, <3.12",
)
