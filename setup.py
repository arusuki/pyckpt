import sys
from typing import List

from Cython.Build import cythonize
from setuptools import Extension, setup

DEBUGGABLE_OBJECT = False
if DEBUGGABLE_OBJECT:
    import os

    os.environ["CFLAGS"] = "-g -O0"


def platform_module() -> List[str]:
    if sys.platform == "darwin":
        return [
            Extension("pyckpt.platform", ["pyckpt/platform/darwin.pyx"]),
        ]
    return []


def interpreter_module() -> List[Extension]:
    version = f"{sys.version_info.major}_{sys.version_info.minor}"
    if version == "3_11":
        return ["pyckpt/interpreter/frame.pyx"]
    else:
        raise RuntimeError(
            f"unsupported CPython version:\
                {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )


cython_modules = []
cython_modules += platform_module()
cython_modules += interpreter_module()


setup(
    name="pyckpt",
    version="0.0.0",
    ext_modules=cythonize(
        cython_modules,
        compiler_directives={"language_level": "3"},
    ),
    packages=["pyckpt"],
    requires=["Cython"],
    python_requires=">=3.11, <3.12",
)
