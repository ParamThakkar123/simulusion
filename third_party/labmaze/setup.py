# Copyright 2019 DeepMind Technologies Limited.
# Modifications copyright 2026 the Simulusion authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build labmaze with its native extensions using setuptools and pybind11.

Upstream builds the two pybind11 extensions with Bazel, and ``labmaze-new``
makes them opt-in, so on Python 3.13 neither gives you ``labmaze.RandomMaze``.
This build compiles the same C++ sources with the platform compiler (MSVC on
Windows, gcc/clang elsewhere), and falls back to pure Python when no compiler
is available. See README.md for the source changes.
"""

import fnmatch
import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
import setuptools

CC_DIR = os.path.join("labmaze", "cc")
CORE_SOURCES = [
    os.path.join(CC_DIR, name)
    for name in (
        "algorithm.cc",
        "char_grid.cc",
        "flood_fill.cc",
        "random_maze.cc",
        "text_maze.cc",
    )
]
# Sources include "labmaze/cc/x.h" from the repository root and "logging.h"
# relative to labmaze/cc.
INCLUDE_DIRS = [".", CC_DIR]


def _extension(name, sources):
    return Pybind11Extension(
        f"labmaze.cc.python.{name}",
        sources=sources,
        include_dirs=INCLUDE_DIRS,
        cxx_std=17,
    )


class OptionalBuildExt(build_ext):
    """Build the extensions, but fall back to pure Python if compiling fails.

    labmaze-new 0.0.2 installed without a compiler, as a pure-Python package
    without ``RandomMaze``. Keep that working on machines with no C++
    toolchain. Set ``LABMAZE_REQUIRE_EXTENSIONS=1`` to make a failed build an
    error instead.
    """

    def run(self):
        try:
            super().run()
        except Exception as exc:  # noqa: BLE001 - any compiler failure
            self._fallback(exc)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # noqa: BLE001 - any compiler failure
            self._fallback(exc)

    @staticmethod
    def _fallback(exc):
        if os.environ.get("LABMAZE_REQUIRE_EXTENSIONS") == "1":
            raise exc
        sys.stderr.write(
            "WARNING: building labmaze's native extensions failed "
            f"({exc}). Installing the pure-Python package; labmaze.RandomMaze "
            "will be unavailable. Install a C++17 compiler to get it.\n"
        )


def _png_files(package_dir):
    paths = []
    for directory, _, filenames in os.walk(package_dir):
        for filename in fnmatch.filter(filenames, "*.png"):
            paths.append(
                os.path.relpath(os.path.join(directory, filename), package_dir)
            )
    return paths


setuptools.setup(
    name="labmaze-new",
    version="0.0.3",
    description=(
        "DeepMind Lab's text maze generator, with native extensions built by "
        "setuptools instead of Bazel."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="DeepMind",
    license="Apache 2.0",
    python_requires=">=3.10",
    install_requires=["absl-py", "numpy>=1.8.0"],
    packages=setuptools.find_packages(),
    package_data={"labmaze": _png_files("labmaze")},
    ext_modules=[
        _extension("_defaults", [os.path.join(CC_DIR, "python", "_defaults.cc")]),
        _extension(
            "_random_maze",
            [os.path.join(CC_DIR, "python", "_random_maze.cc"), *CORE_SOURCES],
        ),
    ],
    cmdclass={"build_ext": OptionalBuildExt},
    zip_safe=False,
)
