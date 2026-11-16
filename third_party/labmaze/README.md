# labmaze-new

DeepMind's [labmaze](https://github.com/google-deepmind/labmaze) maze
generator. This version builds its native extensions with setuptools and
pybind11 instead of Bazel, so `labmaze.RandomMaze` works on current Python
versions, including CPython 3.13 on Windows.

Memory Maze and `dm_control.locomotion` need `labmaze.RandomMaze`, which lives
in the compiled `_random_maze` extension. Upstream `labmaze` 1.0.6 ships wheels
only up to CPython 3.12 and otherwise builds from source with Bazel.
labmaze-new 0.0.2 installed on 3.13 by skipping the extensions, so
`RandomMaze` was missing.

```bash
pip install labmaze-new
```

From this repository:

```bash
pip install ./third_party/labmaze
```

Building the extensions needs a C++17 compiler (MSVC on Windows, gcc or clang
elsewhere). Without one, the install falls back to the pure-Python package, as
in 0.0.2, and `labmaze.RandomMaze` is unavailable. Set
`LABMAZE_REQUIRE_EXTENSIONS=1` to make a failed build an error instead.

`labmaze-new` and `labmaze` install the same `labmaze` import package, so
install only one of them.

## Changes in 0.0.3

- Native extensions build with setuptools and pybind11; Bazel is no longer
  needed. The `LABMAZE_BUILD_BAZEL_EXTENSIONS` switch is gone, because the
  extensions are always attempted.
- `labmaze/cc/char_grid.{h,cc}`, `labmaze/cc/random_maze.{h,cc}`: replaced
  `absl::string_view` with `std::string_view`, and `absl::StrSplit(...,
  absl::SkipEmpty())` with a local splitter that behaves the same. Abseil is no
  longer a dependency.
- Removed the Bazel `BUILD`/`WORKSPACE` files and the C++ gtest sources.

Sources are otherwise those of labmaze-new 0.0.2, which matches upstream
labmaze 1.0.6. Licensed under Apache 2.0; see `LICENSE` and `AUTHORS`.
