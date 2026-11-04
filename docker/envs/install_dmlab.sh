#!/bin/bash
# Build DeepMind Lab from source and install it into the active Python.
#
# DeepMind Lab ships no wheels. This follows the build recipe of the DreamerV3
# reference container (github.com/danijar/dreamerv3, Dockerfile): Bazel 7 via
# Bazelisk (Bazel 8 removed the native sh_binary rule the lab repo uses), a
# Python-3-only BUILD target, and headers taken from the local interpreter.
# It also fetches the Brady/Konkle/Oliva (2008) object images that the
# psychlab_* levels of DMLab-30 need.
#
# Linux x86-64, Python <= 3.11, run as root. Takes 10-30 minutes.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
WORKDIR="${DMLAB_BUILD_DIR:-/tmp/dmlab-build}"
DATA_DIR="${DMLAB_DATA_DIR:-/opt/dmlab_data}"
export USE_BAZEL_VERSION="${USE_BAZEL_VERSION:-7.4.1}"

apt-get update
apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl freeglut3-dev gettext git libffi-dev \
    libglu1-mesa libglu1-mesa-dev libjpeg-dev liblua5.1-0-dev libosmesa6-dev \
    libsdl2-dev lua5.1 pkg-config unzip wget zip zlib1g-dev g++
rm -rf /var/lib/apt/lists/*

# deepmind_lab imports pkg_resources, which setuptools 81 removed.
"$PYTHON" -m pip install -U pip "setuptools<81" wheel numpy dm-env Pillow

wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 \
    -O /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
git clone https://github.com/google-deepmind/lab.git
cd lab
git checkout b1db91af5b4d2f3a24466f4632a3e5e1b0829cca  # last commit, 2023-01-04

# The WORKSPACE fetches several dependencies from their default branches,
# which have moved on (current abseil-cpp, for one, loads rules_cc symbols this
# Bazel setup lacks). Pin each to the release current at the lab's last commit.
"$PYTHON" - << 'EOF'
import re

path = "WORKSPACE"
text = open(path).read()
gh = "https://github.com/%s/archive/refs/tags/%s.tar.gz"
pins = {
    "com_google_absl": (
        "abseil-cpp-20230125.3",
        gh % ("abseil/abseil-cpp", "20230125.3"),
        "5366d7e7fa7ba0d915014d387b66d0d002c03236448e1ba9ef98122c13b35c36",
    ),
    "com_google_absl_py": (
        "abseil-py-1.4.0",
        gh % ("abseil/abseil-py", "v1.4.0"),
        "0fb3a4916a157eb48124ef309231cecdfdd96ff54adf1660b39c0d4a9790a2c0",
    ),
    "com_google_googletest": (
        "googletest-1.13.0",
        gh % ("google/googletest", "v1.13.0"),
        "ad7fdba11ea011c1d925b3289cf4af2c66a352e18d4c7264392fead75e919363",
    ),
    "bazel_skylib": (
        "bazel-skylib-1.3.0",
        gh % ("bazelbuild/bazel-skylib", "1.3.0"),
        "3b620033ca48fcd6f5ef2ac85e0f6ec5639605fa2f627968490e52fc91a9932f",
    ),
    "pybind11_archive": (
        "pybind11-2.10.3",
        gh % ("pybind/pybind11", "v2.10.3"),
        "5d8c4c5dda428d3a944ba3d2a5212cb988c2fae4670d58075a5a49075a6ca315",
    ),
    "dm_env_archive": (
        "dm_env-1.5",
        gh % ("deepmind/dm_env", "v1.5"),
        "a35d10fb4a70a00187a23f128bc4041560847154b8cd1441bbdf2845f5ad3519",
    ),
    "tree_archive": (
        "tree-0.1.8",
        gh % ("deepmind/tree", "0.1.8"),
        "5139f26b0995514978fc214d20c8302d95c35ce58e5e1adc3278b159d39f6a52",
    ),
}
for name, (prefix, url, sha) in pins.items():
    # Edit the rule in place so attributes such as build_file are kept.
    pattern = r'http_archive\(\s*name = "%s",.*?\n\)' % re.escape(name)
    match = re.search(pattern, text, flags=re.S)
    assert match, name
    rule = match.group(0)
    rule = re.sub(r'\n    sha256 = "[^"]*",', "", rule)
    rule = re.sub(r'strip_prefix = "[^"]*"', 'strip_prefix = "%s"' % prefix, rule)
    rule = re.sub(r"urls = \[[^\]]*\]", 'urls = ["%s"]' % url, rule, flags=re.S)
    rule = rule.replace(
        'name = "%s",' % name, 'name = "%s",\n    sha256 = "%s",' % (name, sha), 1
    )
    text = text[: match.start()] + rule + text[match.end() :]
open(path, "w").write(text)
EOF

cat > .bazelrc << EOF
build --enable_workspace
build --python_version=PY3
build --action_env=PYTHON_BIN_PATH=$(command -v "$PYTHON")
# Schedule compile actions within half the RAM, so small machines (or a
# memory-capped WSL/Docker VM) swap instead of having compilers OOM-killed.
build --local_ram_resources=HOST_RAM*.5
EOF

# The upstream BUILD declares PY2 and PY3 game binaries; keep only PY3.
sed -i '/\[py_binary(/,/\]]/c\
py_binary(\
    name = "python_game_py3",\
    srcs = ["examples/game_main.py"],\
    data = ["//:deepmind_lab.so"],\
    main = "examples/game_main.py",\
    python_version = "PY3",\
    srcs_version = "PY3",\
    tags = ["manual"],\
    deps = ["@six_archive//:six"],\
)' BUILD

# Point Bazel at this interpreter's Python and NumPy headers.
cat > python_system.bzl << 'EOF'
_BUILD_FILE = '''
exports_files(["defs.bzl"])

cc_library(
    name = "python_headers",
    hdrs = glob(["python3/**/*.h", "numpy3/**/*.h"]),
    includes = ["python3", "numpy3"],
    visibility = ["//visibility:public"],
)
'''

_GET_PYTHON_INCLUDE_DIR = """
import sys
import sysconfig
from numpy import get_include
sys.stdout.write("{}\\n{}\\n".format(sysconfig.get_paths()["include"], get_include()))
""".strip()

def _python_repo_impl(repository_ctx):
    repository_ctx.file("BUILD", _BUILD_FILE)
    result = repository_ctx.execute(["python3", "-c", _GET_PYTHON_INCLUDE_DIR])
    if result.return_code:
        fail("Failed to run local Python3 interpreter: %s" % result.stderr)
    pypath, nppath = result.stdout.splitlines()
    repository_ctx.symlink(pypath, "python3")
    repository_ctx.symlink(nppath, "numpy3")

python_repo = repository_rule(
    implementation = _python_repo_impl,
    configure = True,
    local = True,
    attrs = {"py_version": attr.string(default = "PY3", values = ["PY3"])},
)
EOF

bazel build -c opt //python/pip_package:build_pip_package --verbose_failures
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
"$PYTHON" -m pip install --force-reinstall /tmp/dmlab_pkg/deepmind_lab-*.whl
bazel shutdown || true

# Bazel 7 packages the runfiles under _main/; move them where the module looks.
DEST="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')/deepmind_lab"
if [ -d "$DEST/_main" ]; then
    cp -r "$DEST/_main/." "$DEST/"
    rm -rf "$DEST/_main"
fi
cd /
rm -rf "$WORKDIR" /tmp/dmlab_pkg "$HOME/.cache/bazel" "$HOME/.cache/bazelisk"

# Object images for the psychlab_* levels.
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"
curl -fsSL https://bradylab.ucsd.edu/stimuli/ObjectsAll.zip -o ObjectsAll.zip
unzip -q ObjectsAll.zip
"$PYTHON" - << 'EOF'
import os
from PIL import Image

src = "OBJECTSALL"
files = sorted(f for f in os.listdir(src) if f.lower().endswith("jpg"))
for i, name in enumerate(files):
    Image.open(os.path.join(src, name)).save("%04d.png" % (i + 1))
EOF
rm -rf __MACOSX OBJECTSALL ObjectsAll.zip
ESCAPED_DATA_DIR=$(printf '%s' "$DATA_DIR" | sed 's/[\/&]/\\&/g')
sed -i "s|DATASET_PATH = ''|DATASET_PATH = '$ESCAPED_DATA_DIR'|g" \
    "$DEST/baselab/game_scripts/datasets/brady_konkle_oliva2008.lua"

"$PYTHON" -c "import deepmind_lab; deepmind_lab.Lab('contributed/dmlab30/explore_goal_locations_small', []).reset()"
"$PYTHON" -c "import deepmind_lab; deepmind_lab.Lab('contributed/dmlab30/psychlab_arbitrary_visuomotor_mapping', []).reset()"
echo "DeepMind Lab installed"
