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

"$PYTHON" -m pip install -U pip setuptools wheel numpy dm-env Pillow

wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 \
    -O /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
git clone --depth 1 https://github.com/google-deepmind/lab.git
cd lab

cat > .bazelrc << EOF
build --enable_workspace
build --python_version=PY3
build --action_env=PYTHON_BIN_PATH=$(command -v "$PYTHON")
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
