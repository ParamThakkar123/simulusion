#!/usr/bin/env bash
#
# Auto-run the Simulusion model benchmark, or run inference / interactive play.
#
# Compute (default): installs uv if it is missing, installs the project with
# `uv sync`, times every architecture, and prints the results table. Needs no
# checkpoints.
#
# Inference: pass --infer to record a video of a trained model playing (or
# generating), or --play for an interactive window -- play WITH the policy
# (keys override it) or AGAINST it (--versus).
#
# Usage:
#   scripts/benchmark_models.sh                          # compute, tiny scale
#   scripts/benchmark_models.sh --preset small --all
#   scripts/benchmark_models.sh --infer --model diamond -c ckpt.pt --game Breakout-v5
#   scripts/benchmark_models.sh --play  --model dreamer -c ckpt.pt --versus
#   scripts/benchmark_models.sh --no-sync --models dit
#
# Wrapper flags are consumed here; every other flag is forwarded.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_PY="${SCRIPT_DIR}/benchmark_models.py"
INFER_PY="${SCRIPT_DIR}/benchmark_infer.py"
DEFAULT_OUT_DIR="${REPO_ROOT}/results/model_benchmarks"
DEFAULT_INFER_OUT_DIR="${REPO_ROOT}/results/model_inference"

DO_SYNC=1
PYTHON_VERSION=""
SYNC_EXTRAS=()
BENCH_ARGS=()
RUN_MODE="compute"

usage() {
    cat <<'EOF'
Usage: scripts/benchmark_models.sh [wrapper flags] [benchmark or inference flags]

Wrapper flags:
  --no-sync            Skip `uv sync`; use the existing environment as-is.
  --extra NAME         Install an optional dependency group (repeatable),
                       e.g. --extra viz --extra gym.
  --python VERSION     Interpreter for the environment, e.g. --python 3.12.
  --infer              Record a video of a model playing / generating, then exit.
  --play               Open an interactive play window (needs a display).
  --uv-help            Show this message.

Compute flags (default mode) go to scripts/benchmark_models.py:

  --preset tiny|small|paper   Scale to build and feed each model at.
  --all                       Include the heavy tier (full IRIS/Genie stacks).
  --models a,b / --family f   Narrow the sweep.
  --device auto|cpu|cuda|mps  Where to run.
  --dtype fp32|fp16|bf16      Autocast precision.
  --no-backward               Time the forward pass only (not video inference).
  --iters N / --warmup N      Timing loop.
  --out-dir PATH              Where reports are written.
  --list                      Show what would run and exit.

Inference / play flags (--infer or --play) go to scripts/benchmark_infer.py:

  --model NAME                diamond, dreamer, iris, genie, dit or jepa.
  -c / --checkpoint PATH      Trained weights (required except Genie/DiT/JEPA
                              --random-init).
  --game / -g NAME            Env id (Breakout-v5, walker-walk, ALE/Pong-v5).
  --out-dir PATH              Videos land here (default: results/model_inference).
  --steps N                   [record] how many env frames to write.
  --dream-steps N             [diamond record] imagination frames (0 to skip).
  --control assist|human|versus
                              assist (default): play WITH the model, keys override.
                              human: you always drive.
                              versus / --versus: you drive; the policy's action
                              is shown as the opponent.
  --record PATH               [play] also save an MP4 while the window is open.
  --list                      List inference models and exit.

Examples:
  scripts/benchmark_models.sh --infer --model diamond -c ckpt.pt
  scripts/benchmark_models.sh --play --model iris -c ckpt.pt --game ALE/Pong-v5
  scripts/benchmark_models.sh --play --model diamond -c ckpt.pt --versus
  scripts/benchmark_models.sh --infer --model genie --random-init
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-sync)
            DO_SYNC=0
            shift
            ;;
        --infer)
            if [ "${RUN_MODE}" != "compute" ]; then
                echo "error: pass only one of --infer or --play" >&2
                exit 2
            fi
            RUN_MODE="infer"
            shift
            ;;
        --play)
            if [ "${RUN_MODE}" != "compute" ]; then
                echo "error: pass only one of --infer or --play" >&2
                exit 2
            fi
            RUN_MODE="play"
            shift
            ;;
        --extra)
            [ "$#" -ge 2 ] || { echo "error: --extra needs a value" >&2; exit 2; }
            SYNC_EXTRAS+=("--extra" "$2")
            shift 2
            ;;
        --extra=*)
            SYNC_EXTRAS+=("--extra" "${1#*=}")
            shift
            ;;
        --python)
            [ "$#" -ge 2 ] || { echo "error: --python needs a value" >&2; exit 2; }
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --python=*)
            PYTHON_VERSION="${1#*=}"
            shift
            ;;
        --uv-help)
            usage
            exit 0
            ;;
        *)
            BENCH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Reports / videos land here unless the caller redirected them.
if [ "${RUN_MODE}" = "compute" ]; then
    OUT_DIR="${DEFAULT_OUT_DIR}"
else
    OUT_DIR="${DEFAULT_INFER_OUT_DIR}"
fi
if [ "${#BENCH_ARGS[@]}" -gt 0 ]; then
    for index in "${!BENCH_ARGS[@]}"; do
        case "${BENCH_ARGS[$index]}" in
            --out-dir)
                next=$((index + 1))
                [ "${next}" -lt "${#BENCH_ARGS[@]}" ] && OUT_DIR="${BENCH_ARGS[$next]}"
                ;;
            --out-dir=*)
                OUT_DIR="${BENCH_ARGS[$index]#*=}"
                ;;
        esac
    done
fi

# Play/record need OpenCV and usually Gymnasium. Pull those extras in unless
# the caller already named them (or skipped the install).
if [ "${RUN_MODE}" != "compute" ] && [ "${DO_SYNC}" -eq 1 ]; then
    extras_flat=" ${SYNC_EXTRAS[*]} "
    case "${extras_flat}" in
        *" viz "*) ;;
        *) SYNC_EXTRAS+=("--extra" "viz") ;;
    esac
    case "${extras_flat}" in
        *" gym "*) ;;
        *) SYNC_EXTRAS+=("--extra" "gym") ;;
    esac
fi

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    # A previous install may be on disk but not on PATH.
    local candidate
    for candidate in "${HOME}/.local/bin/uv" "${HOME}/.cargo/bin/uv"; do
        if [ -x "${candidate}" ]; then
            PATH="$(dirname "${candidate}"):${PATH}"
            export PATH
            return
        fi
    done

    echo "uv not found -- installing it from https://astral.sh/uv ..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "error: need curl or wget to install uv, or install it yourself:" >&2
        echo "       https://docs.astral.sh/uv/getting-started/installation/" >&2
        exit 1
    fi

    PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    export PATH
    if ! command -v uv >/dev/null 2>&1; then
        echo "error: uv still not on PATH after installing" >&2
        exit 1
    fi
}

cd "${REPO_ROOT}"

ensure_uv
echo "uv: $(uv --version)"

if [ "${DO_SYNC}" -eq 1 ]; then
    # --inexact: install what the lock requires without uninstalling anything
    # else already in the environment (optional extras, editable installs).
    sync_cmd=(uv sync --inexact)
    [ -n "${PYTHON_VERSION}" ] && sync_cmd+=(--python "${PYTHON_VERSION}")
    [ "${#SYNC_EXTRAS[@]}" -gt 0 ] && sync_cmd+=("${SYNC_EXTRAS[@]}")
    echo "installing dependencies: ${sync_cmd[*]}"
    "${sync_cmd[@]}"
else
    echo "skipping install (--no-sync)"
fi

if [ "${RUN_MODE}" = "compute" ]; then
    RUN_PY="${BENCH_PY}"
    RUN_ARGS=(${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"})
else
    RUN_PY="${INFER_PY}"
    if [ "${RUN_MODE}" = "play" ]; then
        RUN_ARGS=("--mode" "play")
    else
        RUN_ARGS=("--mode" "record")
    fi
    RUN_ARGS+=(${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"})
    # Inference writes under OUT_DIR unless the caller already passed --out-dir.
    has_out=0
    for arg in ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}; do
        case "${arg}" in
            --out-dir|--out-dir=*) has_out=1 ;;
        esac
    done
    if [ "${has_out}" -eq 0 ]; then
        RUN_ARGS+=("--out-dir" "${OUT_DIR}")
    fi
fi

echo
echo "running: uv run python -u ${RUN_PY} ${RUN_ARGS[*]-}"
echo

# Timestamp reference: a report older than this is left over from an earlier
# run (--list, --no-report, or a run that died), so it must not be echoed as if
# it were this run's result.
marker="${TMPDIR:-/tmp}/.simulusion-bench-marker.$$"
: > "${marker}"
trap 'rm -f "${marker}"' EXIT

# `--no-sync`: the environment is already resolved above, so do not re-check it.
set +e
uv run --no-sync python -u "${RUN_PY}" ${RUN_ARGS[@]+"${RUN_ARGS[@]}"}
status=$?
set -e

if [ "${RUN_MODE}" = "compute" ]; then
    report="${OUT_DIR}/model_benchmarks.md"
    if [ -f "${report}" ] && [ "${report}" -nt "${marker}" ]; then
        echo
        echo "=============================== results ================================"
        cat "${report}"
        echo "========================================================================"
        echo "json:     ${OUT_DIR}/model_benchmarks.json"
        echo "csv:      ${OUT_DIR}/model_benchmarks.csv"
        echo "markdown: ${report}"
    fi
else
    echo
    echo "inference artifacts: ${OUT_DIR}"
    if [ -d "${OUT_DIR}" ]; then
        ls -1 "${OUT_DIR}" 2>/dev/null || true
    fi
fi

exit "${status}"
