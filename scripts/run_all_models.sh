#!/usr/bin/env bash
#
# Train, then run inference, for every Simulusion model -- one model after another.
#
# Each model is a pair of stages: a training run scaled by --preset, followed by
# a recorded inference pass that picks up the checkpoint that training just
# wrote. A stage that fails is logged and the sweep moves on, so one broken
# model does not hide the state of the others. A summary table prints at the end
# and the exit status is non-zero if any stage failed. Ctrl+C ends the whole
# sweep, not just the stage it lands on.
#
# This is the end-to-end counterpart to scripts/benchmark_models.sh, which times
# forward/backward passes and never trains.
#
# Usage:
#   scripts/run_all_models.sh                       # tiny train+infer, every model
#   scripts/run_all_models.sh --preset small
#   scripts/run_all_models.sh --models diamond,iris --preset paper --device cuda
#   scripts/run_all_models.sh --train-only
#   scripts/run_all_models.sh --infer-only          # reuse existing checkpoints
#   scripts/run_all_models.sh --list
#   scripts/run_all_models.sh --dry-run --preset paper
#
# Presets:
#   tiny   A couple of epochs at batch 2. Minutes on CPU. Proves the loop runs
#          end to end; it learns nothing. This is the default.
#   small  The repo's single-GPU configs (iris_small_gpu.yaml,
#          jepa_small_gpu.yaml) and equivalent scale-downs elsewhere.
#   paper  The published configs, untouched. Days of GPU time per model.
#
# tiny and small change scale only -- batch sizes, step counts, epochs. Nothing
# that defines a method (token counts, masking geometry, imagination horizons,
# loss weights) is touched at any preset, so --preset paper reproduces the
# published configuration exactly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFER_PY="${SCRIPT_DIR}/benchmark_infer.py"
EXP_DIR="${REPO_ROOT}/simulusion/configs/experiments"

# Models with both a training entrypoint and a recorded-inference demo.
ALL_MODELS="diamond dreamer iris genie dit jepa"
# Train-only extras: no inference demo exists for these.
EXTRA_MODELS="planet rssm world-model"

PRESET="tiny"
MODELS=""
INCLUDE_EXTRA=0
LIST_ONLY=0
RUN_TRAIN=1
RUN_INFER=1
DEVICE=""
STEPS="120"
EPOCHS=""
TRAIN_STEPS=""
ENV_STEPS=""
UNTIL_CONVERGED=0
PATIENCE=""
MIN_DELTA=""
TRAIN_ARGS=()
SEED="0"
TIMEOUT="0"
DRY_RUN=0
FAIL_FAST=0
DO_SYNC=1
USE_UV=1
PYTHON_BIN="${PYTHON:-python}"
PYTHON_VERSION=""
SYNC_EXTRAS=()
CKPT_ROOT="${REPO_ROOT}/checkpoints"
OUT_DIR="${REPO_ROOT}/results/model_runs"
JEPA_DATA="${SIMULUSION_JEPA_DATA:-${IMAGENET_ROOT:-}}"
GENIE_DATASET="SONIC"
GENIE_DRY_RUN=0
GENIE_DATA_FILE=""
WM_ENV="Pendulum-v1"

usage() {
    cat <<'USAGE_EOF'
Usage: scripts/run_all_models.sh [flags]

Selection:
  --models a,b,c       Only these models (default: diamond,dreamer,iris,genie,dit,jepa).
  --all                Also run the train-only extras: planet, rssm, world-model.
  --preset tiny|small|paper
                       Scale of the training runs (default: tiny).
  --list               Show what would run, then exit.

Stages:
  --train-only         Train, skip inference.
  --infer-only         Skip training; use whatever checkpoints already exist.

Training length (override the preset, for real runs):
  --epochs N           Training epochs, for the models counted in epochs:
                       diamond, iris, dit, jepa, world-model.
  --train-steps N      Training steps, for the models counted in steps:
                       dreamer, genie.
  --env-steps N        Environment-step (data) budget for the Atari agents,
                       diamond and iris. Prefer this over --epochs for those
                       two: their epoch counts ARE the data budget (paper:
                       100000 = Atari 100k), so raising epochs alone makes IRIS
                       spin on a frozen replay buffer instead of seeing more
                       data. Assumes the config's per-epoch step counts.
  --train-arg ARG      Append ARG verbatim to the training command (repeatable).
                       Model-specific, so pair it with a single --models.
  --until-converged    Train until the model stops improving instead of for a
                       fixed length. The epoch/step count then becomes a
                       ceiling, so pair it with a generous --epochs.
                       diamond, iris and dreamer stop on evaluation return;
                       dit, jepa and genie on held-out loss. planet, rssm and
                       world-model have no convergence signal wired up.
  --patience N         Evaluations without improvement before stopping
                       (default: each model's own, 10).
  --min-delta X        Improvement needed to count, as a RELATIVE fraction of
                       the best value so far (default 1e-4, i.e. 0.01%).

Run control:
  --device NAME        cpu / cuda / cuda:0. Passed to every model that takes one.
  --steps N            Inference frames to record per model (default: 120).
                       For diamond and dreamer it also sets the length of the
                       imagined clip; for iris it is the per-episode cap.
  --seed N             Seed for training and inference (default: 0).
  --timeout SECONDS    Kill any stage that runs longer (0 = no limit). Needs
                       coreutils timeout. Useful for planet/rssm, which have no
                       CLI knobs to shorten them.
  --fail-fast          Stop at the first failing stage (default: keep going).
  --dry-run            Print every command instead of running it.

Data (models whose training needs a dataset):
  --jepa-data PATH     Image folder for I-JEPA. Also read from SIMULUSION_JEPA_DATA
                       or IMAGENET_ROOT. Without it JEPA trains on CIFAR-10,
                       which downloads itself.
  --genie-dataset NAME TinyWorlds dataset for Genie: SONIC (default), ZELDA or
                       POLE_POSITION. Downloads from HuggingFace on first use.
  --genie-dry-run      Only build the Genie trainer and exit, touching no
                       dataset. Use when you want no download.
  --genie-data-file PATH
                       Local TinyWorlds HDF5 file; avoids the download.
  --wm-env NAME        Gym env for the world-model trainer (default: Pendulum-v1).

DiT trains on CIFAR-10, which also downloads itself; it needs no data flag.

Output:
  --ckpt-root PATH     Where checkpoints are written (default: ./checkpoints).
  --out-dir PATH       Where videos and logs land (default: ./results/model_runs).
                       An index.html gallery of every recorded clip is written
                       here at the end of the sweep. An inference stage that
                       exits 0 without writing anything is reported as NO-DEMO
                       and counts as a failure.

Environment:
  --no-sync            Skip uv sync; use the environment as-is.
  --no-uv              Do not use uv at all; run with $PYTHON (default: python).
  --extra NAME         Add an optional dependency group to the sync (repeatable).
  --python VERSION     Interpreter for the uv environment, e.g. --python 3.12.
  -h, --help           This message.

Examples:
  scripts/run_all_models.sh --preset tiny --device cpu
  scripts/run_all_models.sh --models dreamer --preset small --device cuda
  scripts/run_all_models.sh --infer-only --models diamond,iris
  scripts/run_all_models.sh --all --timeout 300 --dry-run
USAGE_EOF
}

die() { echo "error: $*" >&2; exit 2; }
need_value() { [ "$2" -ge 2 ] || die "$1 needs a value"; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --models)          need_value "$1" "$#"; MODELS="$2"; shift 2 ;;
        --models=*)        MODELS="${1#*=}"; shift ;;
        --preset)          need_value "$1" "$#"; PRESET="$2"; shift 2 ;;
        --preset=*)        PRESET="${1#*=}"; shift ;;
        --all)             INCLUDE_EXTRA=1; shift ;;
        --list)            LIST_ONLY=1; shift ;;
        --train-only)      RUN_INFER=0; shift ;;
        --infer-only)      RUN_TRAIN=0; shift ;;
        --device)          need_value "$1" "$#"; DEVICE="$2"; shift 2 ;;
        --device=*)        DEVICE="${1#*=}"; shift ;;
        --steps)           need_value "$1" "$#"; STEPS="$2"; shift 2 ;;
        --steps=*)         STEPS="${1#*=}"; shift ;;
        --epochs)          need_value "$1" "$#"; EPOCHS="$2"; shift 2 ;;
        --epochs=*)        EPOCHS="${1#*=}"; shift ;;
        --train-steps)     need_value "$1" "$#"; TRAIN_STEPS="$2"; shift 2 ;;
        --train-steps=*)   TRAIN_STEPS="${1#*=}"; shift ;;
        --env-steps)       need_value "$1" "$#"; ENV_STEPS="$2"; shift 2 ;;
        --env-steps=*)     ENV_STEPS="${1#*=}"; shift ;;
        --until-converged) UNTIL_CONVERGED=1; shift ;;
        --patience)        need_value "$1" "$#"; PATIENCE="$2"; shift 2 ;;
        --patience=*)      PATIENCE="${1#*=}"; shift ;;
        --min-delta)       need_value "$1" "$#"; MIN_DELTA="$2"; shift 2 ;;
        --min-delta=*)     MIN_DELTA="${1#*=}"; shift ;;
        --train-arg)       need_value "$1" "$#"; TRAIN_ARGS+=("$2"); shift 2 ;;
        --train-arg=*)     TRAIN_ARGS+=("${1#*=}"); shift ;;
        --seed)            need_value "$1" "$#"; SEED="$2"; shift 2 ;;
        --seed=*)          SEED="${1#*=}"; shift ;;
        --timeout)         need_value "$1" "$#"; TIMEOUT="$2"; shift 2 ;;
        --timeout=*)       TIMEOUT="${1#*=}"; shift ;;
        --fail-fast)       FAIL_FAST=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        --jepa-data)       need_value "$1" "$#"; JEPA_DATA="$2"; shift 2 ;;
        --jepa-data=*)     JEPA_DATA="${1#*=}"; shift ;;
        --genie-dry-run)   GENIE_DRY_RUN=1; shift ;;
        --genie-dataset)   need_value "$1" "$#"; GENIE_DATASET="$2"; shift 2 ;;
        --genie-dataset=*) GENIE_DATASET="${1#*=}"; shift ;;
        --genie-data-file) need_value "$1" "$#"; GENIE_DATA_FILE="$2"; shift 2 ;;
        --genie-data-file=*) GENIE_DATA_FILE="${1#*=}"; shift ;;
        --wm-env)          need_value "$1" "$#"; WM_ENV="$2"; shift 2 ;;
        --wm-env=*)        WM_ENV="${1#*=}"; shift ;;
        --ckpt-root)       need_value "$1" "$#"; CKPT_ROOT="$2"; shift 2 ;;
        --ckpt-root=*)     CKPT_ROOT="${1#*=}"; shift ;;
        --out-dir)         need_value "$1" "$#"; OUT_DIR="$2"; shift 2 ;;
        --out-dir=*)       OUT_DIR="${1#*=}"; shift ;;
        --no-sync)         DO_SYNC=0; shift ;;
        --no-uv)           USE_UV=0; DO_SYNC=0; shift ;;
        --extra)           need_value "$1" "$#"; SYNC_EXTRAS+=("--extra" "$2"); shift 2 ;;
        --extra=*)         SYNC_EXTRAS+=("--extra" "${1#*=}"); shift ;;
        --python)          need_value "$1" "$#"; PYTHON_VERSION="$2"; shift 2 ;;
        --python=*)        PYTHON_VERSION="${1#*=}"; shift ;;
        -h|--help)         usage; exit 0 ;;
        *)                 die "unknown flag '$1' (try --help)" ;;
    esac
done

case "${PRESET}" in
    tiny|small|paper) ;;
    *) die "unknown preset '${PRESET}' (tiny, small or paper)" ;;
esac

if [ "${RUN_TRAIN}" -eq 0 ] && [ "${RUN_INFER}" -eq 0 ]; then
    die "--train-only and --infer-only cancel out"
fi

# ---------------------------------------------------------------- model table

selected_models() {
    if [ -n "${MODELS}" ]; then
        echo "${MODELS}" | tr ',' ' '
    elif [ "${INCLUDE_EXTRA}" -eq 1 ]; then
        echo "${ALL_MODELS} ${EXTRA_MODELS}"
    else
        echo "${ALL_MODELS}"
    fi
}

known_model() {
    case " ${ALL_MODELS} ${EXTRA_MODELS} " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

can_train() { [ -n "$1" ]; }

can_infer() {
    case " ${ALL_MODELS} " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

# The env a model is trained on at this preset. Inference has to replay the same
# one -- a Pendulum policy cannot be rolled out in walker-walk.
model_env() {
    case "$1" in
        diamond) echo "Breakout-v5" ;;
        iris)    echo "ALE/Pong-v5" ;;
        dreamer)
            if [ "${PRESET}" = "paper" ]; then echo "walker-walk"; else echo "Pendulum-v1"; fi
            ;;
        *) echo "" ;;
    esac
}

# ------------------------------------------------------------ command builders
#
# Each builder fills CMD. A non-empty SKIP_REASON means the stage cannot run
# here and is reported as SKIP rather than FAIL. NOTE is a caveat printed
# alongside the result without changing it.

CMD=()
SKIP_REASON=""
NOTE=""

# Optional third-party packages a model cannot run without, checked once so a
# missing extra reports as an actionable SKIP instead of a stack trace 30 lines
# into the traceback. Empty means "nothing beyond the base install".
missing_requirement() {
    local model="$1" module="" extra=""
    case "${model}" in
        # ALE registers the "ALE/" gym namespace; without it gym.make raises
        # NamespaceNotFound. Both Atari models need it.
        diamond|iris) module="ale_py"; extra="ale-py (in the 'gym' extra)" ;;
        *) return 0 ;;
    esac

    [ "${DRY_RUN}" -eq 1 ] && return 0
    if "${RUNNER[@]}" -c "import ${module}" >/dev/null 2>&1; then
        return 0
    fi
    SKIP_REASON="${extra} is not installed -- reinstall with --extra gym, or: uv pip install ale-py"
    return 1
}

# --epochs / --train-steps, translated to whatever each trainer calls its
# duration knob. Appended after the preset so the later value wins: both the
# OmegaConf dot-list and IRIS's runtime dict take the last occurrence of a key.
append_duration_overrides() {
    local model="$1"

    # --env-steps first, so an explicit --epochs below still wins on the total.
    if [ -n "${ENV_STEPS}" ]; then
        case "${model}" in
            diamond)
                # DIAMOND collects environment_steps_per_epoch every epoch with
                # no cap, so epochs x 100 IS the env-step budget (paper: 1000 x
                # 100 = 100k).
                CMD+=("num_epochs=$(( (ENV_STEPS + 99) / 100 ))")
                ;;
            iris)
                # IRIS collects only for the first collection_epochs, then keeps
                # training on what it gathered (paper: 500 x 200 = 100k, then
                # 100 more epochs). Move both, plus the hard cap, together.
                local collection=$(( (ENV_STEPS + 199) / 200 ))
                CMD+=("collection_epochs=${collection}"
                      "max_env_steps=${ENV_STEPS}"
                      "epochs=$(( collection + 100 ))")
                ;;
            *)
                NOTE="${NOTE:+${NOTE}; }--env-steps does not apply to ${model}"
                ;;
        esac
    fi

    if [ -n "${EPOCHS}" ]; then
        case "${model}" in
            diamond)     CMD+=("num_epochs=${EPOCHS}") ;;
            iris)        CMD+=("epochs=${EPOCHS}") ;;
            dit)         CMD+=("EPOCHS=${EPOCHS}") ;;
            jepa)        CMD+=("optimization.epochs=${EPOCHS}") ;;
            world-model) CMD+=(--vae_epochs "${EPOCHS}" --rnn_epochs "${EPOCHS}") ;;
            *)           NOTE="${NOTE:+${NOTE}; }--epochs does not apply to ${model}" ;;
        esac
    fi

    if [ -n "${TRAIN_STEPS}" ]; then
        case "${model}" in
            dreamer) CMD+=("total_steps=${TRAIN_STEPS}") ;;
            genie)
                # The dataset path takes key=value, the dry-run path a flag.
                if [ "${GENIE_DRY_RUN}" -eq 0 ]; then
                    CMD+=("max_steps=${TRAIN_STEPS}")
                else
                    CMD+=(--max-steps "${TRAIN_STEPS}")
                fi
                ;;
            *) NOTE="${NOTE:+${NOTE}; }--train-steps does not apply to ${model}" ;;
        esac
    fi

    append_convergence_overrides "${model}"

    if [ "${#TRAIN_ARGS[@]}" -gt 0 ]; then
        CMD+=("${TRAIN_ARGS[@]}")
    fi
}

# --until-converged, spelled the way each config names those fields. DiT uses
# the original codebase's UPPER_CASE, JEPA nests under `optimization`, and the
# rest are flat lower-case.
append_convergence_overrides() {
    local model="$1"
    [ "${UNTIL_CONVERGED}" -eq 1 ] || return 0

    case "${model}" in
        diamond|iris|dreamer|genie)
            CMD+=("early_stopping=true")
            [ -n "${PATIENCE}" ] && CMD+=("patience=${PATIENCE}")
            [ -n "${MIN_DELTA}" ] && CMD+=("min_delta=${MIN_DELTA}")
            ;;
        dit)
            CMD+=("EARLY_STOPPING=true")
            [ -n "${PATIENCE}" ] && CMD+=("PATIENCE=${PATIENCE}")
            [ -n "${MIN_DELTA}" ] && CMD+=("MIN_DELTA=${MIN_DELTA}")
            ;;
        jepa)
            CMD+=("optimization.early_stopping=true")
            [ -n "${PATIENCE}" ] && CMD+=("optimization.patience=${PATIENCE}")
            [ -n "${MIN_DELTA}" ] && CMD+=("optimization.min_delta=${MIN_DELTA}")
            # An imagefolder run needs data held out; CIFAR-10 uses its test
            # split and ignores this.
            CMD+=("data.val_split=0.05")
            ;;
        *)
            NOTE="${NOTE:+${NOTE}; }--until-converged is not wired up for ${model}"
            ;;
    esac
}

build_train_cmd() {
    local model="$1"
    CMD=(); SKIP_REASON=""; NOTE=""

    case "${model}" in
        diamond)
            CMD=("${RUNNER[@]}" -m simulusion.training.train_diamond "seed=${SEED}"
                 "checkpoint_dir=${CKPT_ROOT}/diamond")
            case "${PRESET}" in
                tiny)
                    # max_episode_steps caps the *evaluation* rollout. Left at
                    # its Atari default of 27000 an untrained policy that has
                    # not learnt to fire the ball never terminates the episode,
                    # so a preset advertised as "minutes" spent all of its time
                    # in evaluate() and never reached the first checkpoint.
                    CMD+=(preset=small num_epochs=2 training_steps_per_epoch=2
                          environment_steps_per_epoch=64 batch_size=2
                          num_sampling_steps=1 use_amp=false
                          data_loader_num_workers=0 pin_memory=false
                          persistent_workers=false max_episode_steps=200
                          save_interval=1 eval_interval=1 log_interval=1)
                    ;;
                small)
                    CMD+=(--config "${EXP_DIR}/diamond.yaml"
                          preset=small num_epochs=50 training_steps_per_epoch=100
                          environment_steps_per_epoch=100 batch_size=8
                          save_interval=10 eval_interval=10)
                    ;;
                paper)
                    CMD+=(--config "${EXP_DIR}/diamond.yaml")
                    ;;
            esac
            [ -n "${DEVICE}" ] && CMD+=("device=${DEVICE}")
            ;;

        dreamer)
            CMD=("${RUNNER[@]}" -m simulusion.training.train_dreamer
                 "logdir=${CKPT_ROOT}/dreamer" "seed=${SEED}")
            case "${PRESET}" in
                tiny)
                    CMD+=(env_backend=gym env=Pendulum-v1
                          total_steps=600 seed_steps=200 collect_steps=200
                          update_steps=2 batch_size=4 train_seq_len=10
                          max_episode_length=100 time_limit=100
                          checkpoint_interval=200 test_interval=1000000
                          scalar_freq=100 log_video_freq=-1)
                    ;;
                small)
                    CMD+=(env_backend=gym env=Pendulum-v1
                          total_steps=100000 seed_steps=5000 batch_size=16
                          train_seq_len=25 checkpoint_interval=10000)
                    ;;
                paper)
                    # Config defaults are the paper's: dmc walker-walk, 5M steps.
                    ;;
            esac
            # Dreamer selects its device with a flag, not a device string.
            [ "${DEVICE}" = "cpu" ] && CMD+=(no_gpu=true)
            ;;

        iris)
            CMD=("${RUNNER[@]}" -m simulusion.training.train_iris
                 "save_dir=${CKPT_ROOT}/iris" "seed=${SEED}")
            case "${PRESET}" in
                tiny)
                    # Scale levers only. tokens_per_frame, imagination_horizon,
                    # burn_in_length and every loss weight stay at paper values.
                    # max_episode_steps caps the evaluation rollout. At the
                    # Atari default of 27000 a single untrained Pong episode
                    # runs longer than the entire rest of this preset, and IRIS
                    # evaluates three times (epoch 0, epoch 1, and the final
                    # benchmark pass).
                    CMD+=(epochs=2 collection_epochs=1 env_steps_per_epoch=64
                          training_steps_per_epoch=2 transformer_steps_per_epoch=2
                          actor_critic_steps_per_epoch=2
                          autoencoder_batch_size=2 transformer_batch_size=2
                          actor_critic_batch_size=2
                          start_autoencoder_after=0 start_transformer_after=0
                          start_actor_critic_after=0
                          checkpoint_interval=1 eval_episodes=1
                          max_episode_steps=200
                          max_env_steps=128 use_amp=false)
                    ;;
                small)
                    CMD+=(--config "${EXP_DIR}/iris_small_gpu.yaml")
                    ;;
                paper)
                    CMD+=(--config "${EXP_DIR}/iris.yaml")
                    ;;
            esac
            [ -n "${DEVICE}" ] && CMD+=("device=${DEVICE}")
            ;;

        genie)
            if [ "${GENIE_DRY_RUN}" -eq 0 ]; then
                CMD=("${RUNNER[@]}" "${SCRIPT_DIR}/train_genie_tinyworlds.py"
                     "dataset=${GENIE_DATASET}"
                     "checkpoint_dir=${CKPT_ROOT}/genie")
                case "${PRESET}" in
                    tiny)
                        # GenieSmallConfig is 462M parameters, and batch_size=1
                        # does not change that -- the tiny preset OOMed on a 4GB
                        # card before the architecture came down too. These are
                        # width/depth only; the method (MaskGIT schedule, latent
                        # action vocabulary, mask probabilities) is untouched.
                        CMD+=(max_steps=20 batch_size=1 num_frames=8
                              num_workers=0 log_interval=5 val_interval=1000000
                              image_size=32
                              tokenizer_encoder_dim=64 tokenizer_decoder_dim=64
                              tokenizer_encoder_depth=1 tokenizer_decoder_depth=1
                              tokenizer_num_heads=2
                              action_encoder_dim=64 action_decoder_dim=64
                              action_encoder_depth=1 action_num_heads=2
                              dynamics_dim=64 dynamics_depth=1
                              dynamics_num_heads=2
                              checkpoint_interval=10)
                        ;;
                    small)
                        CMD+=(max_steps=5000 batch_size=2 log_interval=100
                              checkpoint_interval=500)
                        ;;
                    paper) CMD+=(max_steps=50000 checkpoint_interval=5000) ;;
                esac
                if [ -n "${GENIE_DATA_FILE}" ]; then
                    CMD+=("data_file=${GENIE_DATA_FILE}")
                else
                    NOTE="downloads the TinyWorlds dataset"
                fi
                [ -n "${DEVICE}" ] && CMD+=("device=${DEVICE}")
            else
                # Only reachable via --genie-dry-run now: build the trainer and
                # stop, without touching a dataset.
                CMD=("${RUNNER[@]}" -m simulusion.training.train_genie --dry-run)
                [ "${PRESET}" = "tiny" ] && CMD+=(--max-steps 20)
                [ -n "${DEVICE}" ] && CMD+=(--device "${DEVICE}")
                NOTE="trainer construction only (--genie-dry-run)"
            fi
            ;;

        jepa)
            CMD=("${RUNNER[@]}" -m simulusion.training.train_jepa)
            case "${PRESET}" in
                tiny)
                    # batch_size is the only lever on run length here: I-JEPA
                    # has no per-epoch step cap, so "one epoch" is the whole
                    # dataset. At batch 2 that is 25000 iterations over
                    # CIFAR-10 -- about an hour, for a preset meant to take
                    # minutes -- and since weights are only written at the end
                    # of an epoch, a --timeout mid-epoch leaves nothing behind.
                    # The per-step overhead (mask collation, an EMA pass over
                    # every parameter) dominates at this size, so a larger
                    # batch cuts wall-clock roughly proportionally.
                    CMD+=(meta.model_name=vit_tiny meta.use_bfloat16=false
                          data.batch_size=64 data.num_workers=0
                          optimization.epochs=1 optimization.warmup=0)
                    NOTE="one epoch over ${JEPA_DATA:-the dataset}"
                    ;;
                small) CMD+=(--config "${EXP_DIR}/jepa_small_gpu.yaml") ;;
                paper) CMD+=(--config "${EXP_DIR}/jepa.yaml") ;;
            esac
            # After --config so these win over the file.
            CMD+=("logging.folder=${CKPT_ROOT}/jepa")
            if [ -n "${JEPA_DATA}" ]; then
                CMD+=("data.dataset=imagefolder" "data.root_path=${JEPA_DATA}")
            else
                # No image folder given: CIFAR-10 downloads itself, so JEPA
                # trains rather than skipping. 32x32 images need a crop and
                # patch size to match -- the paper's 224/16 would upsample
                # every image 7x.
                #
                # min_keep has to drop with them. That 32/4 crop leaves an 8x8
                # grid, on which a (0.15, 0.2)-scale target block is 8 to 12
                # patches; measured over 500 batches, the paper's min_keep of 10
                # rejects 40% of them and the sampler raises. 4 clears the
                # smallest block with margin.
                CMD+=("data.dataset=cifar10" "data.download=true"
                      "data.root_path=${REPO_ROOT}/data"
                      "data.crop_size=32" "mask.patch_size=4" "mask.min_keep=4")
                NOTE="${NOTE:+${NOTE}; }CIFAR-10 at 32px, min_keep=4 (not paper geometry); pass --jepa-data for your own images"
            fi
            ;;

        planet)
            CMD=("${RUNNER[@]}" -m simulusion.training.train_planet
                 --outdir "${CKPT_ROOT}/planet")
            case "${PRESET}" in
                # Episode length is left alone: the trainer samples 50-step
                # traces, so anything under that leaves nothing to sample.
                tiny)  CMD+=(--epochs 1 --iters 5) ;;
                small) CMD+=(--epochs 10 --iters 50) ;;
                paper) ;;
            esac
            [ -n "${DEVICE}" ] && CMD+=(--device "${DEVICE}")
            ;;

        rssm)
            CMD=("${RUNNER[@]}" -m simulusion.training.train_rssm)
            NOTE="no CLI knobs; length is fixed in the module -- use --timeout"
            ;;

        world-model)
            CMD=("${RUNNER[@]}" -m simulusion.training.train_world_model
                 --env "${WM_ENV}"
                 --logdir "${CKPT_ROOT}/world_model"
                 --data_dir "${CKPT_ROOT}/world_model/data")
            case "${PRESET}" in
                tiny)
                    CMD+=(--num_rollouts 4 --vae_epochs 1 --rnn_epochs 1
                          --vae_batch_size 4 --rnn_batch_size 2 --seq_len 8
                          --ctrl_pop_size 2 --ctrl_samples 1 --ctrl_workers 1
                          --ctrl_time_limit 100 --stage vae)
                    NOTE="VAE stage only"
                    ;;
                small)
                    CMD+=(--num_rollouts 100 --vae_epochs 10 --rnn_epochs 10
                          --ctrl_workers 2)
                    ;;
                paper) ;;
            esac
            [ -n "${DEVICE}" ] && CMD+=(--device "${DEVICE}")
            ;;

        dit)
            # CIFAR-10 downloads itself into ROOT_PATH, so this needs no
            # dataset flag. DiT.fit picks its own device; there is no override.
            CMD=("${RUNNER[@]}" -m simulusion.training.train_dit
                 "WORKDIR=${CKPT_ROOT}/dit"
                 "ROOT_PATH=${REPO_ROOT}/data")
            case "${PRESET}" in
                tiny)  CMD+=(EPOCHS=1 BATCH=32 WIDTH=128 DEPTH=4 HEADS=4 EMA=false) ;;
                small) CMD+=(EPOCHS=50 BATCH=128 CHECKPOINT_EVERY=10) ;;
                # 400 epochs is days of GPU time; without an interval the only
                # write is after the last one.
                paper) CMD+=(EPOCHS=400 CHECKPOINT_EVERY=25) ;;
            esac
            [ -n "${DEVICE}" ] && NOTE="DiT.fit selects its own device; --device ignored"
            ;;

        *)
            SKIP_REASON="unknown model"
            ;;
    esac

    [ -z "${SKIP_REASON}" ] && append_duration_overrides "${model}"
    return 0
}

# Newest existing file among the arguments, or empty.
newest_match() {
    local newest="" candidate
    for candidate in "$@"; do
        [ -e "${candidate}" ] || continue
        if [ -z "${newest}" ] || [ "${candidate}" -nt "${newest}" ]; then
            newest="${candidate}"
        fi
    done
    echo "${newest}"
}

find_checkpoint() {
    local model="$1" found=""
    # Unmatched globs must expand to nothing rather than to themselves.
    shopt -s nullglob
    case "${model}" in
        diamond)
            found="$(newest_match "${CKPT_ROOT}"/diamond/checkpoint_*.pt \
                                  "${CKPT_ROOT}"/diamond/*.pt)"
            ;;
        iris)
            found="$(newest_match "${CKPT_ROOT}"/iris/final_*.pt \
                                  "${CKPT_ROOT}"/iris/best_*.pt \
                                  "${CKPT_ROOT}"/iris/checkpoint_*.pt)"
            ;;
        dreamer)
            found="$(newest_match "${CKPT_ROOT}"/dreamer/ckpts/*_ckpt.pt \
                                  "${CKPT_ROOT}"/dreamer/*/ckpts/*_ckpt.pt)"
            ;;
        genie)
            found="$(newest_match "${CKPT_ROOT}"/genie/*.pt)"
            ;;
        jepa)
            found="$(newest_match "${CKPT_ROOT}"/jepa/*.pth.tar \
                                  "${CKPT_ROOT}"/jepa/*.pt)"
            ;;
        dit)
            # DiT.fit writes dit_model.pth (the EMA weights when EMA is on).
            found="$(newest_match "${CKPT_ROOT}"/dit/dit_model.pth \
                                  "${CKPT_ROOT}"/dit/*.pth)"
            ;;
    esac
    shopt -u nullglob
    echo "${found}"
}

build_infer_cmd() {
    local model="$1"
    CMD=(); SKIP_REASON=""; NOTE=""

    local checkpoint env
    checkpoint="$(find_checkpoint "${model}")"

    CMD=("${RUNNER[@]}" "${INFER_PY}" --mode record --model "${model}"
         --out-dir "${OUT_DIR}/videos" --steps "${STEPS}" --seed "${SEED}")
    [ -n "${DEVICE}" ] && CMD+=(--device "${DEVICE}")

    env="$(model_env "${model}")"
    [ -n "${env}" ] && CMD+=(--game "${env}")

    case "${model}" in
        diamond|dreamer|iris)
            if [ -z "${checkpoint}" ]; then
                SKIP_REASON="no checkpoint under ${CKPT_ROOT}/${model} (train first)"
                return
            fi
            CMD+=(--checkpoint "${checkpoint}")
            [ "${model}" = "iris" ] && CMD+=(--episodes 1)
            # Both of these can imagine forward as well as act, and the
            # imagined clip is the one that shows the world model rather than
            # the policy. Recorded next to the real rollout, and stitched.
            case "${model}" in
                diamond|dreamer) CMD+=(--dream-steps "${STEPS}") ;;
            esac
            ;;
        genie|dit|jepa)
            # These demos run without weights, so an untrained sweep still
            # exercises the whole generation path.
            if [ -n "${checkpoint}" ]; then
                CMD+=(--checkpoint "${checkpoint}")
            else
                CMD+=(--random-init)
                NOTE="random init (no checkpoint found)"
            fi
            ;;
        *)
            SKIP_REASON="no inference demo for this model"
            ;;
    esac
}

# --------------------------------------------------------------------- listing

MODEL_LIST="$(selected_models)"
for model in ${MODEL_LIST}; do
    known_model "${model}" || \
        die "unknown model '${model}'. Known: ${ALL_MODELS} ${EXTRA_MODELS}"
done

if [ "${LIST_ONLY}" -eq 1 ]; then
    stages=""
    [ "${RUN_TRAIN}" -eq 1 ] && stages="train"
    [ "${RUN_INFER}" -eq 1 ] && stages="${stages}${stages:+ }infer"
    echo "preset: ${PRESET}    stages: ${stages}"
    echo
    printf '%-14s %-7s %-7s %s\n' "MODEL" "TRAIN" "INFER" "NOTES"
    for model in ${MODEL_LIST}; do
        trainable="yes"; inferable="yes"; notes=""
        can_train "${model}" || { trainable="no"; notes="inference only"; }
        can_infer "${model}" || { inferable="no"; notes="train only"; }
        case "${model}" in
            jepa)  [ -z "${JEPA_DATA}" ] && notes="CIFAR-10 (downloads)" ;;
            dit)   notes="CIFAR-10 (downloads)" ;;
            genie)
                if [ "${GENIE_DRY_RUN}" -eq 1 ]; then
                    notes="dry-run: builds the trainer only"
                else
                    notes="TinyWorlds ${GENIE_DATASET} (downloads)"
                fi
                ;;
            rssm)   notes="fixed length; use --timeout" ;;
        esac
        printf '%-14s %-7s %-7s %s\n' "${model}" "${trainable}" "${inferable}" "${notes}"
    done
    exit 0
fi

# ----------------------------------------------------------------- environment

ensure_uv() {
    command -v uv >/dev/null 2>&1 && return 0

    # A previous install may be on disk but not on PATH.
    local candidate
    for candidate in "${HOME}/.local/bin/uv" "${HOME}/.cargo/bin/uv"; do
        if [ -x "${candidate}" ]; then
            PATH="$(dirname "${candidate}"):${PATH}"
            export PATH
            return 0
        fi
    done

    echo "uv not found -- installing it from https://astral.sh/uv ..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        die "need curl or wget to install uv, or install it yourself and re-run with --no-uv"
    fi

    PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    export PATH
    command -v uv >/dev/null 2>&1 || die "uv still not on PATH after installing"
}

cd "${REPO_ROOT}"

if [ "${USE_UV}" -eq 1 ] && [ "${DRY_RUN}" -eq 0 ]; then
    ensure_uv
    echo "uv: $(uv --version)"
fi

if [ "${DO_SYNC}" -eq 1 ] && [ "${DRY_RUN}" -eq 0 ]; then
    # Training and recording both need Gymnasium and OpenCV; pull those in
    # unless the caller already named the group.
    extras_flat=" ${SYNC_EXTRAS[*]-} "
    case "${extras_flat}" in *" gym "*) ;; *) SYNC_EXTRAS+=("--extra" "gym") ;; esac
    case "${extras_flat}" in *" viz "*) ;; *) SYNC_EXTRAS+=("--extra" "viz") ;; esac

    # --inexact installs what the lock requires without uninstalling anything
    # else already in the environment.
    sync_cmd=(uv sync --inexact)
    [ -n "${PYTHON_VERSION}" ] && sync_cmd+=(--python "${PYTHON_VERSION}")
    sync_cmd+=("${SYNC_EXTRAS[@]}")
    echo "installing dependencies: ${sync_cmd[*]}"
    "${sync_cmd[@]}"
fi

if [ "${USE_UV}" -eq 1 ]; then
    # --no-sync: the environment is already resolved above.
    RUNNER=(uv run --no-sync python -u)
else
    RUNNER=("${PYTHON_BIN}" -u)
fi

LOG_DIR="${OUT_DIR}/logs"
if [ "${DRY_RUN}" -eq 0 ]; then
    mkdir -p "${LOG_DIR}" "${OUT_DIR}/videos" "${CKPT_ROOT}"
fi

if [ "${TIMEOUT}" != "0" ] && ! command -v timeout >/dev/null 2>&1; then
    echo "warning: --timeout ${TIMEOUT} ignored -- no 'timeout' on PATH" >&2
    TIMEOUT="0"
fi

# ------------------------------------------------------------------- execution

ROWS=()
ARTIFACTS=()
FAILURES=0
ABORTED=0

record_row() {
    # model | stage | status | duration | detail
    ROWS+=("$1|$2|$3|$4|$5")
}

# Demo files an inference stage wrote. Recorded as "model|path" so the gallery
# at the end can group them, and counted so a stage that exits 0 without
# producing anything is reported rather than passing silently -- a recorder
# whose writer fails on the last frame still returns 0.
#
# Detection is by mtime against a marker touched immediately before the stage,
# not by comparing directory listings: a re-run overwrites the same filenames,
# so a listing diff would call every repeat run empty.
STAGE_MARKER="${LOG_DIR:-.}/.stage_marker"
VIDEO_DIR="${OUT_DIR}/videos"

mark_stage_start() {
    [ "${DRY_RUN}" -eq 1 ] && return 0
    : > "${STAGE_MARKER}"
}

# Echoes one path per line; empty when the stage wrote nothing.
stage_artifacts() {
    [ -d "${VIDEO_DIR}" ] || return 0
    [ -f "${STAGE_MARKER}" ] || return 0
    find "${VIDEO_DIR}" -type f -newer "${STAGE_MARKER}" 2>/dev/null | LC_ALL=C sort
}

# Ctrl+C must end the sweep, not just the stage it lands on. Without this,
# `set -e` is suspended around the stage (failures are meant to be survivable)
# so bash would shrug off the interrupt and march into the next model, where
# the next Ctrl+C would kill that one too.
on_interrupt() {
    ABORTED=1
    echo
    echo ">> interrupted -- stopping after the current stage" >&2
}
trap on_interrupt INT TERM

run_stage() {
    local model="$1" stage="$2"
    local log="${LOG_DIR}/${model}.${stage}.log"

    CMD=(); SKIP_REASON=""; NOTE=""
    if missing_requirement "${model}"; then
        if [ "${stage}" = "train" ]; then
            build_train_cmd "${model}"
        else
            build_infer_cmd "${model}"
        fi
    fi

    if [ -n "${SKIP_REASON}" ]; then
        echo ">> ${model} / ${stage}: SKIP -- ${SKIP_REASON}"
        record_row "${model}" "${stage}" "SKIP" "-" "${SKIP_REASON}"
        return 0
    fi

    local launch=("${CMD[@]}")
    if [ "${TIMEOUT}" != "0" ]; then
        launch=(timeout --preserve-status -k 10 "${TIMEOUT}" "${CMD[@]}")
    fi

    echo
    echo "===================================================================="
    echo ">> ${model} / ${stage} (preset ${PRESET})"
    [ -n "${NOTE}" ] && echo "   note: ${NOTE}"
    echo "   ${launch[*]}"
    echo "===================================================================="

    if [ "${DRY_RUN}" -eq 1 ]; then
        record_row "${model}" "${stage}" "DRY" "-" "${NOTE}"
        return 0
    fi

    mark_stage_start
    local start=${SECONDS} status=0
    set +e
    "${launch[@]}" 2>&1 | tee "${log}"
    status=${PIPESTATUS[0]}
    set -e
    local elapsed=$((SECONDS - start))

    if [ "${status}" -eq 0 ]; then
        if [ "${stage}" != "infer" ]; then
            record_row "${model}" "${stage}" "OK" "${elapsed}s" "${NOTE}"
            return 0
        fi

        # A recorder that exits 0 having written nothing is a failure of the
        # demo, not a pass. Catching it here is the whole point of running the
        # sweep rather than just the trainers.
        local written count=0 file
        written="$(stage_artifacts)"
        if [ -n "${written}" ]; then
            while IFS= read -r file; do
                [ -n "${file}" ] || continue
                ARTIFACTS+=("${model}|${file}")
                count=$((count + 1))
            done <<<"${written}"
        fi

        if [ "${count}" -eq 0 ]; then
            FAILURES=$((FAILURES + 1))
            record_row "${model}" "${stage}" "NO-DEMO" "${elapsed}s"                 "exited 0 but wrote nothing to ${VIDEO_DIR}"
            echo ">> ${model} / ${stage}: exited 0 but produced no demo file" >&2
            [ "${FAIL_FAST}" -eq 1 ] && return 1
            return 0
        fi

        record_row "${model}" "${stage}" "OK" "${elapsed}s"             "${count} file(s)${NOTE:+; ${NOTE}}"
        return 0
    fi

    # Anything above 128 is a signal, not a bug in the model: Ctrl+C (130),
    # `kill` (143), or --timeout firing. Report it as such and stop -- counting
    # a signal as a failure would blame the model for the operator.
    if [ "${status}" -gt 128 ]; then
        local signal=$((status - 128))
        if [ "${TIMEOUT}" != "0" ] && [ "${signal}" -eq 15 ]; then
            record_row "${model}" "${stage}" "TIMEOUT" "${elapsed}s" "hit --timeout ${TIMEOUT}s"
            echo ">> ${model} / ${stage} hit the ${TIMEOUT}s timeout" >&2
            return 0
        fi
        ABORTED=1
        record_row "${model}" "${stage}" "INT(${signal})" "${elapsed}s" "stopped by signal ${signal}"
        echo ">> ${model} / ${stage} stopped by signal ${signal}" >&2
        return 1
    fi

    FAILURES=$((FAILURES + 1))
    record_row "${model}" "${stage}" "FAIL(${status})" "${elapsed}s" "log: ${log}"
    echo ">> ${model} / ${stage} FAILED (exit ${status}); see ${log}" >&2
    [ "${FAIL_FAST}" -eq 1 ] && return 1
    return 0
}

echo "models:      ${MODEL_LIST}"
echo "preset:      ${PRESET}"
echo "device:      ${DEVICE:-auto}"
echo "checkpoints: ${CKPT_ROOT}"
echo "output:      ${OUT_DIR}"

for model in ${MODEL_LIST}; do
    if [ "${ABORTED}" -eq 1 ]; then
        break
    fi
    ran_any=0
    if [ "${RUN_TRAIN}" -eq 1 ] && can_train "${model}"; then
        ran_any=1
        run_stage "${model}" "train" || break
    fi
    if [ "${RUN_INFER}" -eq 1 ] && can_infer "${model}" && [ "${ABORTED}" -eq 0 ]; then
        ran_any=1
        run_stage "${model}" "infer" || break
    fi
    if [ "${ran_any}" -eq 0 ]; then
        # Asked for a stage this model does not have, e.g. --train-only dit.
        echo ">> ${model}: nothing to do for the selected stage(s)"
        record_row "${model}" "-" "SKIP" "-" "no such stage for this model"
    fi
done

# --------------------------------------------------------------------- summary

echo
echo "============================== summary ============================="
printf '%-14s %-6s %-10s %-8s %s\n' "MODEL" "STAGE" "STATUS" "TIME" "DETAIL"
for row in ${ROWS[@]+"${ROWS[@]}"}; do
    IFS='|' read -r r_model r_stage r_status r_time r_detail <<<"${row}"
    printf '%-14s %-6s %-10s %-8s %s\n' \
        "${r_model}" "${r_stage}" "${r_status}" "${r_time}" "${r_detail}"
done
echo "===================================================================="

# A gallery over everything the inference stages wrote, so the sweep ends with
# something you can look at rather than a directory of loose files. Paths are
# relative to the page, so the whole out-dir can be copied or served as-is.
# Which model a demo file belongs to, by longest matching name prefix.
model_for_file() {
    local name="$1" candidate best=""
    for candidate in ${ALL_MODELS} ${EXTRA_MODELS}; do
        case "${name}" in
            "${candidate}"*)
                [ "${#candidate}" -gt "${#best}" ] && best="${candidate}"
                ;;
        esac
    done
    echo "${best:-other}"
}

# Demo files, ordered so each model's files stay together and the models come
# out in the order the sweep runs them.
gallery_files() {
    local candidate file
    [ -d "${VIDEO_DIR}" ] || return 0
    for candidate in ${ALL_MODELS} ${EXTRA_MODELS} ""; do
        for file in $(find "${VIDEO_DIR}" -maxdepth 1 -type f 2>/dev/null                       | LC_ALL=C sort); do
            if [ "$(model_for_file "$(basename "${file}")")" = "${candidate:-other}" ]
            then
                echo "${file}"
            fi
        done
    done
}

write_gallery() {
    local page="${OUT_DIR}/index.html" model path rel size current=""

    {
        cat <<'HTML_HEAD'
<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulusion model runs</title>
<style>
  :root { color-scheme: light dark; --fg: #16181d; --bg: #fbfbfa; --muted: #6b7280; --line: #e3e3e0; --card: #fff; }
  @media (prefers-color-scheme: dark) {
    :root { --fg: #e8e8e6; --bg: #16181d; --muted: #9aa0aa; --line: #2c2f36; --card: #1d2027; }
  }
  body { margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
         font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
  .sub { color: var(--muted); margin: 0 0 2.5rem; }
  h2 { font-size: 1.05rem; margin: 2.5rem 0 .9rem; padding-bottom: .4rem;
       border-bottom: 1px solid var(--line); text-transform: lowercase; letter-spacing: .02em; }
  .grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
  figure { margin: 0; background: var(--card); border: 1px solid var(--line);
           border-radius: 10px; overflow: hidden; }
  figure video, figure img { display: block; width: 100%; background: #000;
                             image-rendering: pixelated; }
  figcaption { padding: .55rem .7rem; font-size: 12.5px; color: var(--muted);
               word-break: break-all; }
  figcaption b { color: var(--fg); font-weight: 600; }
</style>
<main>
HTML_HEAD
        printf '<h1>Simulusion model runs</h1>
'
        printf '<p class="sub">preset <b>%s</b> &middot; device %s &middot; %s</p>
'             "${PRESET}" "${DEVICE:-auto}" "$(date '+%Y-%m-%d %H:%M')"

        # Everything in the video directory, not just what this invocation
        # produced: re-running one model (--models jepa) would otherwise
        # republish a gallery containing only that model and drop the rest.
        # Files are attributed by filename prefix, which is why every recorder
        # names its output after its model.
        for path in $(gallery_files); do
            model="$(model_for_file "$(basename "${path}")")"
            rel="${path#"${OUT_DIR}/"}"
            size="$(du -h "${path}" 2>/dev/null | cut -f1)"

            if [ "${model}" != "${current}" ]; then
                [ -n "${current}" ] && printf '</div>
'
                printf '<h2>%s</h2>
<div class="grid">
' "${model}"
                current="${model}"
            fi

            case "${rel}" in
                *.mp4|*.webm)
                    printf '<figure><video src="%s" controls loop muted playsinline></video>' "${rel}" ;;
                *.png|*.jpg|*.jpeg|*.gif)
                    printf '<figure><img src="%s" alt="%s">' "${rel}" "${rel}" ;;
                *)
                    printf '<figure>' ;;
            esac
            printf '<figcaption><b>%s</b><br>%s</figcaption></figure>
'                 "$(basename "${rel}")" "${size:-?}"
        done
        [ -n "${current}" ] && printf '</div>
'
        printf '</main>
'
    } > "${page}"

    echo "gallery:     ${page}"
}

if [ "${DRY_RUN}" -eq 0 ]; then
    echo "logs:        ${LOG_DIR}"
    echo "videos:      ${VIDEO_DIR}"
    echo "checkpoints: ${CKPT_ROOT}"
    if [ -n "$(gallery_files)" ]; then
        write_gallery
    fi
    rm -f "${STAGE_MARKER}"
fi

if [ "${ABORTED}" -eq 1 ]; then
    echo
    echo "interrupted; remaining models were not run." >&2
    exit 130
fi

if [ "${FAILURES}" -gt 0 ]; then
    echo
    echo "${FAILURES} stage(s) failed." >&2
    exit 1
fi
