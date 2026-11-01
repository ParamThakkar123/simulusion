PYTHON ?= python
export PYTHONPYCACHEPREFIX ?= build/__pycache__

.PHONY: test lint format install-dmc bench bench-all bench-infer bench-play run-all run-all-train run-all-infer

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

# DeepMind Control, including on CPython 3.13 where dm-control's `labmaze`
# dependency has no wheel and would build with Bazel. See simulusion/install_dmc.py.
install-dmc:
	$(PYTHON) -m simulusion.install_dmc

# Compute benchmark (params, latency, throughput, memory) over every model.
# The shell driver installs everything with uv and runs the sweep with uv run.
# Pass extra flags with BENCH_ARGS, e.g. `make bench BENCH_ARGS="--preset small"`.
bench:
	bash scripts/benchmark_models.sh $(BENCH_ARGS)

bench-all:
	bash scripts/benchmark_models.sh --all --preset small $(BENCH_ARGS)

# Record a video of a trained model playing / generating.
# Example: make bench-infer BENCH_ARGS="--model diamond -c ckpt.pt --game Breakout-v5"
bench-infer:
	bash scripts/benchmark_models.sh --infer $(BENCH_ARGS)

# Interactive play (needs a display). Add --versus to play against the policy.
# Example: make bench-play BENCH_ARGS="--model iris -c ckpt.pt --versus"
bench-play:
	bash scripts/benchmark_models.sh --play $(BENCH_ARGS)

# Train, then record inference, for every model in sequence. Preset scales the
# training runs: tiny (default, minutes on CPU), small (single-GPU configs),
# paper (published configs). Pass extra flags with RUN_ARGS.
# Example: make run-all RUN_ARGS="--preset small --device cuda"
run-all:
	bash scripts/run_all_models.sh $(RUN_ARGS)

run-all-train:
	bash scripts/run_all_models.sh --train-only $(RUN_ARGS)

run-all-infer:
	bash scripts/run_all_models.sh --infer-only $(RUN_ARGS)
