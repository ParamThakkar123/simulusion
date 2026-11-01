"""Efficient-inference runtime: precision, compile fallback, steppers, drift,
quantisation, export formats and deployment bundles. CPU only, no checkpoints."""

from __future__ import annotations

import types
import warnings

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from simulusion.inference import (  # noqa: E402
    DreamerStepModule,
    DreamerStepper,
    IRISStepper,
    InferenceModel,
    benchmark_step,
    load_bundle,
    make_stepper,
    optimize_for_inference,
    quantize_weights,
    resolve_precision,
    rollout_drift,
    save_bundle,
)
from simulusion.inference.quantize import Int8WeightOnlyLinear, weight_memory_bytes  # noqa: E402
from simulusion.utils.memory_utils import maybe_compile  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


class _TinyDreamer:
    """The attributes DreamerStepper reads, built from the real components."""

    def __init__(self, action_size: int = 3) -> None:
        from simulusion.models.dreamer_rssm import RSSM
        from simulusion.vision.dreamer_decoder import (
            ActionDecoder,
            ConvDecoder,
            DenseDecoder,
        )
        from simulusion.vision.dreamer_encoder import ConvEncoder

        stoch, deter, embed = 8, 16, 1024
        self.device = torch.device("cpu")
        self.action_size = action_size
        self.args = types.SimpleNamespace(action_noise=0.3)
        self.rssm = RSSM(action_size, stoch, deter, deter, embed, "elu")
        self.obs_encoder = ConvEncoder((3, 64, 64), embed, "relu")
        self.obs_decoder = ConvDecoder(stoch, deter, (3, 64, 64), "relu")
        self.actor = ActionDecoder(action_size, stoch, deter, 2, 32, "elu")
        self.reward_model = DenseDecoder(stoch, deter, (1,), 2, 32, "elu", "normal")
        for module in (
            self.rssm,
            self.obs_encoder,
            self.obs_decoder,
            self.actor,
            self.reward_model,
        ):
            module.eval()


@pytest.fixture
def dreamer():
    torch.manual_seed(0)
    return _TinyDreamer()


def _noise(batch: int, stoch: int, steps: int, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(batch, stoch, generator=gen),
            torch.randn(batch, stoch, generator=gen),
        )
        for _ in range(steps)
    ]


# --------------------------------------------------------------------------
# Precision
# --------------------------------------------------------------------------


def test_resolve_precision_aliases_and_auto_on_cpu():
    assert resolve_precision("float32", "cpu") == "fp32"
    assert resolve_precision("bfloat16", "cpu") == "bf16"
    assert resolve_precision(None, "cpu") == "fp32"
    assert resolve_precision("auto", "cpu") == "fp32"
    with pytest.raises(ValueError):
        resolve_precision("fp16", "cpu")
    with pytest.raises(ValueError):
        resolve_precision("int3", "cpu")


# --------------------------------------------------------------------------
# maybe_compile
# --------------------------------------------------------------------------


def test_maybe_compile_falls_back_when_first_call_fails(monkeypatch):
    def broken_compile(fn, mode=None):
        def compiled(*args, **kwargs):
            raise RuntimeError("no backend")

        return compiled

    monkeypatch.setattr(torch, "compile", broken_compile)
    fn = maybe_compile(lambda x: x + 1, enabled=True)
    with pytest.warns(RuntimeWarning, match="first call"):
        assert fn(1) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fn(2) == 3  # stays eager, no second warning
    assert fn.is_compiled is False


def test_maybe_compile_propagates_errors_after_success(monkeypatch):
    calls = {"n": 0}

    def flaky_compile(fn, mode=None):
        def compiled(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise ValueError("real error")
            return fn(*args, **kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", flaky_compile)
    fn = maybe_compile(lambda x: x * 2, enabled=True)
    assert fn(3) == 6
    with pytest.raises(ValueError, match="real error"):
        fn(3)


def test_maybe_compile_module_keeps_identity_and_state_dict_keys(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, mode=None: fn)
    module = nn.Linear(2, 2)
    keys = set(module.state_dict())
    compiled = maybe_compile(module, enabled=True)
    assert compiled is module
    assert set(compiled.state_dict()) == keys
    assert compiled(torch.zeros(1, 2)).shape == (1, 2)


def test_maybe_compile_disabled_is_identity():
    fn = lambda x: x  # noqa: E731
    assert maybe_compile(fn) is fn


# --------------------------------------------------------------------------
# InferenceModel
# --------------------------------------------------------------------------


def test_inference_model_fp32_is_exact():
    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    x = torch.randn(3, 8)
    with torch.no_grad():
        expected = module(x)
    wrapped = optimize_for_inference(module)
    assert isinstance(wrapped, InferenceModel)
    out = wrapped(x)
    assert torch.equal(out, expected)
    assert not any(p.requires_grad for p in module.parameters())
    assert not module.training


def test_inference_model_bf16_weights_return_fp32():
    torch.manual_seed(0)
    module = nn.Linear(8, 4)
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = module(x)
    wrapped = InferenceModel(module, precision="bf16", cast_weights=True)
    assert module.weight.dtype == torch.bfloat16
    out = wrapped(x)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected, atol=5e-2, rtol=5e-2)


def test_inference_model_output_dtype_none_and_clone(monkeypatch):
    module = nn.Linear(4, 4)
    wrapped = InferenceModel(
        module, precision="bf16", output_dtype=None, clone_outputs=True
    )
    out = wrapped(torch.randn(1, 4))
    assert out.dtype == torch.bfloat16


# --------------------------------------------------------------------------
# Steppers
# --------------------------------------------------------------------------


def test_make_stepper_picks_family(dreamer):
    assert isinstance(make_stepper(dreamer), DreamerStepper)
    agent = types.SimpleNamespace(dreamer=dreamer)
    assert isinstance(make_stepper(agent), DreamerStepper)
    with pytest.raises(TypeError):
        make_stepper(object())


def test_dreamer_stepper_loop_shapes(dreamer):
    stepper = DreamerStepper(dreamer, action_mode="mean")
    state = stepper.init_state(2)
    obs = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    with torch.inference_mode():
        state = stepper.observe(state, obs, None)
        action = stepper.act(state)
        assert action.shape == (2, 3)
        assert torch.all(action.abs() <= 1)
        step = stepper.imagine(state, action)
        assert step.reward.shape == (2,)
        assert step.state["deter"].shape == (2, 16)
        assert stepper.decode(state).shape == (2, 3, 64, 64)
        explore = stepper.act(state, explore=True)
        assert explore.shape == (2, 3)


def test_dreamer_observe_with_noise_is_deterministic(dreamer):
    stepper = DreamerStepper(dreamer)
    obs = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    noise = _noise(1, 8, 1)[0]
    with torch.inference_mode():
        a = stepper.observe(stepper.init_state(1), obs, None, noise=noise)
        b = stepper.observe(stepper.init_state(1), obs, None, noise=noise)
    for key in a:
        assert torch.equal(a[key], b[key])


def test_mean_action_is_deterministic_and_bounded(dreamer):
    features = torch.randn(4, 24)
    with torch.no_grad():
        first = dreamer.actor.mean_action(features)
        second = dreamer.actor.mean_action(features)
    assert torch.equal(first, second)
    assert torch.all(first.abs() < 1)


def test_step_module_matches_stepper(dreamer):
    stepper = DreamerStepper(dreamer, action_mode="mean")
    module = stepper.step_module()
    obs = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    prior_noise, post_noise = _noise(2, 8, 1)[0]
    init = stepper.init_state(2)
    prev_action = torch.zeros(2, 3)
    with torch.inference_mode():
        state = stepper.observe(init, obs, prev_action, noise=(prior_noise, post_noise))
        action = stepper.act(state)
        deter, stoch, mod_action = module(
            init["deter"],
            init["stoch"],
            prev_action,
            obs.float(),
            prior_noise,
            post_noise,
        )
    torch.testing.assert_close(deter, state["deter"])
    torch.testing.assert_close(stoch, state["stoch"])
    torch.testing.assert_close(mod_action, action)


class _FakeIRIS:
    device = torch.device("cpu")
    action_size = 4

    def __init__(self):
        self.lstm = nn.LSTM(6, 5, batch_first=True)
        self.head = nn.Linear(5, 4)

    def eval(self):
        return self

    def _init_lstm_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 5), torch.zeros(1, batch_size, 5)

    def forward_actor_critic(self, frames, hidden=None):
        feats = frames.flatten(2)[..., :6]
        out, hidden = self.lstm(feats, hidden)
        logits = self.head(out)
        return logits, logits.sum(-1), hidden


def test_iris_stepper_threads_lstm_state():
    stepper = IRISStepper(_FakeIRIS(), temperature=0)
    state = stepper.init_state(2)
    frame = torch.rand(2, 3, 4, 4)
    with pytest.raises(ValueError):
        stepper.act(state)
    with torch.inference_mode():
        s1 = stepper.observe(state, frame, None)
        s2 = stepper.observe(s1, frame, None)
        action = stepper.act(s2)
    assert action.shape == (2,)
    assert not torch.equal(s1["h"], s2["h"])  # the recurrent state advanced
    assert stepper.supports_imagination is False


# --------------------------------------------------------------------------
# Benchmark and drift
# --------------------------------------------------------------------------


def test_benchmark_step_reports_percentiles():
    x = torch.randn(4, 16)
    layer = nn.Linear(16, 16)
    report = benchmark_step(layer, x, warmup=2, iterations=20, batch_size=4)
    assert report.iterations == 20
    assert 0 < report.min_ms <= report.p50_ms <= report.p99_ms
    assert report.items_per_sec == pytest.approx(4 * report.steps_per_sec)
    assert "p50" in report.summary()
    assert report.peak_memory_mb is None  # CPU


def test_rollout_drift_zero_for_identical_and_grows_for_perturbed():
    torch.manual_seed(0)
    cell = nn.GRUCell(3, 8)
    actions = [(torch.randn(2, 3),) for _ in range(6)]
    h0 = torch.zeros(2, 8)

    def ref(h, a):
        return cell(a, h)

    def perturbed(h, a):
        return cell(a, h) + 1e-3

    same = rollout_drift(ref, ref, h0, actions)
    assert same.final_max_abs_error == 0.0
    assert same.within(atol=0.0)
    drift = rollout_drift(ref, perturbed, h0, actions)
    assert drift.steps == 6
    assert drift.final_max_abs_error > 0
    assert not drift.within(atol=1e-6)
    assert set(drift.as_dict()) >= {"max_abs_error", "rel_error"}


# --------------------------------------------------------------------------
# Quantisation
# --------------------------------------------------------------------------


def test_quantize_weights_replaces_large_linears_and_skips_patterns():
    torch.manual_seed(0)
    model = nn.Sequential()
    model.add_module("body", nn.Linear(128, 128))
    model.add_module("act", nn.ReLU())
    model.add_module("fc_state_prior", nn.Linear(128, 128))
    model.add_module("tiny", nn.Linear(128, 8))
    model.add_module("small_in", nn.Linear(8, 8))
    model.eval()
    x = torch.randn(4, 128)
    before_bytes = weight_memory_bytes(model)
    with torch.no_grad():
        body_expected = model.body(x)

    quantized = quantize_weights(model)
    assert quantized == ["body", "tiny"]
    assert isinstance(model.body, Int8WeightOnlyLinear)
    assert isinstance(model.fc_state_prior, nn.Linear)  # skipped by name
    assert isinstance(model.small_in, nn.Linear)  # below min_in_features
    assert weight_memory_bytes(model) < before_bytes
    with torch.no_grad():
        body_out = model.body(x)
    rel = (body_out - body_expected).norm() / body_expected.norm()
    assert rel < 1e-2


def test_quantize_rejects_unknown_backend():
    with pytest.raises(ValueError):
        quantize_weights(nn.Linear(64, 64), backend="bogus")


# --------------------------------------------------------------------------
# Export formats, verification and closed-loop drift
# --------------------------------------------------------------------------


def test_exported_program_round_trip_and_rollout_drift(dreamer, tmp_path):
    from simulusion.export import export_model, load_exported, verify_export

    module = DreamerStepModule(dreamer).eval()
    inputs = module.example_inputs(batch_size=1)
    path = export_model(
        module, tmp_path / "step.pt2", format="exported_program", example_inputs=inputs
    )
    assert path.exists()
    assert verify_export(module, path, inputs) < 1e-4

    loaded = load_exported(path)
    obs = torch.randint(0, 256, (1, 3, 64, 64)).float()
    action0 = torch.zeros(1, 3)
    noise = _noise(1, 8, 8)
    steps = [(obs, n[0], n[1]) for n in noise]

    def run(step_fn):
        def step(state, obs, prior_noise, post_noise):
            deter, stoch, action = state
            return step_fn(deter, stoch, action, obs, prior_noise, post_noise)

        return step

    init = (torch.zeros(1, 16), torch.zeros(1, 8), action0)
    report = rollout_drift(run(module), run(loaded), init, steps)
    assert report.steps == 8
    assert report.within(atol=1e-4)


def test_unknown_export_format_lists_new_formats(tmp_path):
    from simulusion.export import export_model

    with pytest.raises(ValueError, match="exported_program"):
        export_model(nn.Linear(2, 2), tmp_path / "x", format="nope")


def test_importing_simulusion_does_not_patch_nn_module():
    import simulusion  # noqa: F401
    import simulusion.export  # noqa: F401

    assert not hasattr(nn.Module, "export")


def test_every_public_module_class_is_exportable():
    """Public ``nn.Module`` classes carry ``.export()`` through the mixin."""
    import inspect

    import simulusion
    from simulusion.export import ExportableAgentMixin

    missing = []
    for name in simulusion.__all__:
        try:
            obj = getattr(simulusion, name)
        except ImportError:  # optional backend not installed
            continue
        if (
            inspect.isclass(obj)
            and issubclass(obj, nn.Module)
            and not issubclass(obj, ExportableAgentMixin)
        ):
            missing.append(name)
    assert not missing, f"public nn.Module classes without .export(): {missing}"


def test_public_module_export_method_round_trips(tmp_path):
    from simulusion import RMSNorm, load_exported

    norm = RMSNorm(4).eval()
    x = torch.randn(2, 4)
    path = norm.export(
        tmp_path / "norm.pt2", format="exported_program", example_inputs=x
    )
    torch.testing.assert_close(load_exported(path)(x), norm(x))


# --------------------------------------------------------------------------
# Bundles and CLI
# --------------------------------------------------------------------------


def test_bundle_round_trip(dreamer, tmp_path):
    module = DreamerStepModule(dreamer).eval()
    inputs = module.example_inputs(batch_size=1)
    bundle = save_bundle(
        tmp_path / "bundle",
        module,
        example_inputs=inputs,
        config={"stoch_size": 8},
        metadata={"env": "walker-walk"},
    )
    assert bundle.formats == ["exported_program"]
    assert bundle.config == {"stoch_size": 8}
    manifest = bundle.manifest
    assert manifest["inputs"][0] == {"shape": [1, 16], "dtype": "float32"}
    assert manifest["verification"]["exported_program"]["max_abs_error"] < 1e-4
    assert manifest["metadata"] == {"env": "walker-walk"}

    reloaded = load_bundle(tmp_path / "bundle")
    runner = reloaded.load_artifact()
    with torch.no_grad():
        expected = module(*inputs)
    for got, want in zip(runner(*inputs), expected):
        torch.testing.assert_close(got, want)

    fresh = DreamerStepModule(_TinyDreamer())
    reloaded.load_weights(fresh)
    for key, value in module.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value)


def test_weights_only_bundle_needs_no_inputs(tmp_path):
    bundle = save_bundle(tmp_path / "b", nn.Linear(3, 2), formats=())
    assert bundle.formats == []
    with pytest.raises(FileNotFoundError):
        bundle.load_artifact()


def test_load_bundle_rejects_non_bundle(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path)


def test_cli_deploy_inspect_and_bench(tmp_path):
    from click.testing import CliRunner

    from simulusion import cli

    torch.manual_seed(0)
    save_bundle(
        tmp_path / "lin",
        nn.Sequential(nn.Linear(4, 8), nn.ReLU()),
        example_inputs=(torch.zeros(2, 4),),
    )
    runner = CliRunner()
    inspect = runner.invoke(cli.app, ["deploy", "inspect", str(tmp_path / "lin")])
    assert inspect.exit_code == 0, inspect.output
    assert '"exported_program"' in inspect.output

    bench = runner.invoke(
        cli.app,
        ["deploy", "bench", str(tmp_path / "lin"), "--iterations", "5", "--json"],
    )
    assert bench.exit_code == 0, bench.output
    assert '"p99_ms"' in bench.output
    assert '"batch_size": 2' in bench.output

    missing = runner.invoke(cli.app, ["deploy", "inspect", str(tmp_path)])
    assert missing.exit_code != 0
