import os
import pickle
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch


class TestUtils:
    _needs_cv2 = {
        "test_to_tensor_obs",
        "test_save_video_chw",
        "test_save_video_hwc",
        "test_save_video_invalid_dims",
        "test_combine_videos",
        "test_combine_videos_no_files",
        "test_StreamingVideoWriter",
        "test_StreamingVideoWriter_float_frame",
        "test_TorchImageEnvWrapper_reset",
    }
    _needs_plotly = {"test_plot_metrics"}
    _needs_gym = {"test_TorchImageEnvWrapper_init_with_string"}

    @pytest.fixture(autouse=True)
    def _deps(self, request):
        if request.function.__name__ in self._needs_cv2:
            pytest.importorskip("cv2")
        if request.function.__name__ in self._needs_plotly:
            pytest.importorskip("plotly")
        if request.function.__name__ in self._needs_gym:
            pytest.importorskip("gymnasium")

    def test_to_tensor_obs(self):
        from simulusion.utils.utils import to_tensor_obs

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = to_tensor_obs(image)

        assert result.shape == (3, 64, 64)
        assert result.dtype == torch.float32

    def test_postprocess_img(self):
        from simulusion.utils.utils import postprocess_img

        image = np.random.rand(64, 64, 3).astype(np.float32) * 255
        result = postprocess_img(image, depth=8)

        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)
        assert result.min() >= 0 and result.max() <= 255

    def test_preprocess_img(self):
        from simulusion.utils.utils import preprocess_img

        image = torch.rand(3, 64, 64) * 255
        preprocess_img(image, depth=8)

        assert image.min() >= -0.5 and image.max() <= 0.5

    def test_bottle(self):
        from simulusion.utils.utils import bottle

        def sum_fn(x):
            return x.sum(dim=-1)

        tensors = [torch.randn(2, 3, 10)]
        result = bottle(sum_fn, *tensors)

        assert result.shape == (2, 3)

    def test_get_combined_params(self):
        from simulusion.utils.utils import get_combined_params

        model1 = torch.nn.Linear(10, 5)
        model2 = torch.nn.Linear(5, 3)

        params = get_combined_params(model1, model2)

        assert len(params) == 4

    def test_load_yml_config(self):
        from simulusion.utils.utils import load_yml_config
        import yaml
        import shutil

        tmpdir = tempfile.mkdtemp()
        config_path = os.path.join(tmpdir, "test.yaml")
        try:
            with open(config_path, "w") as f:
                yaml.dump({"test": {"value": 1, "name": "test"}}, f)

            config = load_yml_config(config_path)
            assert config["test"]["value"] == 1
        finally:
            shutil.rmtree(tmpdir)

    def test_ensure_results_dir_exists(self):
        from simulusion.utils.utils import ensure_results_dir_exists

        with tempfile.TemporaryDirectory() as tmpdir:
            ensure_results_dir_exists(tmpdir)

        with pytest.raises(FileNotFoundError):
            ensure_results_dir_exists("/nonexistent/path")

    def test_flatten_dict(self):
        from simulusion.utils.utils import flatten_dict

        data = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        result = flatten_dict(data)

        assert result["a"] == 1
        assert result["b.c"] == 2
        assert "b.d.e" in result or "d.e" in result

    def test_normalize_frames_for_saving(self):
        from simulusion.utils.utils import normalize_frames_for_saving

        frames_chw = np.random.rand(10, 3, 64, 64).astype(np.float32)
        result = normalize_frames_for_saving(frames_chw)

        assert result.shape == (10, 64, 64, 3)
        assert result.min() >= 0 and result.max() <= 1

        frames_hwc = np.random.rand(10, 64, 64, 3).astype(np.float32)
        result = normalize_frames_for_saving(frames_hwc)

        assert result.shape == (10, 64, 64, 3)

    def test_normalize_frames_with_neg_half(self):
        from simulusion.utils.utils import normalize_frames_for_saving

        frames = np.random.rand(10, 64, 64, 3).astype(np.float32) - 0.5
        result = normalize_frames_for_saving(frames)

        assert result.min() >= 0 and result.max() <= 1

    def test_get_mask_1d(self):
        from simulusion.utils.utils import get_mask

        tensor = torch.tensor([1.0, 2.0, 3.0])
        lengths = [2, 3, 1]

        mask = get_mask(tensor, lengths)

        assert mask.shape == (3, 3)
        assert mask[0, 0] == 1.0
        assert mask[0, 1] == 1.0
        assert mask[0, 2] == 0.0

    def test_get_mask_2d(self):
        from simulusion.utils.utils import get_mask

        tensor = torch.randn(2, 5, 3)
        lengths = [3, 5]

        mask = get_mask(tensor, lengths)

        assert mask.shape == (2, 5, 3)

    def test_load_memory_blocks_untrusted_pickle_globals(self, tmp_path):
        from simulusion.utils.utils import load_memory

        class Evil:
            def __reduce__(self):
                return (eval, ("1 + 1",))

        replay_path = tmp_path / "evil_replay.pkl"
        replay_path.write_bytes(pickle.dumps(Evil()))

        with pytest.raises(pickle.UnpicklingError, match="not allowed"):
            load_memory(replay_path, torch.device("cpu"))

    def test_load_memory_trusted_flag_still_uses_restricted_unpickler(self, tmp_path):
        from simulusion.utils.utils import load_memory

        class Evil:
            def __reduce__(self):
                return (eval, ("1 + 1",))

        replay_path = tmp_path / "evil_trusted_replay.pkl"
        replay_path.write_bytes(pickle.dumps(Evil()))

        with pytest.warns(DeprecationWarning, match="unrestricted pickle loading"):
            with pytest.raises(pickle.UnpicklingError, match="not allowed"):
                load_memory(replay_path, torch.device("cpu"), trusted=True)

    def test_apply_model_placeholder(self):
        from simulusion.utils.utils import apply_model

        model = Mock()
        inputs = torch.randn(2, 3)
        result = apply_model(model, inputs)

        assert result is None

    def test_tensorboard_metrics(self):
        from simulusion.utils.utils import TensorBoardMetrics

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = TensorBoardMetrics(tmpdir)
            metrics.update({"loss": 1.0, "accuracy": 0.9})
            metrics.update({"loss": 0.8})

            assert metrics.steps["loss"] == 2
            assert metrics.steps["accuracy"] == 1

    def test_plot_metrics(self):
        from simulusion.utils.utils import plot_metrics
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = {"loss": [1.0, 0.5, 0.3]}
            plot_metrics(metrics, tmpdir, prefix="test_")

            html_file = os.path.join(tmpdir, "test_loss.html")
            assert os.path.exists(html_file)

    def test_save_video_chw(self):
        from simulusion.utils.utils import save_video
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            frames = np.random.rand(5, 3, 64, 64).astype(np.float32)
            path = save_video(frames, tmpdir, "test_video")

            assert os.path.exists(path)

    def test_save_video_hwc(self):
        from simulusion.utils.utils import save_video
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            frames = np.random.rand(5, 64, 64, 3).astype(np.float32)
            path = save_video(frames, tmpdir, "test_video")

            assert os.path.exists(path)

    def test_save_video_invalid_dims(self):
        from simulusion.utils.utils import save_video

        with tempfile.TemporaryDirectory() as tmpdir:
            frames = np.random.rand(5, 64, 64).astype(np.float32)

            with pytest.raises(ValueError):
                save_video(frames, tmpdir, "test_video")

    def test_combine_videos(self):
        from simulusion.utils.utils import combine_videos
        import os
        import cv2

        with tempfile.TemporaryDirectory() as tmpdir:
            frames = np.random.rand(5, 64, 64, 3).astype(np.float32) * 255
            frames_u8 = frames.astype(np.uint8)

            for i in range(3):
                writer = cv2.VideoWriter(
                    os.path.join(tmpdir, f"vid_{i}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    25.0,
                    (64, 64),
                    True,
                )
                for frame in frames_u8:
                    writer.write(frame)
                writer.release()

            result = combine_videos(tmpdir, output_name="combined.mp4")

            assert os.path.exists(result)

    def test_combine_videos_no_files(self):
        from simulusion.utils.utils import combine_videos

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                combine_videos(tmpdir)

    def test_apply_masks(self):
        from simulusion.utils.utils import apply_masks

        x = torch.randn(2, 10, 5)
        mask1 = torch.tensor([0, 1, 2]).unsqueeze(0).expand(2, -1)
        mask2 = torch.tensor([3, 4, 5]).unsqueeze(0).expand(2, -1)

        result = apply_masks(x, [mask1, mask2])

        assert result.shape == (4, 3, 5)

    def test_TorchImageEnvWrapper_init_with_string(self):
        from simulusion.utils.utils import TorchImageEnvWrapper

        with patch("gymnasium.make") as mock_make:
            mock_env = Mock()
            mock_env.reset.return_value = np.zeros(100)
            mock_env.action_space = Mock()
            mock_env.action_space.shape = (2,)
            mock_make.return_value = mock_env

            wrapper = TorchImageEnvWrapper("CartPole-v1", bit_depth=8)
            assert wrapper.bit_depth == 8

    def test_TorchImageEnvWrapper_reset(self):
        from simulusion.utils.utils import TorchImageEnvWrapper

        mock_env = Mock()
        mock_env.reset.return_value = (np.zeros((64, 64, 3), dtype=np.uint8), {})
        mock_env.render.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

        wrapper = TorchImageEnvWrapper(mock_env, bit_depth=8)
        result = wrapper.reset()

        assert result.shape == (3, 64, 64)

    def test_TorchImageEnvWrapper_observation_size(self):
        from simulusion.utils.utils import TorchImageEnvWrapper

        mock_env = Mock()
        mock_env.reset.return_value = (np.zeros((64, 64, 3), dtype=np.uint8), {})
        mock_env.render.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_env.action_space.sample.return_value = np.array([0])

        wrapper = TorchImageEnvWrapper(mock_env, bit_depth=8)
        assert wrapper.observation_size == (3, 64, 64)

    def test_TorchImageEnvWrapper_action_size(self):
        from simulusion.utils.utils import TorchImageEnvWrapper

        mock_env = Mock()
        mock_env.reset.return_value = (np.zeros((64, 64, 3), dtype=np.uint8), {})
        mock_env.render.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_action_space = Mock()
        mock_action_space.shape = (4,)
        mock_env.action_space = mock_action_space

        wrapper = TorchImageEnvWrapper(mock_env, bit_depth=8)
        assert wrapper.action_size == 4

    def test_TorchImageEnvWrapper_discrete_action(self):
        from simulusion.utils.utils import TorchImageEnvWrapper

        mock_env = Mock()
        mock_env.reset.return_value = (np.zeros((64, 64, 3), dtype=np.uint8), {})
        mock_env.render.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_action_space = Mock()
        mock_action_space.n = 5
        mock_action_space.shape = None
        mock_action_space.sample.return_value = 0
        mock_env.action_space = mock_action_space

        wrapper = TorchImageEnvWrapper(mock_env, bit_depth=8)
        assert wrapper.action_size == 1

    def test_StreamingVideoWriter(self):
        from simulusion.utils.utils import StreamingVideoWriter
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.mp4")
            writer = StreamingVideoWriter(path, fps=20, frame_shape=(64, 64))

            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            writer.write_frame(frame)
            writer.close()

            assert os.path.exists(path)

    def test_StreamingVideoWriter_float_frame(self):
        from simulusion.utils.utils import StreamingVideoWriter
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.mp4")
            writer = StreamingVideoWriter(path, fps=20, frame_shape=(64, 64))

            frame = np.random.rand(64, 64, 3).astype(np.float32)
            writer.write_frame(frame)
            writer.close()

            assert os.path.exists(path)


def _video_codec(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return fourcc.to_bytes(4, "little").decode("ascii", "replace").lower(), frames


def test_mp4_writer_writes_browser_playable_h264(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from simulusion.utils.utils import Mp4Writer

    path = tmp_path / "clip.mp4"
    # Odd dimensions exercise the even-size padding yuv420p needs.
    with Mp4Writer(str(path), 10, (63, 47)) as writer:
        assert writer.codec == "h264"
        for _ in range(5):
            writer.write(np.random.randint(0, 255, (47, 63, 3), dtype=np.uint8))

    codec, frames = _video_codec(path)
    assert codec in ("h264", "avc1")
    assert frames == 5


def test_mp4_writer_accepts_grayscale_and_rejects_wrong_size(tmp_path):
    from simulusion.utils.utils import Mp4Writer

    with Mp4Writer(str(tmp_path / "gray.mp4"), 10, (32, 32)) as writer:
        writer.write(np.zeros((32, 32), dtype=np.uint8))
        with pytest.raises(ValueError):
            writer.write(np.zeros((16, 32, 3), dtype=np.uint8))


def test_streaming_video_writer_uses_h264(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from simulusion.utils.utils import StreamingVideoWriter

    path = tmp_path / "stream.mp4"
    writer = StreamingVideoWriter(str(path), fps=20)
    for _ in range(3):
        writer.write_frame(np.random.rand(40, 64, 3).astype(np.float32))
    writer.close()

    codec, _ = _video_codec(path)
    assert codec in ("h264", "avc1")


def test_metrics_logger_writes_jsonl(tmp_path):
    import json

    from simulusion.utils.logging_utils import MetricsLogger

    metrics = MetricsLogger(
        str(tmp_path),
        enable_console=False,
        enable_jsonl=True,
        enable_tensorboard=False,
        enable_wandb=False,
    )
    metrics.log({"loss": 1.25}, step=7)
    metrics.flush()
    metrics.close()

    payload = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert payload["step"] == 7
    assert payload["loss"] == 1.25


def test_assert_finite_decorator_rejects_nan():
    import pytest
    import torch

    from simulusion.utils.logging_utils import assert_finite

    @assert_finite
    def bad_loss():
        return torch.tensor(float("nan"))

    with pytest.raises(FloatingPointError):
        bad_loss()


def test_metrics_logger_writes_tensorboard_scalars_and_videos(tmp_path, monkeypatch):
    import torch

    import simulusion.utils.logging_utils as logging_utils

    events = []

    class FakeSummaryWriter:
        def __init__(self, log_dir):
            self.log_dir = log_dir

        def add_scalar(self, key, value, step):
            events.append(("scalar", key, value, step))

        def add_video(self, key, value, global_step=None, fps=20):
            events.append(("video", key, tuple(value.shape), global_step, fps))

        def flush(self):
            events.append(("flush",))

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(
        logging_utils, "_load_summary_writer", lambda: FakeSummaryWriter
    )

    metrics = logging_utils.MetricsLogger(
        str(tmp_path),
        enable_console=False,
        enable_jsonl=False,
        enable_tensorboard=True,
        enable_wandb=False,
    )
    metrics.log({"loss": 2.0, "nested": {"not": "scalar"}}, step=3)
    metrics.log_video("rollout", torch.zeros(4, 8, 8, 3), step=3, fps=12)
    metrics.close()

    assert ("scalar", "loss", 2.0, 3) in events
    assert ("video", "rollout", (1, 4, 3, 8, 8), 3, 12) in events
    assert ("flush",) in events
    assert ("close",) in events
