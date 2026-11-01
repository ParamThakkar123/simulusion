"""Saving the dream trajectory on demand in ``simulusion play`` (issue #187)."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("cv2", reason="play_diamond imports OpenCV")

import cv2
from simulusion.inference import play_diamond
from simulusion.inference.play_base import DreamBuffer

SIZE = 16
L = 4
TAB, NONE, DOWN = ord("\t"), -1, 5
NAME = re.compile(r"dream_\d{8}-\d{6}-\d{3}")
DREAM_VALUE = 0.5


def frame(value: int) -> np.ndarray:
    return np.full((SIZE, SIZE, 3), value, dtype=np.uint8)


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    return count


def test_buffer_keeps_only_the_last_n_steps(tmp_path):
    buffer = DreamBuffer(3)
    for step in range(1, 6):
        buffer.add(frame(step * 40), action=step, reward=float(step % 3 - 1))

    video, data = buffer.save(tmp_path, fps=10, game="Breakout-v5")

    saved = json.loads(data.read_text())
    assert saved["actions"] == [3, 4, 5]
    assert saved["rewards"] == [-1.0, 0.0, 1.0]
    assert saved["game"] == "Breakout-v5"
    assert saved["fps"] == 10
    assert saved["video"] == video.name
    assert video_frame_count(video) == 3


def test_save_names_both_files_after_one_timestamp(tmp_path):
    buffer = DreamBuffer(5)
    buffer.add(frame(0), action=1, reward=0.0)

    video, data = buffer.save(tmp_path, fps=20)

    assert video.parent == data.parent == tmp_path
    assert NAME.fullmatch(video.stem)
    assert (video.suffix, data.suffix) == (".mp4", ".json")
    assert video.stem == data.stem
    assert video.stat().st_size > 0


def test_saving_an_empty_buffer_writes_nothing(tmp_path):
    with pytest.raises(ValueError):
        DreamBuffer(3).save(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_clear_empties_the_buffer():
    buffer = DreamBuffer(5)
    buffer.add(frame(0), action=1, reward=0.0)
    buffer.clear()
    assert len(buffer) == 0


class Sampler:
    """Imagines a uniform frame and records each conditioning window."""

    def __init__(self):
        self.actions: list[list[int]] = []

    def sample(self, model, shape, device, obs_history, actions):
        self.actions.append(actions[0].tolist())
        return torch.full(shape, DREAM_VALUE)


class ActorCritic:
    """Takes actions 1, 2, 3, ... so each step's action is identifiable."""

    def __init__(self):
        self.t = 0

    def init_hidden(self, batch, device):
        return None

    def get_action(self, obs, hidden, deterministic=True):
        self.t += 1
        return self.t, hidden


class RewardModel:
    """Predicts ``action % 3 - 1`` and records what it was shown."""

    def __init__(self):
        self.calls: list[dict] = []

    def init_hidden(self, batch, device):
        return "fresh"

    def predict(self, obs, actions, hidden_state=None):
        self.calls.append(
            {"obs": obs.clone(), "action": int(actions[0]), "hidden": hidden_state}
        )
        reward = torch.tensor([float(int(actions[0]) % 3 - 1)])
        return reward, torch.tensor([False]), "used"


class Env:
    def __init__(self):
        self.actions: list[int] = []

    def reset(self):
        return np.zeros((SIZE, SIZE, 3), dtype=np.uint8), {}

    def step(self, action):
        self.actions.append(int(action))
        return np.zeros((SIZE, SIZE, 3), dtype=np.uint8), 0.0, False, {}

    def close(self):
        pass


def fake_agent() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(num_conditioning_frames=L, obs_size=SIZE),
        device=torch.device("cpu"),
        sampler=Sampler(),
        diffusion_model=None,
        actor_critic=ActorCritic(),
        reward_model=RewardModel(),
        env=Env(),
    )


@pytest.fixture
def play(monkeypatch, tmp_path):
    """Run ``run_play`` headlessly in ``tmp_path`` with scripted key presses."""
    monkeypatch.chdir(tmp_path)
    texts: list[str] = []
    real_put_text = cv2.putText

    def put_text(img, text, *args, **kwargs):
        texts.append(text)
        return real_put_text(img, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", put_text)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **k: None)

    def run(keys, **kwargs):
        agent = fake_agent()
        pending = iter(keys)
        monkeypatch.setattr(cv2, "waitKey", lambda delay: next(pending))
        monkeypatch.setattr(play_diamond, "make_agent", lambda *a, **k: agent)
        play_diamond.run_play("ckpt.pt", **kwargs)
        return agent, texts

    return run


def saved_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("dream_*"))


def test_s_in_dream_saves_the_last_n_steps(play, tmp_path):
    keys = [TAB, NONE, NONE, NONE, NONE, ord("s"), ord("q")]
    agent, texts = play(keys, dream_buffer=3)

    data, video = saved_files(tmp_path)
    assert (video.suffix, data.suffix) == (".mp4", ".json")
    saved = json.loads(data.read_text())
    assert saved["actions"] == [3, 4, 5]
    assert saved["rewards"] == [-1.0, 0.0, 1.0]
    assert saved["game"] == "Breakout-v5"
    assert video_frame_count(video) == 3
    # Dreamed frames, not the reset frame or the HUD: 0.5 in [-1, 1] is 191.
    first = cv2.VideoCapture(str(video)).read()[1]
    assert abs(int(np.median(first)) - 191) <= 3
    assert "Saved" in texts


def test_s_in_dream_does_not_also_steer(play):
    agent, _ = play([TAB, ord("s"), ord("q")])
    # The step after S is the policy's 2nd action, not DOWN.
    assert agent.sampler.actions[-1][-1] == 2


def test_reward_is_predicted_on_the_frame_the_action_was_taken_at(play):
    agent, _ = play([TAB, NONE, ord("q")])

    first, second = agent.reward_model.calls
    # Black reset frame is -1 in the model domain; the next is the dream.
    assert first["obs"].max().item() == -1.0
    assert second["obs"].min().item() == DREAM_VALUE
    assert (first["action"], second["action"]) == (1, 2)
    assert (first["hidden"], second["hidden"]) == ("fresh", "used")


def test_reset_starts_a_new_dream(play, tmp_path):
    agent, _ = play([TAB, NONE, ord("r"), NONE, ord("s"), ord("q")])

    data = saved_files(tmp_path)[0]
    assert json.loads(data.read_text())["actions"] == [3, 4]
    assert agent.reward_model.calls[2]["hidden"] == "fresh"


def test_entering_dream_again_starts_a_new_dream(play, tmp_path):
    play([TAB, NONE, TAB, TAB, NONE, ord("s"), ord("q")])

    data = saved_files(tmp_path)[0]
    assert json.loads(data.read_text())["actions"] == [4, 5]


def test_s_in_real_mode_steers_and_saves_nothing(play, tmp_path):
    agent, _ = play([ord("s"), ord("q")])

    assert agent.env.actions == [DOWN]
    assert saved_files(tmp_path) == []


def test_module_cli_passes_dream_buffer(monkeypatch):
    calls = {}
    monkeypatch.setattr(play_diamond, "run_play", lambda **kw: calls.update(kw))
    monkeypatch.setattr("sys.argv", ["play_diamond", "-c", "x.pt"])
    play_diamond.main()
    assert calls["dream_buffer"] == 50

    monkeypatch.setattr(
        "sys.argv", ["play_diamond", "-c", "x.pt", "--dream-buffer", "7"]
    )
    play_diamond.main()
    assert calls["dream_buffer"] == 7


def test_module_cli_rejects_a_negative_dream_buffer(monkeypatch, capsys):
    monkeypatch.setattr(play_diamond, "run_play", lambda **kw: None)
    monkeypatch.setattr(
        "sys.argv", ["play_diamond", "-c", "x.pt", "--dream-buffer", "-1"]
    )
    with pytest.raises(SystemExit):
        play_diamond.main()
    assert "--dream-buffer: must be >= 0" in capsys.readouterr().err


def test_zero_buffer_turns_saving_off_and_s_steers_again(play, tmp_path):
    agent, texts = play([TAB, NONE, ord("s"), ord("q")], dream_buffer=0)

    assert saved_files(tmp_path) == []
    assert agent.reward_model.calls == []
    assert agent.sampler.actions[-1][-1] == DOWN
    assert not any("[S] save" in text for text in texts)


def test_a_failed_save_keeps_playing(play, monkeypatch, tmp_path):
    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(DreamBuffer, "save", disk_full)
    agent, texts = play([TAB, ord("s"), NONE, ord("q")])

    assert "Save failed" in texts
    assert len(agent.sampler.actions) == 3
