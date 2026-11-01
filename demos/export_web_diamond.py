#!/usr/bin/env python3
"""Export a DIAMOND checkpoint for playing inside the dream in a web browser.

Writes everything the site's in-browser player needs into one folder:

  dream_step.onnx   - one dream frame: the whole Euler sampler in one graph
  policy_step.onnx  - one actor-critic step, LSTM state in and out
  starts.bin        - real opening frames (uint8, [N, L, H, W, 3]) to seed dreams
  manifest.json     - shapes, action names and the checkpoint it came from

Both graphs are constant-folded, then checked against the eager model with ONNX
Runtime before the script finishes, so a file it writes is one that reproduces the checkpoint.

Usage:
    python demos/export_web_diamond.py -c demos/runs/diamond/continue_21/checkpoint_30.pt \
        --out ../TorchWMUI/public/play/diamond
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from simulusion.export import export_any
from simulusion.inference.play_diamond import make_agent
from simulusion.training.train_diamond import _normalize_frame


def collect_starts(
    agent, count: int, seed: int, warmup: int
) -> tuple[np.ndarray, np.ndarray]:
    """Real L-frame openings with the ball in play, and the actions between them.

    Each opening presses FIRE to launch the ball, then plays ``warmup`` random
    steps so the starts differ, and keeps the last L frames and the L actions
    taken at them.
    """
    n = agent.config.num_conditioning_frames
    env = agent.env
    rng = np.random.default_rng(seed)
    num_actions = env.action_space.n
    frames, actions = [], []
    for i in range(count):
        obs, _ = env.reset(seed=seed + i)
        history, taken = [obs], []
        for step in range(warmup + n):
            action = 1 if step == 0 else int(rng.integers(num_actions))
            obs, _, done, _ = env.step(action)
            taken.append(action)
            history.append(obs)
            if done:
                break
        # history[k] is the frame action taken[k] was chosen at.
        frames.append(np.stack(history[-n - 1 : -1]))
        actions.append(np.array(taken[-n:], dtype=np.int64))
    return np.stack(frames).astype(np.uint8), np.stack(actions)


def fold_constants(path: Path) -> None:
    """Rewrite ``path`` with ONNX Runtime's basic (provider-independent) passes.

    The TorchScript exporter leaves the shape arithmetic of every reshape and
    chunk in the graph; folding it removes about 60% of the nodes, and on
    WebGPU every node is a kernel dispatch.
    """
    import onnxruntime as ort

    folded = path.with_suffix(".folded.onnx")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(folded)
    ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    folded.replace(path)


def check_dream(agent, path: Path, starts: np.ndarray, actions: np.ndarray) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    obs = np.stack([_normalize_frame(f) for f in starts[0]]).transpose(0, 3, 1, 2)
    obs_t = torch.from_numpy(obs).unsqueeze(0)
    act_t = torch.from_numpy(actions[:1])
    cfg = agent.config
    shape = (1, 3, cfg.obs_size, cfg.obs_size)

    # Same noise for both: EulerSampler.sample draws randn(shape) first thing.
    torch.manual_seed(0)
    noise = torch.randn(shape)
    torch.manual_seed(0)
    eager = agent.sampler.sample(
        model=agent.diffusion_model,
        shape=shape,
        device=torch.device("cpu"),
        obs_history=obs_t,
        actions=act_t,
    )
    (onnx_out,) = session.run(
        None,
        {
            "obs_history": obs.astype(np.float32)[None],
            "actions": actions[:1],
            "noise": noise.numpy(),
        },
    )
    return float(np.abs(onnx_out - eager.numpy()).max())


def check_policy(agent, path: Path, starts: np.ndarray) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    frame = _normalize_frame(starts[0, -1]).transpose(2, 0, 1)[None]
    hidden = agent.actor_critic.init_hidden(1, torch.device("cpu"))
    with torch.no_grad():
        logits, _, _ = agent.actor_critic(torch.from_numpy(frame).unsqueeze(1), hidden)
    (onnx_logits, _, _) = session.run(
        None,
        {"frame": frame, "h": hidden[0].numpy(), "c": hidden[1].numpy()},
    )
    return float(np.abs(onnx_logits - logits[:, 0].numpy()).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("--game", default="Breakout-v5")
    parser.add_argument("--out", default="demos/out/web/diamond")
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    agent = make_agent(args.checkpoint, args.game, device="cpu", seed=args.seed)
    cfg = agent.config

    dream_path = export_any(
        agent,
        out / "dream_step.onnx",
        target="dream_step",
        input_names=["obs_history", "actions", "noise"],
        output_names=["frame"],
        opset_version=args.opset,
        dynamo=False,
    )
    policy_path = export_any(
        agent,
        out / "policy_step.onnx",
        target="policy_step",
        input_names=["frame", "h", "c"],
        output_names=["logits", "h_out", "c_out"],
        opset_version=args.opset,
        dynamo=False,
    )

    fold_constants(dream_path)
    fold_constants(policy_path)

    starts, start_actions = collect_starts(agent, args.starts, args.seed, args.warmup)
    (out / "starts.bin").write_bytes(starts.tobytes())

    dream_err = check_dream(agent, dream_path, starts, start_actions)
    policy_err = check_policy(agent, policy_path, starts)
    print(f"max |onnx - eager|: dream {dream_err:.2e}, policy {policy_err:.2e}")
    if dream_err > 1e-3 or policy_err > 1e-3:
        raise SystemExit("ONNX export does not match the eager model")

    meanings = agent.env.env.unwrapped.get_action_meanings()
    manifest = {
        "model": "diamond",
        "game": args.game,
        "checkpoint": Path(args.checkpoint).as_posix(),
        "obs_size": cfg.obs_size,
        "conditioning_frames": cfg.num_conditioning_frames,
        "sampling_steps": cfg.num_sampling_steps,
        "lstm_dim": cfg.actor_lstm_dim,
        "actions": list(meanings),
        "starts": {
            "file": "starts.bin",
            "shape": list(starts.shape),
            "actions": start_actions.tolist(),
        },
        "files": {"dream_step": "dream_step.onnx", "policy_step": "policy_step.onnx"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for f in sorted(out.iterdir()):
        print(f"{f.stat().st_size / 1e6:7.2f} MB  {f}")


if __name__ == "__main__":
    main()
