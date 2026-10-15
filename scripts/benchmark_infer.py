"""Inference / play front-end for ``scripts/benchmark_models.sh``.

The compute sweep in ``benchmark_models.py`` times synthetic forwards. This
module is the other half: run a **trained** (or randomly initialised) model
and either

* **record** a video of it playing / generating, or
* **play** interactively -- with the policy (keys override it) or against it
  (you drive, the policy's action is shown as the opponent).

Wired from the shell as::

    scripts/benchmark_models.sh --infer --model diamond -c ckpt.pt
    scripts/benchmark_models.sh --play  --model dreamer -c ckpt.pt --versus
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_GAMES = {
    "diamond": "Breakout-v5",
    "dreamer": "walker-walk",
    "iris": "ALE/Pong-v5",
}

RECORD_MODELS = ("diamond", "dreamer", "iris", "genie", "dit", "jepa")
PLAY_MODELS = ("diamond", "dreamer", "iris", "genie")


def _load_demo(name: str) -> Any:
    """Load a file under ``demos/`` -- that directory is not a Python package."""
    path = REPO_ROOT / "demos" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"simulusion_demo_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _python() -> str:
    return sys.executable


def _run(argv: list[str]) -> int:
    print("running:", " ".join(str(part) for part in argv))
    # -u after the interpreter: these demos are launched with their stdout on a
    # pipe (the sweep tees every stage to a log), where Python block-buffers by
    # default and a long sampling loop looks like a hang until it exits.
    if argv and argv[0] == _python():
        argv = [argv[0], "-u", *argv[1:]]
    return subprocess.call(argv, cwd=str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("record", "play"),
        default="record",
        help="record: write a video and exit. play: open an interactive window.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="",
        help="diamond, dreamer, iris, genie, dit or jepa.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Alias of --model (the compute sweep uses this flag name).",
    )
    parser.add_argument("--checkpoint", "-c", default=None)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Skip the checkpoint (Genie / DiT / I-JEPA pipeline check only).",
    )
    parser.add_argument("--game", "-g", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "results" / "model_inference"),
        help="Where recorded videos and stills land.",
    )
    parser.add_argument("--steps", type=int, default=300, help="[record] env frames.")
    parser.add_argument(
        "--dream-steps",
        type=int,
        default=100,
        help="[diamond/dreamer record] imagination frames. 0 skips the dream clip.",
    )
    parser.add_argument(
        "--dream-context",
        type=int,
        default=5,
        help="[dreamer record] real frames the RSSM sees before the open-loop "
        "rollout begins.",
    )
    parser.add_argument("--episodes", type=int, default=2, help="[iris record]")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--record", default=None, help="[play] also save an MP4.")
    parser.add_argument(
        "--control",
        choices=("assist", "human", "versus"),
        default="assist",
        help="assist: play WITH the model (keys override). "
        "human: you always drive. "
        "versus: you drive, the model's action is shown as the opponent.",
    )
    parser.add_argument(
        "--versus",
        action="store_true",
        help="Shortcut for --control versus (play against the policy).",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--list", action="store_true", help="List models and exit.")
    return parser


def resolve_model(args: argparse.Namespace) -> str:
    name = (args.model or args.models or "").strip().lower()
    if "," in name:
        name = name.split(",", 1)[0].strip()
    aliases = {
        "dreamer-v1": "dreamer",
        "dreamerv1": "dreamer",
        "dreamer-v2": "dreamer",
        "dreamerv2": "dreamer",
        "ijepa": "jepa",
        "i-jepa": "jepa",
        "genie-small": "genie",
    }
    return aliases.get(name, name)


def list_models() -> None:
    print("record: " + ", ".join(RECORD_MODELS))
    print("play:   " + ", ".join(PLAY_MODELS))
    print()
    print("record = video of the model playing / generating")
    print("play   = interactive window")
    print("         --control assist  (default): play WITH the policy")
    print("         --control versus / --versus: you drive, policy is the opponent HUD")


def record_diamond(args: argparse.Namespace, model: str) -> int:
    script = REPO_ROOT / "demos" / "record_diamond.py"
    argv = [
        _python(),
        str(script),
        "--checkpoint",
        str(args.checkpoint),
        "--game",
        args.game or DEFAULT_GAMES["diamond"],
        "--out-dir",
        args.out_dir,
        "--steps",
        str(args.steps),
        "--dream-steps",
        str(args.dream_steps),
        "--fps",
        str(args.fps),
        "--seed",
        str(args.seed),
    ]
    if args.device:
        argv += ["--device", args.device]
    if args.stochastic:
        argv.append("--stochastic")
    return _run(argv)


def record_iris(args: argparse.Namespace, model: str) -> int:
    script = REPO_ROOT / "demos" / "record_iris.py"
    out = Path(args.out_dir) / "iris.mp4"
    argv = [
        _python(),
        str(script),
        "--checkpoint",
        str(args.checkpoint),
        "--game",
        args.game or DEFAULT_GAMES["iris"],
        "--episodes",
        str(args.episodes),
        "--out",
        str(out),
        "--fps",
        str(args.fps),
        "--seed",
        str(args.seed),
        # --steps means "frames to record" everywhere else in this front-end;
        # for IRIS it is the per-episode cap, so an episode that never ends
        # cannot run the sweep past its budget.
        "--max-steps",
        str(args.steps),
    ]
    if args.device:
        argv += ["--device", args.device]
    return _run(argv)


def record_dit(args: argparse.Namespace, model: str) -> int:
    script = REPO_ROOT / "demos" / "record_dit.py"
    argv = [_python(), str(script), "--out-dir", args.out_dir, "--seed", str(args.seed)]
    if args.random_init or not args.checkpoint:
        argv.append("--random-init")
    else:
        argv += ["--checkpoint", str(args.checkpoint)]
    if args.device:
        argv += ["--device", args.device]
    return _run(argv)


def record_genie(args: argparse.Namespace, model: str) -> int:
    script = REPO_ROOT / "demos" / "record_genie.py"
    argv = [
        _python(),
        str(script),
        "--out-dir",
        args.out_dir,
        "--num-frames",
        str(min(args.steps, 32)),
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
    ]
    if args.random_init or not args.checkpoint:
        argv.append("--random-init")
    else:
        argv += ["--checkpoint", str(args.checkpoint)]
    if args.device:
        argv += ["--device", args.device]
    return _run(argv)


def record_jepa(args: argparse.Namespace, model: str) -> int:
    script = REPO_ROOT / "demos" / "record_jepa.py"
    argv = [_python(), str(script), "--out-dir", args.out_dir, "--seed", str(args.seed)]
    if args.random_init or not args.checkpoint:
        argv.append("--random-init")
    else:
        argv += ["--checkpoint", str(args.checkpoint)]
    if args.device:
        argv += ["--device", args.device]
    return _run(argv)


def _label_frame(frame: "Any", text: str, scale: int = 4) -> "Any":
    """Upscale a small world-model frame and caption it, for legibility."""
    import cv2
    import numpy as np

    img = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    img = cv2.resize(
        img,
        (img.shape[1] * scale, img.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


def record_dreamer_dream(
    args: argparse.Namespace, player: "Any", out_dir: Path
) -> list:
    """Open-loop imagination beside the real env, the Dreamer paper's figure.

    The model sees ``--dream-context`` real frames, then its observations are cut
    off and the RSSM rolls forward on its own predictions while the real
    environment keeps running on the *same* actions. The two clips therefore
    diverge exactly as much as the world model is wrong, which is the thing
    worth looking at -- a real-only recording shows the policy, not the model.
    """
    import numpy as np
    import torch

    from simulusion.inference.play_dreamer import _observation_frame
    from simulusion.utils.utils import StreamingVideoWriter

    context = max(1, args.dream_context)
    obs = player.env.reset()
    state = player.rssm.init_state(1, player.device)
    prev_action = torch.zeros(1, player.action_size, device=player.device)

    real_frames: list = []
    dream_frames: list = []

    for step in range(context + args.dream_steps):
        grounded = step < context
        with torch.no_grad():
            if grounded:
                # Closed loop: the posterior still sees the real observation.
                state, _ = player.rssm.observe_step(
                    state, prev_action, player.encode(obs)
                )
            else:
                # Open loop: prior only, from here the model is on its own.
                state = player.rssm.imagine_step(state, prev_action)
            action = player.actor(player.features(state), deter=not args.stochastic)
            decoded = player.decode(state)

        tag = "DREAM (context)" if grounded else f"DREAM (+{step - context + 1})"
        real_frames.append(_label_frame(_observation_frame(obs), "REAL"))
        dream_frames.append(_label_frame(decoded, tag))

        action_np = action[0].cpu().numpy()
        obs, _, done, info = player.env.step(action_np)
        executed = (
            info["action"] if isinstance(info, dict) and "action" in info else action_np
        )
        prev_action = torch.tensor(
            np.asarray(executed, dtype=np.float32), device=player.device
        ).unsqueeze(0)
        if done:
            break

    written = []
    dream_path = out_dir / "dreamer_dream.mp4"
    writer = StreamingVideoWriter(str(dream_path), fps=args.fps)
    for frame in dream_frames:
        writer.write_frame(frame)
    writer.close()
    written.append(dream_path)

    side_path = out_dir / "dreamer_side_by_side.mp4"
    writer = StreamingVideoWriter(str(side_path), fps=args.fps)
    for real, dream in zip(real_frames, dream_frames):
        writer.write_frame(np.concatenate([real, dream], axis=1))
    writer.close()
    written.append(side_path)

    print(
        f"Wrote {dream_path} and {side_path} "
        f"({context} context + {len(dream_frames) - context} imagined frames)"
    )
    return written


def record_dreamer(args: argparse.Namespace, model: str) -> int:
    """Headless Dreamer rollout: policy in the real env, no OpenCV window."""
    import numpy as np
    import torch

    from simulusion.inference.play_dreamer import DreamerPlayer, _observation_frame
    from simulusion.utils.utils import StreamingVideoWriter

    game = args.game or DEFAULT_GAMES["dreamer"]
    player = DreamerPlayer(
        str(args.checkpoint), env_name=game, device=args.device, seed=args.seed
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dreamer_real.mp4"
    writer = StreamingVideoWriter(str(path), fps=args.fps)

    obs = player.env.reset()
    state = player.rssm.init_state(1, player.device)
    prev_action = torch.zeros(1, player.action_size, device=player.device)
    reward_sum = 0.0
    for step in range(args.steps):
        # Same upscale and caption as the dream clip, so the two sit together
        # in a gallery instead of one being a 64px thumbnail.
        writer.write_frame(_label_frame(_observation_frame(obs), "REAL"))
        with torch.no_grad():
            state, _ = player.rssm.observe_step(state, prev_action, player.encode(obs))
            action = player.actor(player.features(state), deter=not args.stochastic)
        action_np = action[0].cpu().numpy()
        obs, reward, done, info = player.env.step(action_np)
        executed = (
            info["action"] if isinstance(info, dict) and "action" in info else action_np
        )
        prev_action = torch.tensor(
            np.asarray(executed, dtype=np.float32), device=player.device
        ).unsqueeze(0)
        reward_sum += float(reward)
        if done:
            obs = player.env.reset()
            state = player.rssm.init_state(1, player.device)
            prev_action = torch.zeros(1, player.action_size, device=player.device)
        if (step + 1) % 50 == 0:
            print(f"  real: {step + 1}/{args.steps}")

    writer.close()
    print(f"Wrote {path}  (return across clip: {reward_sum:.1f})")

    if args.dream_steps > 0:
        record_dreamer_dream(args, player, out_dir)

    player.env.close()
    return 0


def play_diamond(args: argparse.Namespace, control: str) -> int:
    from simulusion.inference.play_diamond import run_play

    run_play(
        checkpoint=str(args.checkpoint),
        game=args.game or DEFAULT_GAMES["diamond"],
        device=args.device,
        seed=args.seed,
        deterministic=not args.stochastic,
        record=args.record,
        record_fps=args.fps,
        control=control,
    )
    return 0


def play_dreamer(args: argparse.Namespace, control: str) -> int:
    from simulusion.inference.play_dreamer import run_play

    run_play(
        checkpoint=str(args.checkpoint),
        game=args.game or DEFAULT_GAMES["dreamer"],
        device=args.device,
        seed=args.seed,
        deterministic=not args.stochastic,
        record=args.record,
        record_fps=args.fps,
        control=control,
    )
    return 0


def play_iris(args: argparse.Namespace, control: str) -> int:
    """Atari window: assist / human / versus, using the policy-only loader."""
    import cv2
    import numpy as np
    import torch

    iris_demo = _load_demo("record_iris")
    infer_architecture = iris_demo.infer_architecture
    load_policy = iris_demo.load_policy
    preprocess = iris_demo.preprocess
    read_checkpoint = iris_demo.read_checkpoint
    from simulusion.inference.play_base import get_action_from_key, init_video_recorder
    from simulusion.inference.play_diamond import ACTION_NAMES
    from simulusion.configs.iris_config import IRISConfig
    from simulusion.envs.ale_atari_env import make_atari_env
    from simulusion.models.iris_agent import IRISAgent

    game = args.game or DEFAULT_GAMES["iris"]
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    env = make_atari_env(game, obs_type="rgb", frameskip=4)
    n_actions = int(env.action_space.n)
    ckpt = read_checkpoint(str(args.checkpoint))
    arch = infer_architecture(ckpt)
    config = IRISConfig()
    for field in ("actor_layers", "actor_hidden_size"):
        if field in arch:
            setattr(config, field, arch[field])
    agent = IRISAgent(config=config, action_size=n_actions, device=device)
    report = load_policy(agent, ckpt)
    if report["failed"] or not report["loaded"]:
        print("Policy did not load cleanly; refusing to open play.")
        return 1
    agent.eval()

    video_recorder = init_video_recorder(args.record, fps=args.fps)
    cv2.namedWindow("IRIS Play", cv2.WINDOW_NORMAL)
    obs, _ = env.reset()
    episode_reward = 0.0
    step_count = 0
    # The IRIS policy is recurrent: carry its LSTM state across steps and reset
    # it with the episode, as the trainer does. Calling act() without it made
    # the policy memoryless in play.
    hidden = None
    running = True
    print("IRIS play: arrows/WASD drive, Q quits. --versus shows the policy's action.")

    while running:
        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), 27):
            running = False
            continue
        if key == ord("r"):
            obs, _ = env.reset()
            episode_reward = 0.0
            step_count = 0
            hidden = None

        frame = preprocess(obs, config.frame_height)
        tensor = torch.from_numpy(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            action_tensor, hidden = agent.act(
                tensor,
                epsilon=0.0,
                temperature=0.01,
                hidden=hidden,
                return_hidden=True,
            )
        agent_action = int(action_tensor.item())
        human_action = get_action_from_key(key)
        if control == "assist":
            action = agent_action if human_action is None else human_action
            label = "AGENT" if human_action is None else "HUMAN"
        else:
            action = 0 if human_action is None else human_action
            label = "HUMAN vs AGENT" if control == "versus" else "HUMAN"

        display = obs.astype("uint8")
        if display.ndim == 3 and display.shape[2] == 3:
            display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        else:
            display_bgr = display
        hud = [
            f"{label}  R: {episode_reward:.1f}  Step: {step_count}",
            f"Action: {ACTION_NAMES.get(action, action)}"
            + (
                f"  |  AGENT: {ACTION_NAMES.get(agent_action, agent_action)}"
                if control == "versus"
                else ""
            ),
            "[arrows/WASD] drive  [R] reset  [Q] quit",
        ]
        for i, line in enumerate(hud):
            cv2.putText(
                display_bgr,
                line,
                (5, 15 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
        cv2.imshow("IRIS Play", display_bgr)
        if video_recorder is not None:
            video_recorder.write_frame(obs.astype(np.uint8))

        obs, reward, terminated, truncated, _ = env.step(int(action))
        episode_reward += float(reward)
        step_count += 1
        if terminated or truncated:
            print(f"Episode finished. Reward: {episode_reward:.1f}")
            obs, _ = env.reset()
            episode_reward = 0.0
            step_count = 0
            hidden = None

    if video_recorder is not None:
        video_recorder.close()
    cv2.destroyAllWindows()
    env.close()
    return 0


def play_genie(args: argparse.Namespace, control: str) -> int:
    """Drive Genie with latent-action keys. There is no 'versus' env; 0-7 are you."""
    import cv2
    import numpy as np
    import torch

    genie_demo = _load_demo("record_genie")
    build_model = genie_demo.build_model
    tensor_to_uint8_img = genie_demo.tensor_to_uint8_img
    from simulusion.inference.play_base import init_video_recorder

    class _Args:
        checkpoint = args.checkpoint
        random_init = bool(args.random_init or not args.checkpoint)
        num_frames = 8
        image_size = 64

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_model(_Args()).to(device).eval()
    vocab = int(getattr(model, "action_vocab_size", 8) or 8)
    frame = torch.rand(1, 3, 64, 64, device=device)
    history: torch.Tensor | None = None
    action_id = 0
    video_recorder = init_video_recorder(args.record, fps=args.fps)
    cv2.namedWindow("Genie Play", cv2.WINDOW_NORMAL)
    print(
        "Genie play: keys 0-7 / WASD select a latent action, SPACE steps, "
        "R new prompt, Q quits."
    )

    key_to_action = {
        ord("w"): 0,
        ord("s"): 1,
        ord("a"): 2,
        ord("d"): 3,
        ord(" "): None,
    }
    running = True
    while running:
        img = tensor_to_uint8_img(frame[0])
        display = cv2.cvtColor(np.tile(img, (4, 4, 1)), cv2.COLOR_RGB2BGR)
        cv2.putText(
            display,
            f"latent action {action_id}  [0-7/WASD] choose  [SPACE] step  [R] reset  [Q] quit",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )
        cv2.imshow("Genie Play", display)
        if video_recorder is not None:
            video_recorder.write_frame(np.tile(img, (4, 4, 1)))

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            running = False
            continue
        if key == ord("r"):
            frame = torch.rand(1, 3, 64, 64, device=device)
            history = None
            continue
        if ord("0") <= key <= ord("9"):
            action_id = (key - ord("0")) % vocab
        elif key in key_to_action and key_to_action[key] is not None:
            action_id = int(key_to_action[key]) % vocab
        elif key != ord(" "):
            continue

        with torch.no_grad():
            action = torch.tensor([action_id], device=device)
            nxt = model.play(frame, action, history)
            if history is None:
                history = frame.unsqueeze(2)
            history = torch.cat([history, nxt.unsqueeze(2)], dim=2)
            if history.shape[2] > 8:
                history = history[:, :, -8:]
            frame = nxt

    if video_recorder is not None:
        video_recorder.close()
    cv2.destroyAllWindows()
    return 0


RECORDERS = {
    "diamond": record_diamond,
    "dreamer": record_dreamer,
    "iris": record_iris,
    "genie": record_genie,
    "dit": record_dit,
    "jepa": record_jepa,
}

PLAYERS = {
    "diamond": play_diamond,
    "dreamer": play_dreamer,
    "iris": play_iris,
    "genie": play_genie,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("note: ignoring flags not used for inference:", " ".join(unknown))

    if args.list:
        list_models()
        return 0

    model = resolve_model(args)
    if not model:
        parser.error("pass --model diamond|dreamer|iris|genie|dit|jepa (or --list)")

    control = "versus" if args.versus else args.control
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "record":
        if model not in RECORDERS:
            parser.error(
                f"{model} cannot be recorded. Choose: {', '.join(RECORD_MODELS)}"
            )
        if model in {"diamond", "dreamer", "iris"} and not args.checkpoint:
            parser.error(f"{model} record needs --checkpoint / -c")
        print(f"recording {model} -> {args.out_dir}")
        return RECORDERS[model](args, model)

    if model not in PLAYERS:
        parser.error(
            f"{model} has no interactive play loop. "
            f"Use --infer to record a video. Playable: {', '.join(PLAY_MODELS)}"
        )
    if model != "genie" and not args.checkpoint:
        parser.error(f"{model} play needs --checkpoint / -c")
    print(f"playing {model}  control={control}")
    return PLAYERS[model](args, control)


if __name__ == "__main__":
    raise SystemExit(main())
