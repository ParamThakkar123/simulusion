"""Play Breakout with a trained IRIS agent, using the simulusion inference API.

The stepper threads the policy's LSTM state from frame to frame and resets it at
episode boundaries. Calling ``agent.act(frame)`` without ``hidden`` would make
the recurrent policy memoryless.

Usage:
    python examples/play_iris.py --checkpoint checkpoints/iris/best_Breakout-v5.pt
"""

import argparse

import cv2
import numpy as np
import torch

from simulusion.configs.iris_config import IRISConfig
from simulusion.envs.ale_atari_env import make_atari_env
from simulusion.inference import IRISStepper
from simulusion.models.iris_agent import IRISAgent


def preprocess_frame(frame, size=64):
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    frame = frame.astype(np.float32) / 255.0
    return frame.transpose(2, 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="IRISAgent checkpoint")
    parser.add_argument("--game", default="ALE/Breakout-v5")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = IRISConfig()
    env = make_atari_env(args.game, obs_type="rgb", frameskip=4, render_mode="human")
    action_size = env.action_space.n

    agent = IRISAgent(config=config, action_size=action_size, device=device)
    agent.load(args.checkpoint)
    stepper = IRISStepper(agent, temperature=0.5)
    print("Model loaded!")

    num_episodes = args.episodes
    total_reward = 0

    with torch.inference_mode():
        for ep in range(num_episodes):
            obs, _ = env.reset()
            state = stepper.init_state(batch_size=1)
            episode_reward = 0
            steps = 0

            print(f"\n--- Episode {ep + 1} ---")

            while True:
                env.render()

                frame = torch.from_numpy(preprocess_frame(obs)).unsqueeze(0)
                state = stepper.observe(state, frame, None)
                action = stepper.act(state).item()

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            total_reward += episode_reward
            print(f"Episode {ep + 1}: Reward = {episode_reward}, Steps = {steps}")

    env.close()
    print(
        f"\nAverage reward over {num_episodes} episodes: {total_reward / num_episodes:.2f}"
    )


if __name__ == "__main__":
    main()
