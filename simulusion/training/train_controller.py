"""Training a linear controller on latent + recurrent state with CMA-ES.

This module provides functions to train a linear controller using Covariance
Matrix Adaptation Evolution Strategy (CMA-ES). The controller maps latent
and hidden states to actions for the learned world model.

Reference:
    Ha & Schmidhuber (2018). Recurrent World Models Facilitate Policy Evolution.
    https://arxiv.org/abs/1805.11111
"""

from typing import Any
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep
from torch.multiprocessing import Process, Queue
import torch
import torch.nn.functional as F
import cma
import numpy as np
from tqdm import tqdm
import gymnasium as gym

from simulusion.models.controller import Controller
from simulusion.models.mdrnn import MDRNN, MDRNNCell
from simulusion.vision.VAE.ConvVAE import ConvVAE
from simulusion.configs.wm_config import WMControllerConfig
from simulusion.envs.gym_env import GymImageEnv


def flatten_parameters(parameters: Any) -> np.ndarray:
    return np.concatenate([p.data.cpu().numpy().flatten() for p in parameters])


def load_parameters(params: Any, controller: Any) -> None:
    pointer = 0
    for param in controller.parameters():
        param_shape = param.shape
        size = 1
        for s in param_shape:
            size *= s
        param.data.copy_(
            torch.from_numpy(params[pointer : pointer + size]).view(param_shape)
        )
        pointer += size


def _run_rollout(
    ctrl_params: np.ndarray,
    logdir: str,
    env_name: str,
    action_size: int,
    time_limit: int,
    device: torch.device,
) -> float:
    """Run a single rollout with given controller parameters.

    Args:
        ctrl_params: Flattened controller parameters as numpy array.
        logdir: Directory containing trained VAE and MDRNN checkpoints.
        env_name: Gym environment name.
        action_size: Dimensionality of action space.
        time_limit: Maximum steps per episode.
        device: torch device.

    Returns:
        float: Total cumulative reward.
    """
    vae_file = join(logdir, "vae", "best.tar")
    rnn_file = join(logdir, "mdrnn", "best.tar")

    vae_state = torch.load(vae_file, map_location=device, weights_only=True)
    rnn_state = torch.load(rnn_file, map_location=device, weights_only=True)
    latent_size = 32

    vae = ConvVAE(img_channels=3, latent_size=latent_size).to(device)
    vae.load_state_dict(vae_state["state_dict"])
    vae.eval()

    batch_rnn = MDRNN(
        latents=latent_size, actions=action_size, hiddens=256, gaussians=5
    )
    batch_rnn.load_state_dict(rnn_state["state_dict"])
    batch_rnn.eval()

    cell_rnn = MDRNNCell(
        latents=latent_size, actions=action_size, hiddens=256, gaussians=5
    ).to(device)
    cell_rnn.rnn.weight_ih.data.copy_(batch_rnn.rnn.weight_ih_l0.data)  # type: ignore[arg-type]
    cell_rnn.rnn.weight_hh.data.copy_(batch_rnn.rnn.weight_hh_l0.data)  # type: ignore[arg-type]
    cell_rnn.rnn.bias_ih.data.copy_(batch_rnn.rnn.bias_ih_l0.data)  # type: ignore[arg-type]
    cell_rnn.rnn.bias_hh.data.copy_(batch_rnn.rnn.bias_hh_l0.data)  # type: ignore[arg-type]
    cell_rnn.gmm_linear.load_state_dict(batch_rnn.gmm_linear.state_dict())
    cell_rnn.eval()
    del batch_rnn

    ctrl = Controller(latent_size, 256, action_size).to(device)
    load_parameters(ctrl_params, ctrl)
    ctrl.eval()

    try:
        env: Any = gym.make(env_name, continuous=True)
    except Exception:
        env = gym.make(env_name)
    env = GymImageEnv(env=env, size=(64, 64))

    obs, _ = env.reset()
    h, c = cell_rnn.get_init_hidden(1)
    h, c = h.to(device), c.to(device)

    total_reward = 0.0
    with torch.no_grad():
        for _ in range(time_limit):
            obs_tensor = torch.tensor(obs["image"]).float().unsqueeze(0).to(device)
            obs_tensor = F.interpolate(
                obs_tensor, size=64, mode="bilinear", align_corners=True
            )
            obs_tensor = obs_tensor / 255.0

            mu, logsigma = vae.encoder(obs_tensor)
            z = mu + logsigma.exp() * torch.randn_like(logsigma)

            action = ctrl(h, z).cpu().numpy().flatten()

            action_t = torch.tensor(action).float().to(device)
            _, _, _, _, _, (h, c) = cell_rnn(action_t, z, (h, c))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            obs = next_obs
            if terminated or truncated:
                break

    env.close()
    return total_reward


def slave_routine(
    p_queue: Any, r_queue: Any, e_queue: Any, p_index: int, config: Any, time_limit: int
) -> None:
    """Worker process routine for parallel rollout evaluation.

    Args:
        p_queue: Queue containing (s_id, parameters) to evaluate.
        r_queue: Queue where to place results (s_id, reward).
        e_queue: End queue - when non-empty, process terminates.
        p_index: Process index for GPU assignment.
        config: Controller configuration (must include env_name and action_size).
        time_limit: Maximum steps per episode.
    """
    gpu = p_index % max(torch.cuda.device_count(), 1)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    sys.stdout = open(join(config.logdir, "tmp", str(getpid()) + ".out"), "a")
    sys.stderr = open(join(config.logdir, "tmp", str(getpid()) + ".err"), "a")

    while e_queue.empty():
        if p_queue.empty():
            sleep(0.1)
        else:
            s_id, params = p_queue.get()
            reward = _run_rollout(
                params,
                config.logdir,
                config.env_name,
                config.action_size,
                time_limit,
                device,
            )
            r_queue.put((s_id, reward))


def evaluate(
    solutions: Any, results: Any, rollouts: int, p_queue: Any, r_queue: Any
) -> Any:
    """Evaluate current controller."""
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    for s_id in range(rollouts):
        p_queue.put((s_id, best_guess))

    print("Evaluating...")
    for _ in tqdm(range(rollouts)):
        while r_queue.empty():
            sleep(0.1)
        restimates.append(r_queue.get()[1])

    return best_guess, np.mean(restimates), np.std(restimates)


def train_controller(config: WMControllerConfig) -> None:
    """Train a linear controller using CMA-ES.

    Args:
        config: WMControllerConfig containing training hyperparameters,
                including env_name and action_size.

    The training process includes:
        - Setting up parallel evaluation workers (each loads VAE + MDRNN)
        - Running CMA-ES optimization with parallel rollout evaluation
        - Evaluating and saving best controller checkpoint
    """
    n_samples = config.n_samples
    pop_size = config.pop_size
    num_workers = min(config.max_workers, n_samples * pop_size)
    time_limit = config.time_limit

    tmp_dir = join(config.logdir, "tmp")
    if not exists(tmp_dir):
        mkdir(tmp_dir)
    else:
        for fname in listdir(tmp_dir):
            unlink(join(tmp_dir, fname))

    ctrl_dir = join(config.logdir, "ctrl")
    if not exists(ctrl_dir):
        mkdir(ctrl_dir)

    p_queue: Queue = Queue()
    r_queue: Queue = Queue()
    e_queue: Queue = Queue()

    for p_index in range(num_workers):
        Process(
            target=slave_routine,
            args=(p_queue, r_queue, e_queue, p_index, config, time_limit),
        ).start()

    controller = Controller(config.latent_size, config.hidden_size, config.action_size)

    cur_best = None
    ctrl_file = join(ctrl_dir, "best.tar")
    print("Attempting to load previous best...")
    if exists(ctrl_file):
        state = torch.load(ctrl_file, map_location={"cuda:0": "cpu"}, weights_only=True)
        cur_best = -state["reward"]
        controller.load_state_dict(state["state_dict"])
        print(f"Previous best was {-cur_best}...")

    parameters = controller.parameters()
    es = cma.CMAEvolutionStrategy(
        flatten_parameters(parameters), 0.1, {"popsize": pop_size}
    )

    epoch = 0
    log_step = 3
    while not es.stop():
        if cur_best is not None and -cur_best > config.target_return:
            print("Already better than target, breaking...")
            break

        r_list = [0] * pop_size
        solutions = es.ask()

        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.put((s_id, s))

        if config.display:
            pbar = tqdm(total=pop_size * n_samples)
        for _ in range(pop_size * n_samples):
            while r_queue.empty():
                sleep(0.1)
            r_s_id, r = r_queue.get()
            r_list[r_s_id] += r / n_samples
            if config.display:
                pbar.update(1)
        if config.display:
            pbar.close()

        es.tell(solutions, r_list)
        es.disp()

        if epoch % log_step == log_step - 1:
            best_params, best, std_best = evaluate(
                solutions, r_list, n_samples * pop_size, p_queue, r_queue
            )
            print(f"Current evaluation: {best}")
            if not cur_best or cur_best > best:
                cur_best = best
                print(f"Saving new best with value {-cur_best} +- {std_best}...")
                load_parameters(best_params, controller)
                torch.save(
                    {
                        "epoch": epoch,
                        "reward": -cur_best,
                        "state_dict": controller.state_dict(),
                    },
                    join(ctrl_dir, "best.tar"),
                )
            if -best > config.target_return:
                print(f"Terminating controller training with value {best}...")
                break

        epoch += 1

    es.result_pretty()
    e_queue.put("EOP")
