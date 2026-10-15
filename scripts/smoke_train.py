import torch
from omegaconf import OmegaConf

from simulusion.configs.diamond_config import DiamondConfig
from simulusion.training.train_diamond import DiamondAgent


def main(game: str, device: str):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DiamondConfig(game=game, preset="small", device=device)

    cfg.num_epochs = 3
    cfg.environment_steps_per_epoch = 20
    cfg.training_steps_per_epoch = 10
    cfg.batch_size = 4
    cfg.num_sampling_steps = 2
    cfg.eval_interval = 1
    cfg.save_interval = 1
    cfg.log_interval = 1

    print(f"Smoke-training DIAMOND on {cfg.game} (device={cfg.device})")
    agent = DiamondAgent(cfg)
    agent.train()


if __name__ == "__main__":
    cli_cfg = OmegaConf.from_cli()
    game = cli_cfg.get("game", "Breakout-v5")
    device = cli_cfg.get("device", "auto")

    main(game, device)
