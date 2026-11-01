import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

from typing import Any
from types import ModuleType
import copy
import importlib
import importlib.util
import logging
import sys
import torch.multiprocessing as mp
import torch.nn.functional as F
import yaml

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from simulusion.masks.multiblock import MaskCollator as MBMaskCollator
from simulusion.utils.utils import apply_masks
from simulusion.utils.jepa_utils import init_distributed, AllReduce
from simulusion.utils.jepa_utils import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter,
)
from simulusion.utils.jepa_utils import repeat_interleave_batch
from simulusion.datasets.imagenet1k import make_imagenet1k, make_imagefolder
from simulusion.datasets.cifar10 import make_cifar10
from simulusion.utils.train_utils import EarlyStopping
from simulusion.helpers.jepa_helper import load_checkpoint, init_model, init_opt
from simulusion.transforms.image import make_transforms
from simulusion.configs.jepa_config import JEPAConfig
from simulusion.experiments import (
    dump_config,
    load_experiment_config,
    parse_experiment_args,
)

# Imported on first use by `_require_wandb`, not at module load: a W&B install
# that is present but broken would otherwise make this whole module (and every
# test that imports it) unusable even with W&B logging turned off.
_wandb: ModuleType | None = None

log_timings = True
log_freq = 10
checkpoint_freq = 50

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def build_loss_fn(loss_type: str) -> Any:
    """Return the prediction loss used to compare predictor and target tokens.

    The I-JEPA paper (Sec. 3) defines the objective as the squared L2 distance
    between predicted and target patch representations, averaged over the ``M``
    target blocks. Two reductions of that same objective are offered:

    * ``"l2"`` (default) averages the squared error over patches and channels
      as well. It has the same minimizer and the same gradient direction as the
      paper's formula, but a magnitude that does not grow with block size --
      which is what the paper's learning rates were tuned against, since the
      reference implementation also reduces by mean.
    * ``"l2_sum"`` is the literal formula: summed over patches within a block,
      averaged over blocks. Its gradients are larger than ``"l2"``'s by roughly
      the number of patches times the embedding dimension, so lower the
      learning rate accordingly.

    ``"smooth_l1"`` reproduces the reference implementation, which uses
    Smooth-L1 rather than the paper's L2.
    """

    def l2_sum(z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # z, h: (M * B, num_patches_per_block, embed_dim). Sum the squared L2
        # distance over patches, average over blocks and batch.
        return ((z - h) ** 2).sum(dim=-1).sum(dim=-1).mean()

    losses = {
        "l2": F.mse_loss,
        "l2_sum": l2_sum,
        "smooth_l1": F.smooth_l1_loss,
    }
    if loss_type not in losses:
        raise ValueError(
            f"Unknown loss_type {loss_type!r}. Available: {sorted(losses)}"
        )
    return losses[loss_type]


def _require_wandb() -> Any:
    """Return the wandb module, or explain how to install the ``ml`` extra."""
    global _wandb
    if _wandb is None:
        if importlib.util.find_spec("wandb") is None:
            raise ImportError(
                "Weights & Biases is required for JEPA sweeps. "
                "Install it with `pip install simulusion[ml]`."
            )
        _wandb = importlib.import_module("wandb")
    return _wandb


def make_jepa_loaders(
    args: Any,
    *,
    transform: Any,
    mask_collator: Any,
    world_size: int = 1,
    rank: int = 0,
    with_val: bool = False,
) -> tuple:
    """I-JEPA's ``(train_loader, train_sampler, val_loader)`` for a nested config.

    The loaders shard across ``world_size`` processes themselves (a
    ``DistributedSampler`` per rank). ``val_loader`` is built only with
    ``with_val``, from CIFAR-10's test split or ``data.val_split`` of an image
    folder, with the same mask collator, so its loss is the training objective.
    """
    batch_size = args["data"]["batch_size"]
    pin_mem = args["data"]["pin_mem"]
    num_workers = args["data"]["num_workers"]
    root_path = args["data"]["root_path"]
    image_folder = args["data"]["image_folder"]
    copy_data = args["meta"]["copy_data"]
    dataset_type = args["data"]["dataset"]
    val_split = args["data"]["val_split"]
    download = args["data"].get("download", False)
    if dataset_type.lower() == "imagenet":
        _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True,
        )
    elif dataset_type.lower() == "cifar10":
        _, unsupervised_loader, unsupervised_sampler = make_cifar10(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            drop_last=True,
            train=True,
            download=download,  # pass through
        )
    else:
        _, unsupervised_loader, unsupervised_sampler = make_imagefolder(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            drop_last=True,
            val_split=val_split,
        )
    # Held-out loader for early stopping. Built with the same mask collator, so
    # the validation loss is the exact objective being trained, just on data the
    # encoder never sees.
    val_loader = None
    if with_val:
        if dataset_type.lower() == "cifar10":
            _, val_loader, _ = make_cifar10(
                transform=transform,
                batch_size=batch_size,
                collator=mask_collator,
                pin_mem=pin_mem,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                root_path=root_path,
                drop_last=True,
                train=False,
                download=download,
            )
        else:
            if not val_split:
                raise ValueError(
                    "optimization.early_stopping needs data.val_split > 0 so "
                    "there is held-out data to measure; pass e.g. "
                    "data.val_split=0.05"
                )
            _, val_loader, _ = make_imagefolder(
                transform=transform,
                batch_size=batch_size,
                collator=mask_collator,
                pin_mem=pin_mem,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                root_path=root_path,
                image_folder=image_folder,
                drop_last=True,
                val_split=val_split,
                split="val",
            )
    return unsupervised_loader, unsupervised_sampler, val_loader


def main(args: Any = None, resume_preempt: bool = False) -> Any:
    """Run JEPA training using a CLI argv, nested dict, or `JEPAConfig` instance.

    This entrypoint initializes distributed context, data pipeline, masking,
    models, optimizers/schedulers, checkpointing, and the full epoch loop.
    """
    if args is None or isinstance(args, list):
        return main_from_cli(args)
    if isinstance(args, JEPAConfig):
        args = args.to_train_dict()

    logging_args = args.get("logging", {})
    if logging_args.get("enable_sweep", False):
        wandb = _require_wandb()
        sweep_id = wandb.sweep(
            logging_args.get("sweep_config", {}),
            project=logging_args.get("wandb_project"),
            entity=logging_args.get("wandb_entity") or None,
        )
        wandb.agent(sweep_id, function=sweep_train)
        return

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args["meta"]["use_bfloat16"]
    model_name = args["meta"]["model_name"]
    load_model = args["meta"]["load_checkpoint"] or resume_preempt
    r_file = args["meta"]["read_checkpoint"]
    pred_depth = args["meta"]["pred_depth"]
    pred_emb_dim = args["meta"]["pred_emb_dim"]
    loss_fn = build_loss_fn(args["meta"].get("loss_type", "l2"))
    from simulusion.utils.device import get_default_device

    device = get_default_device()
    if device.type == "cuda":
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    elif device.type == "cpu":
        print("WARNING: no GPU (CUDA or MPS) available, using CPU")

    # -- DATA
    use_gaussian_blur = args["data"]["use_gaussian_blur"]
    use_horizontal_flip = args["data"]["use_horizontal_flip"]
    use_color_distortion = args["data"]["use_color_distortion"]
    color_jitter = args["data"]["color_jitter_strength"]
    # --
    batch_size = args["data"]["batch_size"]
    crop_size = args["data"]["crop_size"]
    crop_scale = args["data"]["crop_scale"]
    # --

    # -- MASK
    allow_overlap = args["mask"][
        "allow_overlap"
    ]  # whether to allow overlap b/w context and target blocks
    patch_size = args["mask"]["patch_size"]  # patch-size for model training
    num_enc_masks = args["mask"]["num_enc_masks"]  # number of context blocks
    min_keep = args["mask"]["min_keep"]  # min number of patches in context block
    enc_mask_scale = args["mask"]["enc_mask_scale"]  # scale of context blocks
    num_pred_masks = args["mask"]["num_pred_masks"]  # number of target blocks
    pred_mask_scale = args["mask"]["pred_mask_scale"]  # scale of target blocks
    aspect_ratio = args["mask"]["aspect_ratio"]  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args["optimization"]["ema"]
    ipe_scale = args["optimization"]["ipe_scale"]  # scheduler scale factor (def: 1.0)
    wd = float(args["optimization"]["weight_decay"])
    final_wd = float(args["optimization"]["final_weight_decay"])
    num_epochs = args["optimization"]["epochs"]
    warmup = args["optimization"]["warmup"]
    start_lr = args["optimization"]["start_lr"]
    lr = args["optimization"]["lr"]
    final_lr = args["optimization"]["final_lr"]
    lr_reference_batch_size = args["optimization"].get("lr_reference_batch_size")

    # -- LOGGING
    folder = args["logging"]["folder"]
    tag = args["logging"]["write_tag"]
    enable_wandb = args["logging"]["enable_wandb"]
    wandb_project = args["logging"]["wandb_project"]
    wandb_entity = args["logging"]["wandb_entity"]

    os.makedirs(folder, exist_ok=True)  # ensure output dir exists

    dump = os.path.join(folder, "params-ijepa.yaml")
    with open(dump, "w") as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- scale the learning rates to the effective batch size. The paper's
    #    values (start 1e-4 -> ref 1e-3) are quoted for a batch size of 2048.
    if lr_reference_batch_size:
        lr_scale = (batch_size * world_size) / float(lr_reference_batch_size)
        if lr_scale != 1.0:
            start_lr, lr = start_lr * lr_scale, lr * lr_scale
            logger.info(
                "Scaled learning rates by %.4g (effective batch %d vs reference %d): "
                "start_lr=%.3e lr=%.3e",
                lr_scale,
                batch_size * world_size,
                lr_reference_batch_size,
                start_lr,
                lr,
            )

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    save_path = os.path.join(folder, f"{tag}" + "-ep{epoch}.pth.tar")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")
    load_path: str | None = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        enable_wandb,
        wandb_project,
        wandb_entity,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "mask-A"),
        ("%.5f", "mask-B"),
        ("%d", "time (ms)"),
    )

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep,
    )

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter,
    )

    # -- init data-loaders/samplers
    early_stopping = args["optimization"].get("early_stopping", False)
    unsupervised_loader, unsupervised_sampler, val_loader = make_jepa_loaders(
        args,
        transform=transform,
        mask_collator=mask_collator,
        world_size=world_size,
        rank=rank,
        with_val=early_stopping,
    )

    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )

    is_distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and world_size > 1
    )
    if is_distributed:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
    # keep modules unwrapped when not distributed
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        assert load_path is not None
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = (
            load_checkpoint(
                device=device,
                r_path=load_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=optimizer,
                scaler=scaler,
            )
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch: int) -> None:
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f"{epoch + 1}"))

    @torch.no_grad()
    def validate() -> float:
        """Mean held-out loss, using the same objective the epoch just trained."""
        if val_loader is None:
            raise RuntimeError("validate() called without a held-out loader")
        encoder.eval()
        predictor.eval()
        total, count = 0.0, 0
        for udata, masks_enc, masks_pred in val_loader:
            imgs = udata[0].to(device, non_blocking=True)
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                h = target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),))
                h = apply_masks(h, masks_pred)
                h = repeat_interleave_batch(h, len(imgs), repeat=len(masks_enc))
                z = predictor(encoder(imgs, masks_enc), masks_enc, masks_pred)
                batch_loss = float(loss_fn(z, h))
            total += batch_loss * len(imgs)
            count += len(imgs)
        encoder.train()
        predictor.train()
        return total / max(count, 1)

    stopper = None
    best_val = float("inf")
    if early_stopping:
        stopper = EarlyStopping(
            mode="min",
            patience=args["optimization"].get("patience", 10),
            threshold=args["optimization"].get("min_delta", 1e-4),
        )

    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs() -> tuple:
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)

            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step() -> tuple:
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target() -> torch.Tensor:
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context() -> torch.Tensor:
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def reduced_loss(z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
                    loss_val = loss_fn(z, h)
                    loss_val = AllReduce.apply(loss_val)  # type: ignore[no-untyped-call]
                    return loss_val

                # Step 1. Forward
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bfloat16,
                ):
                    h = forward_target()
                    z = forward_context()
                    loss = reduced_loss(z, h)

                #  Step 2. Backward & step. bfloat16 has the dynamic range of
                #  fp32, so unlike fp16 it needs no gradient scaling.
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                enc_for_log = encoder.module if is_distributed else encoder
                grad_stats = grad_logger(enc_for_log.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    # One fused multi-tensor op per step instead of two kernels
                    # per parameter tensor, and `alpha=` applies the (1 - m)
                    # scale inside the add rather than allocating a scaled copy
                    # of every source tensor first. Same arithmetic.
                    target_params: list[torch.Tensor] = list(
                        target_encoder.parameters()
                    )
                    online_params: list[torch.Tensor] = [
                        p.detach() for p in encoder.parameters()
                    ]
                    torch._foreach_mul_(target_params, m)
                    torch._foreach_add_(target_params, online_params, alpha=1.0 - m)

                return (float(loss.detach()), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            def log_stats() -> None:
                global_step = epoch * ipe + itr
                csv_logger.log(
                    global_step,
                    epoch + 1,
                    itr,
                    loss,
                    maskA_meter.val,
                    maskB_meter.val,
                    etime,
                )
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "masks: %.1f %.1f "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "(%.1f ms)"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            maskA_meter.avg,
                            maskB_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            time_meter.avg,
                        )
                    )

                    if grad_stats is not None:
                        logger.info(
                            "[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)"
                            % (
                                epoch + 1,
                                itr,
                                grad_stats.first_layer,
                                grad_stats.last_layer,
                                grad_stats.min,
                                grad_stats.max,
                            )
                        )

            log_stats()

            assert not np.isnan(loss), "loss is nan"

        logger.info("avg. loss %.3f" % loss_meter.avg)
        save_checkpoint(epoch + 1)

        if stopper is not None:
            val_loss = validate()
            stopper.step(val_loss)
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
            logger.info(
                "epoch %d val loss %.4f (best %.4f%s)"
                % (epoch + 1, val_loss, best_val, "*" if improved else "")
            )
            if stopper.stop:
                logger.info(
                    "Early stopping at epoch %d: held-out loss has not improved "
                    "by %s for %s epochs."
                    % (
                        epoch + 1,
                        args["optimization"].get("min_delta", 1e-4),
                        args["optimization"].get("patience", 10),
                    )
                )
                break


def sweep_train() -> None:
    """Function for WandB sweep agent."""
    wandb = _require_wandb()
    with wandb.init():
        cfg = JEPAConfig()
        # Update config with sweep parameters
        for key, value in wandb.config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        main(cfg.to_train_dict())


def main_from_cli(argv: list[str] | None = None) -> Any:
    """Compose JEPA config from YAML/dot-list overrides and launch training."""
    parsed = parse_experiment_args(argv, description="Train JEPA")
    cfg_dict = load_experiment_config(
        JEPAConfig().to_dict(), parsed.config, parsed.overrides
    )
    if parsed.print_config:
        print(dump_config(cfg_dict))
        return cfg_dict

    logging_cfg = cfg_dict.get("logging", {})
    if logging_cfg.get("enable_sweep", False):
        wandb = _require_wandb()
        sweep_id = wandb.sweep(
            logging_cfg.get("sweep_config", {}),
            project=logging_cfg.get("wandb_project", "simulusion"),
            entity=logging_cfg.get("wandb_entity") or None,
        )
        wandb.agent(sweep_id, function=sweep_train)
    else:
        main(cfg_dict)
    return cfg_dict


if __name__ == "__main__":
    main_from_cli()
