from typing import Any

_PROBING = (
    "DCIScores",
    "Latents",
    "LinearProbe",
    "MIGScores",
    "MLPProbe",
    "ProbeDataset",
    "ProbeScores",
    "StateTargets",
    "collect_probe_episodes",
    "dci",
    "extract_latents",
    "factor_groups",
    "fit_linear_probe",
    "fit_mlp_probe",
    "latent_features",
    "mig",
    "pixel_features",
    "r2_score",
    "reinitialized",
    "score_latents",
    "unobservable_dims",
)


def __getattr__(name: str) -> Any:
    if name == "FID":
        from simulusion.evals.fid import FID

        return FID
    if name == "FVD":
        from simulusion.evals.fvd import FVD

        return FVD
    if name == "LPIPS":
        from simulusion.evals.lpips import LPIPS

        return LPIPS
    if name == "PSNR":
        from simulusion.evals.psnr import PSNR

        return PSNR
    if name in _PROBING:
        from simulusion.evals import probing

        return getattr(probing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FID", "FVD", "LPIPS", "PSNR", *_PROBING]
