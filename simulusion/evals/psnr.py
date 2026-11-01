"""Peak Signal-to-Noise Ratio (PSNR) metric.

Computes the PSNR between real and generated images or video frames.
Higher PSNR values indicate higher similarity.

Reference:
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
"""

import torch


class PSNR:
    """Peak Signal-to-Noise Ratio.

    Usage:
        psnr = PSNR()
        score = psnr(real_images, generated_images)
    """

    def __init__(self, max_val: float = 1.0, batch_size: int = 64):
        self.max_val = max_val
        self.batch_size = batch_size

    def __call__(
        self,
        real: torch.Tensor,
        generated: torch.Tensor,
    ) -> float:
        """Compute mean PSNR over a batch of image/video pairs.

        Args:
            real: Reference images [N, C, H, W] or videos [N, C, T, H, W] in [0, 1].
            generated: Generated images/videos with the same shape.

        Returns:
            Mean PSNR score (higher is better).
        """
        if real.shape != generated.shape:
            raise ValueError(
                f"Shape mismatch: real {real.shape} vs generated {generated.shape}"
            )

        all_scores = []
        for i in range(0, len(real), self.batch_size):
            batch_real = real[i : i + self.batch_size]
            batch_gen = generated[i : i + self.batch_size]

            mse = torch.mean(
                (batch_real - batch_gen) ** 2, dim=tuple(range(1, batch_real.ndim))
            )
            psnr = 10.0 * torch.log10(self.max_val**2 / (mse + 1e-8))
            all_scores.append(psnr.cpu())

        return float(torch.cat(all_scores).mean())

    def __repr__(self) -> str:
        return f"PSNR(max_val={self.max_val})"
