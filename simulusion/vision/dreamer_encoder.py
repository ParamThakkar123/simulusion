import torch
import torch.nn as nn
from simulusion.export import ExportableAgentMixin

_str_to_activation = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}


class ConvEncoder(ExportableAgentMixin, nn.Module):
    """Convolutional observation encoder used by Dreamer world models.

    This encoder transforms raw image observations (typically RGB frames from
    environments like Atari or DeepMind Control) into compact latent embeddings
    that can be processed by the RSSM (Recurrent State-Space Model).

    - Input: (B, C, H, W) raw images, values in [-0.5, 0.5]
    - Process: 4 convolutional layers with stride 2, halving spatial dimensions
    - Output: (B, embed_size) compact representation

    The encoder uses a depth doubling pattern: 32 -> 64 -> 128 -> 256 channels.
    After convolutions, a fully connected layer projects from 1024 features to
    the desired embedding size.

    **Usage with Dreamer**::

        encoder = ConvEncoder(
            input_shape=(3, 64, 64),  # RGB 64x64 images
            embed_size=256,           # RSSM observation embedding size
            activation='relu'         # Activation function
        )
        obs_embedding = encoder(observation)  # (B, 256)

    Args:
        input_shape: Tuple (C, H, W) for input images, typically (3, 64, 64)
        embed_size: Output embedding dimension, typically 256 or 1024
        activation: Activation function name ('relu', 'elu', 'tanh', etc.)
        depth: Base channel depth for first layer (default 32)
    """

    def __init__(
        self, input_shape: tuple, embed_size: int, activation: str, depth: int = 32
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.act_fn = _str_to_activation[activation]
        self.depth = depth
        self.kernels = [4, 4, 4, 4]

        self.embed_size = embed_size

        layers: list[nn.Module] = []
        for i, kernel_size in enumerate(self.kernels):
            in_ch = input_shape[0] if i == 0 else self.depth * (2 ** (i - 1))
            out_ch = self.depth * (2**i)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=2))
            layers.append(self.act_fn)

        self.conv_block = nn.Sequential(*layers)
        self.fc = (
            nn.Identity()
            if self.embed_size == 1024
            else nn.Linear(1024, self.embed_size)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        reshaped = inputs.reshape(-1, *self.input_shape)
        embed = self.conv_block(reshaped)
        embed = embed.reshape(
            *inputs.shape[:-3], embed.shape[1] * embed.shape[2] * embed.shape[3]
        )
        embed = self.fc(embed)

        return embed
