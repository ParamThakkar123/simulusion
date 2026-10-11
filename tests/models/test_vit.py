import pytest
import numpy as np
import torch
from simulusion.models.vit import (
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
    get_1d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
    drop_path,
    DropPath,
    MLP,
    Attention,
    Block,
)


class TestPositionalEmbeddings:
    def test_get_2d_sincos_pos_embed(self):
        pos_embed = get_2d_sincos_pos_embed(embed_dim=64, grid_size=8)

        assert pos_embed.shape == (64, 64)

    def test_get_2d_sincos_pos_embed_with_cls_token(self):
        pos_embed = get_2d_sincos_pos_embed(embed_dim=64, grid_size=8, cls_token=True)

        assert pos_embed.shape == (65, 64)

    def test_get_2d_sincos_pos_embed_from_grid(self):
        grid = np.array([[[0, 1], [2, 3]], [[0, 1], [2, 3]]])
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim=64, grid=grid)

        assert pos_embed.shape == (4, 64)

    def test_get_1d_sincos_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(embed_dim=64, grid_size=10)

        assert pos_embed.shape == (10, 64)

    def test_get_1d_sincos_pos_embed_with_cls_token(self):
        pos_embed = get_1d_sincos_pos_embed(embed_dim=64, grid_size=10, cls_token=True)

        assert pos_embed.shape == (11, 64)

    def test_get_1d_sincos_pos_embed_from_grid(self):
        pos = np.arange(10, dtype=float)
        pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=64, pos=pos)

        assert pos_embed.shape == (10, 64)

    def test_get_1d_sincos_pos_embed_from_grid_odd_dim(self):
        pos = np.arange(10, dtype=float)
        with pytest.raises(AssertionError):
            get_1d_sincos_pos_embed_from_grid(embed_dim=65, pos=pos)


class TestDropPath:
    def test_drop_path_no_drop(self):
        x = torch.randn(4, 8, 16)

        output = drop_path(x, drop_prob=0.0, training=True)

        assert output.shape == x.shape
        assert torch.allclose(output, x)

    def test_drop_path_not_training(self):
        x = torch.randn(4, 8, 16)

        output = drop_path(x, drop_prob=0.5, training=False)

        assert output.shape == x.shape

    def test_drop_path_with_drop(self):
        x = torch.randn(4, 8, 16)

        output = drop_path(x, drop_prob=0.5, training=True)

        assert output.shape == x.shape

    def test_drop_path_module(self):
        drop_path_mod = DropPath(drop_prob=0.3)

        x = torch.randn(4, 8, 16)
        output = drop_path_mod(x)

        assert output.shape == x.shape


class TestMLP:
    def test_mlp_init(self):
        mlp = MLP(in_features=128, hidden_features=256, out_features=128)

        assert mlp.fc1.out_features == 256
        assert mlp.fc2.out_features == 128

    def test_mlp_forward(self):
        mlp = MLP(in_features=128, hidden_features=256)
        x = torch.randn(4, 16, 128)

        output = mlp(x)

        assert output.shape == (4, 16, 128)

    def test_mlp_default_hidden(self):
        mlp = MLP(in_features=128)
        x = torch.randn(4, 16, 128)

        output = mlp(x)

        assert output.shape == (4, 16, 128)


class TestAttention:
    def test_attention_init(self):
        attn = Attention(dim=128, num_heads=4)

        assert attn.num_heads == 4

    def test_attention_forward(self):
        attn = Attention(dim=128, num_heads=4)
        x = torch.randn(4, 16, 128)

        output = attn(x)

        assert output.shape == (4, 16, 128)

    def test_attention_with_bias(self):
        attn = Attention(dim=128, num_heads=4, qkv_bias=True)
        x = torch.randn(4, 16, 128)

        output = attn(x)

        assert output.shape == (4, 16, 128)


class TestBlock:
    def test_block_init(self):
        block = Block(dim=128, num_heads=4)

        assert block.norm1 is not None
        assert block.norm2 is not None

    def test_block_forward(self):
        block = Block(dim=128, num_heads=4)
        x = torch.randn(4, 16, 128)

        output = block(x)

        assert output.shape == (4, 16, 128)

    def test_block_with_drop_path(self):
        block = Block(dim=128, num_heads=4, drop_path=0.1)
        x = torch.randn(4, 16, 128)

        output = block(x)

        assert output.shape == (4, 16, 128)
