"""Lightweight DenseNet in pure grilly for RAVEN perception.

DenseNet concatenates features from ALL previous layers, keeping
low-level edge/texture info accessible throughout. Better than ResNet
for subtle visual differences (shape edges, fill patterns).

Architecture (DenseNet-Small for 80x80 grayscale):
    Conv(1, 32, 3, stride=2, pad=1) + ReLU + MaxPool(2)  -> (32, 20, 20)
    DenseBlock(4 layers, growth=16)                        -> (96, 20, 20)
    Transition(96 -> 48) + AvgPool(2)                      -> (48, 10, 10)
    DenseBlock(4 layers, growth=16)                        -> (112, 10, 10)
    GlobalAvgPool                                          -> (112,)
    Linear(112, d_vsa)                                     -> (512,)

Total growth: 32 + 4*16 = 96 -> compress to 48 -> 48 + 4*16 = 112
Much smaller than ResNet-18 (112 vs 512 feature dim), faster inference.
"""

from __future__ import annotations

import numpy as np

from grilly.nn.conv import Conv2d
from grilly.nn.pooling import MaxPool2d
from grilly.nn.linear import Linear


def _relu(x):
    return np.maximum(x, 0).astype(np.float32)


def _concat_channels(a, b):
    """Concatenate along channel dimension (axis=1)."""
    return np.concatenate([a, b], axis=1).astype(np.float32)


class DenseLayer:
    """Single DenseNet layer: BN-ReLU-Conv(3x3).

    Input: all accumulated channels from previous layers.
    Output: growth_rate new channels (concatenated by the block).
    """

    def __init__(self, in_channels: int, growth_rate: int):
        self.conv = Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.in_channels = in_channels
        self.growth_rate = growth_rate

    def forward(self, x: np.ndarray) -> np.ndarray:
        return _relu(self.conv.forward(x))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # ReLU backward mask from cached forward
        return self.conv.backward(grad)

    def zero_grad(self):
        self.conv.zero_grad()


class DenseBlock:
    """DenseNet block: stack of DenseLayers with concat connections.

    Each layer receives ALL previous features concatenated along channels.
    Output channels = in_channels + num_layers * growth_rate.
    """

    def __init__(self, in_channels: int, growth_rate: int = 16, num_layers: int = 4):
        self.layers = []
        self.growth_rate = growth_rate
        ch = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.out_channels = ch

    def forward(self, x: np.ndarray) -> np.ndarray:
        features = [x]
        for layer in self.layers:
            # Concatenate ALL previous features
            combined = np.concatenate(features, axis=1).astype(np.float32)
            new = layer.forward(combined)
            features.append(new)
        return np.concatenate(features, axis=1).astype(np.float32)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()


class Transition:
    """DenseNet transition: 1x1 conv (compress channels) + AvgPool(2)."""

    def __init__(self, in_channels: int, out_channels: int):
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = _relu(self.conv.forward(x))
        # Average pool 2x2 stride 2
        n, c, h, w = x.shape
        oh, ow = h // 2, w // 2
        out = np.zeros((n, c, oh, ow), dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                out[:, :, i, j] = x[:, :, i*2:i*2+2, j*2:j*2+2].mean(axis=(2, 3))
        return out

    def zero_grad(self):
        self.conv.zero_grad()


class DenseNet:
    """Lightweight DenseNet for RAVEN perception.

    Args:
        in_channels: Input channels (1 for grayscale).
        growth_rate: New channels per dense layer.
        block_layers: Layers per dense block.
        num_blocks: Number of dense blocks.
        init_channels: Initial conv output channels.
        num_classes: Output dimension (0 = feature extractor).
    """

    def __init__(
        self,
        in_channels: int = 1,
        growth_rate: int = 16,
        block_layers: int = 4,
        num_blocks: int = 2,
        init_channels: int = 32,
        num_classes: int = 0,
    ):
        # Initial conv + pool
        self.conv0 = Conv2d(in_channels, init_channels, 3, stride=2, padding=1)
        self.pool0 = MaxPool2d(2, 2)

        # Dense blocks + transitions
        self.blocks = []
        self.transitions = []
        ch = init_channels

        for i in range(num_blocks):
            block = DenseBlock(ch, growth_rate, block_layers)
            self.blocks.append(block)
            ch = block.out_channels

            if i < num_blocks - 1:
                # Compress channels by half
                out_ch = ch // 2
                trans = Transition(ch, out_ch)
                self.transitions.append(trans)
                ch = out_ch

        self.feature_dim = ch
        self.fc = Linear(ch, num_classes) if num_classes > 0 else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Stem
        x = _relu(self.conv0.forward(x))
        x = self.pool0.forward(x)

        # Dense blocks + transitions
        for i, block in enumerate(self.blocks):
            x = block.forward(x)
            if i < len(self.transitions):
                x = self.transitions[i].forward(x)

        # Global average pool
        x = x.mean(axis=(2, 3))  # (N, feature_dim)

        if self.fc is not None:
            x = self.fc.forward(x.reshape(x.shape[0], -1))

        return x

    def zero_grad(self):
        self.conv0.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        for trans in self.transitions:
            trans.zero_grad()
        if self.fc is not None:
            self.fc.zero_grad()


if __name__ == "__main__":
    import time

    print("Building grilly DenseNet-Small (grayscale)...")
    model = DenseNet(in_channels=1, growth_rate=16, block_layers=4,
                     num_blocks=2, init_channels=32)
    print(f"Feature dim: {model.feature_dim}")

    # Test forward
    x = np.random.randn(1, 1, 80, 80).astype(np.float32) * 0.1

    t0 = time.perf_counter()
    feat = model.forward(x)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"Output: {feat.shape}, latency: {elapsed:.1f}ms")

    # Second forward (warm)
    t0 = time.perf_counter()
    feat2 = model.forward(x)
    elapsed2 = (time.perf_counter() - t0) * 1000
    print(f"Warm: {feat2.shape}, latency: {elapsed2:.1f}ms")
