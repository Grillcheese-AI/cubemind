"""Lightweight DenseNet in pure grilly with full backward support.

DenseNet concatenates features from ALL previous layers. Backward
distributes gradients through the concat→conv chain in reverse.

Architecture (DenseNet-Small for 80x80 grayscale):
    Conv(1, 32, 3, stride=2, pad=1) + ReLU + MaxPool(2)  -> (32, 20, 20)
    DenseBlock(4 layers, growth=16)                        -> (96, 20, 20)
    Transition(96 -> 48) + AvgPool(2)                      -> (48, 10, 10)
    DenseBlock(4 layers, growth=16)                        -> (112, 10, 10)
    GlobalAvgPool                                          -> (112,)
"""

from __future__ import annotations

import numpy as np

from grilly.nn.conv import Conv2d
from grilly.nn.pooling import MaxPool2d
from grilly.nn.linear import Linear


def _relu(x):
    return np.maximum(x, 0).astype(np.float32)


class DenseLayer:
    """Single DenseNet layer: Conv(3x3) + ReLU with cached activations."""

    def __init__(self, in_channels: int, growth_rate: int):
        self.conv = Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self._conv_out = None  # cached for ReLU backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._conv_out = self.conv.forward(x)
        return _relu(self._conv_out)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # ReLU backward: mask where conv output was > 0
        if self._conv_out is not None:
            grad = grad * (self._conv_out > 0).astype(np.float32)
        return self.conv.backward(grad)

    def zero_grad(self):
        self.conv.zero_grad()
        self._conv_out = None


class DenseBlock:
    """DenseNet block with full backward through concat connections."""

    def __init__(self, in_channels: int, growth_rate: int = 16, num_layers: int = 4):
        self.layers = []
        self.growth_rate = growth_rate
        ch = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.out_channels = ch
        self.in_channels = in_channels
        self._feature_channels = []  # channel counts for backward split

    def forward(self, x: np.ndarray) -> np.ndarray:
        features = [x]
        self._feature_channels = [x.shape[1]]

        for layer in self.layers:
            combined = np.concatenate(features, axis=1).astype(np.float32)
            new = layer.forward(combined)
            features.append(new)
            self._feature_channels.append(new.shape[1])

        return np.concatenate(features, axis=1).astype(np.float32)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward through dense block.

        Output = concat(x, layer0_out, layer1_out, ...).
        Split gradient, route back through each layer's conv.
        """
        if not self._feature_channels:
            return grad_output

        n_feats = len(self._feature_channels)  # 1 + num_layers

        # Split grad_output into per-feature gradients
        grads = [None] * n_feats
        offset = 0
        for k in range(n_feats):
            ch = self._feature_channels[k]
            grads[k] = grad_output[:, offset:offset + ch].copy()
            offset += ch

        # Backward through layers in reverse
        for i in range(len(self.layers) - 1, -1, -1):
            # Layer i produced feature i+1
            g_layer_out = grads[i + 1]

            # Backward through layer's conv
            g_combined = self.layers[i].backward(g_layer_out)

            # Layer i's input was concat(features[0], ..., features[i])
            # Distribute g_combined back to those features
            split_offset = 0
            for j in range(i + 1):
                ch = self._feature_channels[j]
                grads[j] += g_combined[:, split_offset:split_offset + ch]
                split_offset += ch

        # grads[0] is the accumulated gradient for the block input
        return grads[0]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
        self._feature_channels = []


class Transition:
    """DenseNet transition: 1x1 conv + AvgPool(2) with backward."""

    def __init__(self, in_channels: int, out_channels: int):
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._conv_out = None
        self._pre_pool_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._conv_out = self.conv.forward(x)
        activated = _relu(self._conv_out)
        self._pre_pool_shape = activated.shape
        # Average pool 2x2 stride 2
        n, c, h, w = activated.shape
        oh, ow = h // 2, w // 2
        out = np.zeros((n, c, oh, ow), dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                out[:, :, i, j] = activated[:, :, i*2:i*2+2, j*2:j*2+2].mean(axis=(2, 3))
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # AvgPool backward: distribute gradient evenly
        n, c, oh, ow = grad.shape
        _, _, h, w = self._pre_pool_shape
        g_pool = np.zeros(self._pre_pool_shape, dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                g_pool[:, :, i*2:i*2+2, j*2:j*2+2] = grad[:, :, i:i+1, j:j+1] / 4.0

        # ReLU backward
        if self._conv_out is not None:
            g_pool = g_pool * (self._conv_out > 0).astype(np.float32)

        # Conv backward
        return self.conv.backward(g_pool)

    def zero_grad(self):
        self.conv.zero_grad()
        self._conv_out = None


class DenseNet:
    """Lightweight DenseNet with full backward for end-to-end training."""

    def __init__(
        self,
        in_channels: int = 1,
        growth_rate: int = 16,
        block_layers: int = 4,
        num_blocks: int = 2,
        init_channels: int = 32,
        num_classes: int = 0,
    ):
        self.conv0 = Conv2d(in_channels, init_channels, 3, stride=2, padding=1)
        self.pool0 = MaxPool2d(2, 2)

        self.blocks = []
        self.transitions = []
        ch = init_channels

        for i in range(num_blocks):
            block = DenseBlock(ch, growth_rate, block_layers)
            self.blocks.append(block)
            ch = block.out_channels
            if i < num_blocks - 1:
                out_ch = ch // 2
                trans = Transition(ch, out_ch)
                self.transitions.append(trans)
                ch = out_ch

        self.feature_dim = ch
        self.fc = Linear(ch, num_classes) if num_classes > 0 else None

        # Cache for backward
        self._conv0_out = None
        self._pre_gap_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Stem
        self._conv0_out = self.conv0.forward(x)
        x = _relu(self._conv0_out)
        x = self.pool0.forward(x)

        # Dense blocks + transitions
        for i, block in enumerate(self.blocks):
            x = block.forward(x)
            if i < len(self.transitions):
                x = self.transitions[i].forward(x)

        # Global average pool
        self._pre_gap_shape = x.shape
        x = x.mean(axis=(2, 3))  # (N, feature_dim)

        if self.fc is not None:
            x = self.fc.forward(x.reshape(x.shape[0], -1))

        return x

    def backward(self, grad_feat: np.ndarray) -> None:
        """Full backward through DenseNet.

        Args:
            grad_feat: Gradient w.r.t. feature output (1, feat_dim) or (feat_dim,).
        """
        g = grad_feat.ravel().reshape(1, -1)  # (1, feat_dim)

        # FC backward (if present)
        if self.fc is not None:
            g = self.fc.backward(g)

        # Global avg pool backward: distribute evenly across spatial dims
        n, c, h, w = self._pre_gap_shape
        g_flat = g.ravel()[:c]  # ensure we only take c values
        g = np.zeros(self._pre_gap_shape, dtype=np.float32)
        for ci in range(c):
            g[0, ci, :, :] = g_flat[ci] / (h * w)

        # Dense blocks + transitions backward (exact reverse of forward)
        # Forward: block0 → trans0 → block1 → [trans1 → block2 ...]
        # Backward: block1 → trans0 → block0
        for i in range(len(self.blocks) - 1, -1, -1):
            g = self.blocks[i].backward(g)
            # Transition i-1 sits BEFORE block i in forward
            t_idx = i - 1
            if t_idx >= 0 and t_idx < len(self.transitions):
                g = self.transitions[t_idx].backward(g)

        # Pool0 backward
        g = self.pool0.backward(g)

        # Stem conv backward with ReLU
        if self._conv0_out is not None:
            g = g * (self._conv0_out > 0).astype(np.float32)
        self.conv0.backward(g)

    def step(self, lr: float = 0.01) -> None:
        """SGD update on all conv weights using accumulated gradients."""
        for module in self._all_convs():
            w = np.asarray(module.weight)
            wg = getattr(module, '_grad_w', None) or getattr(module.weight, 'grad', None)
            if wg is not None:
                w -= lr * np.asarray(wg)
            if module.bias is not None:
                b = np.asarray(module.bias)
                bg = getattr(module, '_grad_b', None) or getattr(module.bias, 'grad', None)
                if bg is not None:
                    b -= lr * np.asarray(bg)

    def _all_convs(self):
        """Yield all Conv2d modules for parameter access."""
        yield self.conv0
        for block in self.blocks:
            for layer in block.layers:
                yield layer.conv
        for trans in self.transitions:
            yield trans.conv

    def zero_grad(self):
        self.conv0.zero_grad()
        # Don't clear _conv0_out — needed for backward ReLU mask
        for block in self.blocks:
            # Only zero conv grads, keep feature_channels for backward
            for layer in block.layers:
                layer.conv.zero_grad()
        for trans in self.transitions:
            trans.conv.zero_grad()
        if self.fc is not None:
            self.fc.zero_grad()


if __name__ == "__main__":
    import time

    print("Building grilly DenseNet-Small (grayscale)...")
    model = DenseNet(in_channels=1, growth_rate=16, block_layers=4,
                     num_blocks=2, init_channels=32)
    print(f"Feature dim: {model.feature_dim}")

    x = np.random.randn(1, 1, 80, 80).astype(np.float32) * 0.1

    # Forward
    t0 = time.perf_counter()
    feat = model.forward(x)
    print(f"Forward: {feat.shape}, {(time.perf_counter()-t0)*1000:.1f}ms")

    # Backward
    g = np.random.randn(*feat.shape).astype(np.float32) * 0.01
    model.zero_grad()
    t0 = time.perf_counter()
    model.backward(g)
    print(f"Backward: {(time.perf_counter()-t0)*1000:.1f}ms")

    # Check gradients exist
    for i, conv in enumerate(model._all_convs()):
        has_grad = (getattr(conv, '_grad_w', None) is not None or
                    getattr(conv.weight, 'grad', None) is not None)
        if i < 3 or not has_grad:
            print(f"  conv{i}: grad={'YES' if has_grad else 'NO'}")

    # Step + verify weights change
    old_w = np.asarray(model.conv0.weight).copy()
    model.step(lr=0.01)
    changed = not np.allclose(old_w, np.asarray(model.conv0.weight))
    print(f"Weights changed after step: {changed}")

    # Warm forward
    t0 = time.perf_counter()
    feat2 = model.forward(x)
    print(f"Warm forward: {(time.perf_counter()-t0)*1000:.1f}ms")
