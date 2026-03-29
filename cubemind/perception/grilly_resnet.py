"""ResNet-18 in pure grilly — Vulkan GPU accelerated.

Minimal ResNet-18 using grilly.nn modules (Conv2d, BatchNorm2d, MaxPool2d,
Linear). All forward passes run on Vulkan compute shaders via the GEMM
im2col path. No PyTorch dependency.

Can load weights exported from PyTorch for transfer learning, or
train from scratch with grilly's conv backward.
"""

from __future__ import annotations

import numpy as np

from grilly.nn.conv import Conv2d
from grilly.nn.normalization import BatchNorm2d
from grilly.nn.pooling import MaxPool2d
from grilly.nn.linear import Linear


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0).astype(np.float32)


class BasicBlock:
    """ResNet basic block: two 3x3 convs with skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        self.conv1 = Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_ch)
        self.conv2 = Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_ch)

        # Shortcut for dimension mismatch
        self.shortcut = None
        self.shortcut_bn = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            self.shortcut_bn = BatchNorm2d(out_ch)

    def forward(self, x: np.ndarray) -> np.ndarray:
        identity = x

        out = _relu(self.bn1.forward(self.conv1.forward(x)))
        out = self.bn2.forward(self.conv2.forward(out))

        if self.shortcut is not None:
            identity = self.shortcut_bn.forward(self.shortcut.forward(x))

        out = _relu(out + identity)
        return out

    def parameters(self):
        params = []
        for m in [self.conv1, self.bn1, self.conv2, self.bn2]:
            if hasattr(m, 'weight') and m.weight is not None:
                params.append(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                params.append(m.bias)
        if self.shortcut is not None:
            params.append(self.shortcut.weight)
            if self.shortcut_bn is not None:
                if self.shortcut_bn.weight is not None:
                    params.append(self.shortcut_bn.weight)
                if self.shortcut_bn.bias is not None:
                    params.append(self.shortcut_bn.bias)
        return params


class ResNet18:
    """ResNet-18 in pure grilly.

    Architecture: conv1(7x7) -> bn -> relu -> maxpool
                  -> layer1(2 blocks, 64ch)
                  -> layer2(2 blocks, 128ch, stride 2)
                  -> layer3(2 blocks, 256ch, stride 2)
                  -> layer4(2 blocks, 512ch, stride 2)
                  -> avgpool -> flatten -> (512,)

    Args:
        in_channels: Input channels (1 for grayscale, 3 for RGB).
        num_classes: Output dimension (0 = feature extractor only).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 0):
        self.conv1 = Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.maxpool = MaxPool2d(3, stride=2)

        self.layer1 = [BasicBlock(64, 64), BasicBlock(64, 64)]
        self.layer2 = [BasicBlock(64, 128, stride=2), BasicBlock(128, 128)]
        self.layer3 = [BasicBlock(128, 256, stride=2), BasicBlock(256, 256)]
        self.layer4 = [BasicBlock(256, 512, stride=2), BasicBlock(512, 512)]

        self.feature_dim = 512
        self.fc = Linear(512, num_classes) if num_classes > 0 else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: (N, C, H, W) float32.

        Returns:
            (N, 512) features if num_classes=0, else (N, num_classes).
        """
        # Stem
        x = _relu(self.bn1.forward(self.conv1.forward(x)))
        x = self.maxpool.forward(x)

        # Residual layers
        for block in self.layer1:
            x = block.forward(x)
        for block in self.layer2:
            x = block.forward(x)
        for block in self.layer3:
            x = block.forward(x)
        for block in self.layer4:
            x = block.forward(x)

        # Global average pool (matches pretrained ImageNet weights)
        # NOTE: Isohanni 2025 showed max pool is better for subtle features
        # (99.7% vs 96.9%) but requires full backbone fine-tuning.
        # Use avg pool with frozen pretrained weights, switch to max pool
        # when fine-tuning the full backbone.
        x = x.mean(axis=(2, 3))  # (N, 512)

        if self.fc is not None:
            x = self.fc.forward(x.reshape(x.shape[0], -1))

        return x

    def load_pytorch_weights(self, state_dict: dict) -> int:
        """Load weights from a PyTorch ResNet-18 state_dict.

        Maps PyTorch parameter names to grilly modules.

        Args:
            state_dict: PyTorch state_dict (keys -> numpy arrays).

        Returns:
            Number of parameters loaded.
        """
        loaded = 0

        def _set(module, attr, key):
            nonlocal loaded
            if key in state_dict:
                val = np.asarray(state_dict[key], dtype=np.float32)
                setattr(module, attr, val)
                loaded += 1

        # Stem
        _set(self.conv1, 'weight', 'conv1.weight')
        _set(self.bn1, 'weight', 'bn1.weight')
        _set(self.bn1, 'bias', 'bn1.bias')
        _set(self.bn1, 'running_mean', 'bn1.running_mean')
        _set(self.bn1, 'running_var', 'bn1.running_var')

        # Residual layers
        for layer_idx, layer in enumerate([self.layer1, self.layer2,
                                            self.layer3, self.layer4], 1):
            for block_idx, block in enumerate(layer):
                prefix = f"layer{layer_idx}.{block_idx}"
                _set(block.conv1, 'weight', f'{prefix}.conv1.weight')
                _set(block.bn1, 'weight', f'{prefix}.bn1.weight')
                _set(block.bn1, 'bias', f'{prefix}.bn1.bias')
                _set(block.bn1, 'running_mean', f'{prefix}.bn1.running_mean')
                _set(block.bn1, 'running_var', f'{prefix}.bn1.running_var')
                _set(block.conv2, 'weight', f'{prefix}.conv2.weight')
                _set(block.bn2, 'weight', f'{prefix}.bn2.weight')
                _set(block.bn2, 'bias', f'{prefix}.bn2.bias')
                _set(block.bn2, 'running_mean', f'{prefix}.bn2.running_mean')
                _set(block.bn2, 'running_var', f'{prefix}.bn2.running_var')
                if block.shortcut is not None:
                    _set(block.shortcut, 'weight',
                         f'{prefix}.downsample.0.weight')
                    _set(block.shortcut_bn, 'weight',
                         f'{prefix}.downsample.1.weight')
                    _set(block.shortcut_bn, 'bias',
                         f'{prefix}.downsample.1.bias')
                    _set(block.shortcut_bn, 'running_mean',
                         f'{prefix}.downsample.1.running_mean')
                    _set(block.shortcut_bn, 'running_var',
                         f'{prefix}.downsample.1.running_var')

        if self.fc is not None:
            _set(self.fc, 'weight', 'fc.weight')
            _set(self.fc, 'bias', 'fc.bias')

        return loaded

    def parameters(self):
        params = []
        for m in [self.conv1, self.bn1]:
            if hasattr(m, 'weight') and m.weight is not None:
                params.append(m.weight)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                params.extend(block.parameters())
        if self.fc is not None:
            params.append(self.fc.weight)
            if self.fc.bias is not None:
                params.append(self.fc.bias)
        return params


def load_pretrained_resnet18(in_channels: int = 1) -> ResNet18:
    """Load ResNet-18 with ImageNet pretrained weights via PyTorch.

    Downloads weights once via torchvision, converts to numpy,
    loads into grilly ResNet18. PyTorch only needed for download.

    Args:
        in_channels: 1 for grayscale (averages conv1 across RGB), 3 for RGB.

    Returns:
        Grilly ResNet18 with pretrained weights.
    """
    from torchvision.models import resnet18, ResNet18_Weights

    # Download PyTorch weights
    pt_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    sd = {k: v.detach().cpu().numpy() for k, v in pt_model.state_dict().items()}

    # Handle grayscale: average conv1 weights across RGB channels
    if in_channels == 1:
        w = sd['conv1.weight']  # (64, 3, 7, 7)
        sd['conv1.weight'] = w.mean(axis=1, keepdims=True)  # (64, 1, 7, 7)

    # Build grilly model and load weights
    model = ResNet18(in_channels=in_channels, num_classes=0)
    n_loaded = model.load_pytorch_weights(sd)
    print(f"Loaded {n_loaded} parameters from PyTorch ResNet-18")

    return model


if __name__ == "__main__":
    import time

    print("Building grilly ResNet-18 (grayscale)...")
    model = load_pretrained_resnet18(in_channels=1)

    # Test forward pass
    x = np.random.randn(1, 1, 224, 224).astype(np.float32)

    t0 = time.perf_counter()
    features = model.forward(x)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"Output shape: {features.shape}")
    print(f"Feature dim: {model.feature_dim}")
    print(f"Latency: {elapsed:.1f}ms")

    # Test with 80x80 (pad to 224)
    small = np.random.randn(1, 1, 80, 80).astype(np.float32)
    # Resize via repeat
    padded = np.zeros((1, 1, 224, 224), dtype=np.float32)
    # Simple nearest-neighbor upscale
    for i in range(224):
        for j in range(224):
            si = min(int(i * 80 / 224), 79)
            sj = min(int(j * 80 / 224), 79)
            padded[0, 0, i, j] = small[0, 0, si, sj]

    t0 = time.perf_counter()
    feat2 = model.forward(padded)
    elapsed2 = (time.perf_counter() - t0) * 1000
    print(f"80x80 upscaled: {feat2.shape}, {elapsed2:.1f}ms")
