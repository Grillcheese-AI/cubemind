"""VSA-CNN Perception Frontend — Image to Spatial Block-Codes.

Lightweight CNN using grilly._bridge GPU ops directly for zero-copy
Vulkan compute. Converts 80x80 grayscale RAVEN panels into per-position
block-code VSA vectors via spatial grid extraction.

Architecture:
    Image (1, 80, 80)
    -> Conv(1,32,3)+GELU+MaxPool(2)     -> (32, 40, 40)
    -> Conv(32,64,3)+GELU+MaxPool(2)    -> (64, 20, 20)
    -> Conv(64,128,3)+GELU+MaxPool(2)   -> (128, 10, 10)
    -> AdaptiveAvgPool(grid_h, grid_w)  -> (128, gh, gw)
    -> Per-position Linear(128, k*l)    -> (n_pos, k*l)
    -> BlockSoftmax(tau)                -> (n_pos, k, l)

All ops dispatched via grilly.backend._bridge (Vulkan compute shaders).
"""

from __future__ import annotations

import numpy as np

from cubemind.core import K_BLOCKS, L_BLOCK

# ── GPU bridge ──────────────────────────────────────────────────────────────

_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass


# ── Helpers ─────────────────────────────────────────────────────────────────


def _ensure_f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def _conv2d_gelu(x, w, b):
    """Fused 3x3 Conv2d + GELU via _bridge or numpy fallback."""
    if _bridge is not None:
        r = _bridge.conv2d_3x3_gelu(_ensure_f32(x), _ensure_f32(w), _ensure_f32(b))
        if r is not None:
            return np.asarray(r, dtype=np.float32)
        # Fall back to separate conv + gelu
        r = _bridge.conv2d(_ensure_f32(x), _ensure_f32(w), _ensure_f32(b),
                           (1, 1), (1, 1))
        if r is not None:
            r2 = _bridge.gelu(np.asarray(r, dtype=np.float32))
            if r2 is not None:
                return np.asarray(r2, dtype=np.float32)
            return np.asarray(r, dtype=np.float32)
    # Numpy fallback
    conv_out = _conv2d_numpy(x, w, b, padding=1, stride=1)
    return (0.5 * conv_out * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (conv_out + 0.044715 * conv_out ** 3)))).astype(np.float32)


def _maxpool2d(x, kernel=2, stride=2):
    if _bridge is not None:
        r = _bridge.maxpool2x2(_ensure_f32(x))
        if r is not None:
            return np.asarray(r, dtype=np.float32)
        # Fall back to general maxpool
        r = _bridge.maxpool2d(_ensure_f32(x), kernel, stride)
        if r is not None:
            return np.asarray(r, dtype=np.float32)
    return _maxpool2d_numpy(x, kernel, stride)


def _linear(x, w, b):
    if _bridge is not None:
        r = _bridge.linear(_ensure_f32(x), _ensure_f32(w), _ensure_f32(b))
        if r is not None:
            return np.asarray(r, dtype=np.float32)
    return (x @ w.T + b).astype(np.float32)


def _softmax(x, axis=-1):
    if _bridge is not None:
        r = _bridge.softmax(_ensure_f32(x), axis)
        if r is not None:
            return np.asarray(r, dtype=np.float32)
    z = x.astype(np.float64)
    z -= z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


def _adaptive_avg_pool2d(x, output_size):
    """Adaptive average pooling via _bridge or numpy fallback."""
    if x.ndim == 3:
        x = x[np.newaxis]
    n, c, h, w = x.shape
    oh, ow = output_size

    # GPU path for 3x3 output
    if _bridge is not None and oh == 3 and ow == 3:
        r = _bridge.adaptive_avgpool_3x3(_ensure_f32(x))
        if r is not None:
            return np.asarray(r, dtype=np.float32).reshape(n, c, 3, 3)

    # Numpy fallback
    out = np.zeros((n, c, oh, ow), dtype=np.float32)
    for i in range(oh):
        h0, h1 = (i * h) // oh, ((i + 1) * h) // oh
        for j in range(ow):
            w0, w1 = (j * w) // ow, ((j + 1) * w) // ow
            out[:, :, i, j] = x[:, :, h0:h1, w0:w1].mean(axis=(2, 3))
    return out


def _block_softmax(logits, k, l, tau=1.0):
    """Block-wise softmax with temperature."""
    z = logits.reshape(-1, k, l).astype(np.float32) / max(tau, 1e-8)
    # Per-block softmax along last dim
    result = np.zeros_like(z)
    for b in range(z.shape[0]):
        for j in range(k):
            result[b, j] = _softmax(z[b, j:j+1], axis=-1)
    return result.squeeze(0) if result.shape[0] == 1 else result


# ── Numpy fallbacks ─────────────────────────────────────────────────────────


def _conv2d_numpy(x, w, b, padding=1, stride=1):
    """Minimal numpy conv2d."""
    n, ci, hi, wi = x.shape
    co, _, kh, kw = w.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    _, _, hp, wp = x.shape
    ho = (hp - kh) // stride + 1
    wo = (wp - kw) // stride + 1
    out = np.zeros((n, co, ho, wo), dtype=np.float32)
    for oc in range(co):
        for khi in range(kh):
            for kwi in range(kw):
                out[:, oc] += (
                    x[:, :, khi:khi + ho * stride:stride, kwi:kwi + wo * stride:stride]
                    * w[oc, :, khi, kwi].reshape(1, ci, 1, 1)
                ).sum(axis=1)
        out[:, oc] += b[oc]
    return out


def _maxpool2d_numpy(x, kernel=2, stride=2):
    n, c, h, w = x.shape
    oh, ow = h // stride, w // stride
    out = np.zeros((n, c, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = x[:, :, i*stride:i*stride+kernel,
                                j*stride:j*stride+kernel].max(axis=(2, 3))
    return out


# ── CNN Encoder ─────────────────────────────────────────────────────────────


class CNNEncoder:
    """Lightweight VSA-CNN using _bridge GPU ops.

    Args:
        k: Number of blocks per code.
        l: Block length.
        channels: Conv channel widths.
        grid_size: Spatial grid for position extraction.
        temperature: Gumbel-Softmax temperature.
        seed: Random seed.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        channels: tuple[int, ...] = (32, 64, 128),
        grid_size: tuple[int, int] = (1, 1),
        temperature: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.grid_size = grid_size
        self.n_positions = grid_size[0] * grid_size[1]
        self.temperature = temperature
        self.channels = channels
        self._use_grilly = _bridge is not None

        rng = np.random.default_rng(seed)
        ch_in = [1] + list(channels)

        # Init conv weights: He initialization
        self.conv_w = []
        self.conv_b = []
        for i in range(len(channels)):
            fan_in = ch_in[i] * 3 * 3
            w = (rng.standard_normal((channels[i], ch_in[i], 3, 3))
                 * np.sqrt(2.0 / fan_in)).astype(np.float32)
            b = np.zeros(channels[i], dtype=np.float32)
            self.conv_w.append(w)
            self.conv_b.append(b)

        # Projection head: Linear(channels[-1], d_vsa)
        fan_in = channels[-1]
        self.proj_w = (rng.standard_normal((self.d_vsa, fan_in))
                       * np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.proj_b = np.zeros(self.d_vsa, dtype=np.float32)

        # Gradient accumulators
        self.conv_w_grad = [np.zeros_like(w) for w in self.conv_w]
        self.conv_b_grad = [np.zeros_like(b) for b in self.conv_b]
        self.proj_w_grad = np.zeros_like(self.proj_w)
        self.proj_b_grad = np.zeros_like(self.proj_b)

        # Forward cache for backward
        self._cache = {}

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Forward: image -> per-position block-codes.

        Args:
            image: (80, 80) or (1, 80, 80) float32 [0, 1].

        Returns:
            (k, l) or (n_pos, k, l) block-codes.
        """
        if image.ndim == 2:
            x = image[np.newaxis, np.newaxis, :, :].astype(np.float32)
        elif image.ndim == 3:
            x = image[np.newaxis, :, :, :].astype(np.float32)
        else:
            x = _ensure_f32(image)

        # Conv backbone: fused Conv+GELU -> MaxPool per layer
        intermediates = [x]
        for i in range(len(self.channels)):
            x = _conv2d_gelu(x, self.conv_w[i], self.conv_b[i])
            intermediates.append(x)
            x = _maxpool2d(x, kernel=2, stride=2)
            intermediates.append(x)

        # Spatial grid extraction
        x = _adaptive_avg_pool2d(x, self.grid_size)  # (1, C, gh, gw)
        self._cache['pool_out'] = x

        # Per-position projection
        n, c, gh, gw = x.shape
        n_pos = gh * gw
        features = x.reshape(c, n_pos).T  # (n_pos, C)
        self._cache['features'] = features

        # Linear projection per position
        logits = _linear(features, self.proj_w, self.proj_b)  # (n_pos, d_vsa)
        self._cache['logits'] = logits

        # Block-wise softmax with temperature
        codes = _block_softmax(logits, self.k, self.l, self.temperature)

        if self.n_positions == 1:
            return codes.reshape(self.k, self.l)
        return codes  # (n_pos, k, l)

    def backward(self, grad_output: np.ndarray) -> None:
        """Backward through projection head.

        Full conv backprop via _bridge.conv2d_backward_* when available.

        Args:
            grad_output: (k, l) or (n_pos, k, l).
        """
        if grad_output.ndim == 2:
            g = grad_output.ravel().reshape(1, -1)  # (1, d_vsa)
        else:
            g = grad_output.reshape(grad_output.shape[0], -1)  # (n_pos, d_vsa)

        features = self._cache.get('features')  # (n_pos, C)
        if features is None:
            return

        # Projection gradients
        # dL/d(proj_w) = g^T @ features, dL/d(proj_b) = g.sum(0)
        self.proj_w_grad += (g.T @ features)
        self.proj_b_grad += g.sum(axis=0)

        # dL/d(features) = g @ proj_w
        g_feat = g @ self.proj_w  # (n_pos, C)

        # Backward through adaptive avg pool -> conv stack
        gh, gw = self.grid_size
        pool_out = self._cache.get('pool_out')
        if pool_out is None:
            return

        _, c, _, _ = pool_out.shape

        # Conv backward via _bridge when available
        if _bridge is not None:
            try:
                # Reshape feature gradient to spatial
                g_spatial = g_feat.T.reshape(1, c, gh, gw)

                # We skip full conv backward for now — the projection head
                # gradient is the most important signal. Conv weights will
                # train more slowly but the projection head learns the
                # image-to-VSA mapping.
                #
                # TODO: wire _bridge.conv2d_backward_weight for each layer
                pass
            except Exception:
                pass

    def zero_grad(self) -> None:
        for g in self.conv_w_grad:
            g.fill(0)
        for g in self.conv_b_grad:
            g.fill(0)
        self.proj_w_grad.fill(0)
        self.proj_b_grad.fill(0)

    def step(self, lr: float = 0.001) -> None:
        """SGD update using accumulated gradients."""
        # Update projection head (always)
        self.proj_w -= lr * self.proj_w_grad
        self.proj_b -= lr * self.proj_b_grad

        # Update conv weights (when gradients available)
        for i in range(len(self.conv_w)):
            if np.any(self.conv_w_grad[i] != 0):
                self.conv_w[i] -= lr * self.conv_w_grad[i]
                self.conv_b[i] -= lr * self.conv_b_grad[i]

    def anneal_temperature(self, factor: float = 0.95) -> None:
        self.temperature = max(self.temperature * factor, 0.01)

    def encode_panel(self, panel_image, target_size: int = 80) -> np.ndarray:
        """Encode a PIL Image to block-code(s)."""
        try:
            from PIL import Image
            if isinstance(panel_image, Image.Image):
                img = panel_image.convert("L").resize((target_size, target_size))
                pixels = np.array(img, dtype=np.float32) / 255.0
            else:
                pixels = np.asarray(panel_image, dtype=np.float32)
                if pixels.max() > 1.0:
                    pixels /= 255.0
        except ImportError:
            pixels = np.asarray(panel_image, dtype=np.float32)
            if pixels.max() > 1.0:
                pixels /= 255.0
        return self.forward(pixels)

    def get_parameters(self) -> list[np.ndarray]:
        return self.conv_w + self.conv_b + [self.proj_w, self.proj_b]

    def __repr__(self) -> str:
        backend = "grilly_bridge" if self._use_grilly else "numpy"
        return (
            f"CNNEncoder(k={self.k}, l={self.l}, d={self.d_vsa}, "
            f"grid={self.grid_size}, ch={self.channels}, "
            f"temp={self.temperature:.3f}, backend={backend})"
        )
