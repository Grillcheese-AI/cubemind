"""VSA-CNN Perception Frontend — Image to Spatial Block-Codes.

Lightweight CNN using grilly.nn modules (Conv2d, MaxPool2d, Linear) with
full forward/backward via the GEMM-based im2col conv2d path.

Architecture:
    Image (1, 80, 80)
    -> Conv(1,32,3,pad=1)+GELU+MaxPool(2)     -> (32, 40, 40)
    -> Conv(32,64,3,pad=1)+GELU+MaxPool(2)    -> (64, 20, 20)
    -> Conv(64,128,3,pad=1)+GELU+MaxPool(2)   -> (128, 10, 10)
    -> AdaptiveAvgPool(grid_h, grid_w)         -> (128, gh, gw)
    -> Per-position Linear(128, k*l)           -> (n_pos, k*l)
    -> BlockSoftmax(tau)                       -> (n_pos, k, l)

Uses grilly.nn.Conv2d GEMM path (im2col + gemm_mnk) for GPU forward
and conv2d_backward_input/weight for full gradient flow.
"""

from __future__ import annotations

import numpy as np

from cubemind.core import K_BLOCKS, L_BLOCK

# ── Import grilly nn modules ────────────────────────────────────────────────

_GRILLY_NN = False
try:
    from grilly.nn.conv import Conv2d as _Conv2d
    from grilly.nn.pooling import MaxPool2d as _MaxPool2d
    from grilly.nn.linear import Linear as _Linear
    _GRILLY_NN = True
except Exception:
    pass

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


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation via _bridge or numpy."""
    if _bridge is not None:
        r = _bridge.gelu(_ensure_f32(x))
        if r is not None:
            return np.asarray(r, dtype=np.float32)
    return (0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))).astype(np.float32)


def _gelu_backward(x: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """GELU backward via _bridge or numpy approximation."""
    if _bridge is not None:
        r = _bridge.gelu_backward(_ensure_f32(grad), _ensure_f32(x))
        if r is not None:
            return np.asarray(r, dtype=np.float32)
    # Approximate GELU derivative
    s = 1.0 / (1.0 + np.exp(-1.702 * x))
    return (grad * s * (1.0 + 1.702 * x * (1.0 - s))).astype(np.float32)


def _adaptive_avg_pool2d(x: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Adaptive average pooling to fixed spatial size."""
    if x.ndim == 3:
        x = x[np.newaxis]
    n, c, h, w = x.shape
    oh, ow = output_size
    out = np.zeros((n, c, oh, ow), dtype=np.float32)
    for i in range(oh):
        h0, h1 = (i * h) // oh, ((i + 1) * h) // oh
        for j in range(ow):
            w0, w1 = (j * w) // ow, ((j + 1) * w) // ow
            out[:, :, i, j] = x[:, :, h0:h1, w0:w1].mean(axis=(2, 3))
    return out


def _adaptive_avg_pool2d_backward(
    grad: np.ndarray, input_shape: tuple, output_size: tuple[int, int],
) -> np.ndarray:
    """Backward for adaptive avg pool: distribute gradient evenly."""
    n, c, h, w = input_shape
    oh, ow = output_size
    g_in = np.zeros(input_shape, dtype=np.float32)
    for i in range(oh):
        h0, h1 = (i * h) // oh, ((i + 1) * h) // oh
        for j in range(ow):
            w0, w1 = (j * w) // ow, ((j + 1) * w) // ow
            region_size = (h1 - h0) * (w1 - w0)
            g_in[:, :, h0:h1, w0:w1] += grad[:, :, i:i+1, j:j+1] / region_size
    return g_in


def _block_softmax(logits: np.ndarray, k: int, l: int, tau: float = 1.0) -> np.ndarray:
    """Block-wise softmax with temperature."""
    z = logits.reshape(-1, k, l).astype(np.float64) / max(tau, 1e-8)
    z -= z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z)
    result = (exp_z / exp_z.sum(axis=-1, keepdims=True)).astype(np.float32)
    return result.squeeze(0) if result.shape[0] == 1 else result


# ── Numpy conv2d fallback ───────────────────────────────────────────────────


def _conv2d_numpy(x, w, b, padding=1):
    """Minimal numpy conv2d for CPU fallback."""
    n, ci, hi, wi = x.shape
    co, _, kh, kw = w.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    _, _, hp, wp = x.shape
    ho = hp - kh + 1
    wo = wp - kw + 1
    out = np.zeros((n, co, ho, wo), dtype=np.float32)
    for oc in range(co):
        for khi in range(kh):
            for kwi in range(kw):
                out[:, oc] += (
                    x[:, :, khi:khi + ho, kwi:kwi + wo]
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
    """Lightweight VSA-CNN with full backward via grilly.nn modules.

    Uses grilly.nn.Conv2d (GEMM path with im2col) for GPU-accelerated
    forward and backward. Falls back to numpy for CPU-only environments.

    Args:
        k: Number of blocks per code.
        l: Block length.
        channels: Conv channel widths per layer.
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
        self._use_grilly = _GRILLY_NN

        ch_in = [1] + list(channels)

        if _GRILLY_NN:
            # GPU path: grilly nn modules with GEMM conv2d + backward
            self.convs = []
            self.pools = []
            for i in range(len(channels)):
                self.convs.append(_Conv2d(ch_in[i], channels[i],
                                          kernel_size=3, padding=1))
                self.pools.append(_MaxPool2d(kernel_size=2, stride=2))
            self.proj = _Linear(channels[-1], self.d_vsa)
        else:
            # CPU fallback: numpy weights
            rng = np.random.default_rng(seed)
            self.conv_w = []
            self.conv_b = []
            for i in range(len(channels)):
                fan_in = ch_in[i] * 9
                w = (rng.standard_normal((channels[i], ch_in[i], 3, 3))
                     * np.sqrt(2.0 / fan_in)).astype(np.float32)
                b = np.zeros(channels[i], dtype=np.float32)
                self.conv_w.append(w)
                self.conv_b.append(b)
            fan_in = channels[-1]
            self.proj_w = (rng.standard_normal((self.d_vsa, fan_in))
                           * np.sqrt(2.0 / fan_in)).astype(np.float32)
            self.proj_b = np.zeros(self.d_vsa, dtype=np.float32)

        # Forward cache
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

        if self._use_grilly:
            return self._forward_grilly(x)
        return self._forward_numpy(x)

    def _forward_grilly(self, x: np.ndarray) -> np.ndarray:
        """GPU forward using grilly.nn Conv2d (GEMM path)."""
        self._cache['conv_inputs'] = []
        self._cache['conv_outputs'] = []
        self._cache['gelu_inputs'] = []
        self._cache['gelu_outputs'] = []

        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            self._cache['conv_inputs'].append(x)
            conv_out = conv.forward(x)
            self._cache['conv_outputs'].append(conv_out)
            self._cache['gelu_inputs'].append(conv_out)
            gelu_out = _gelu(conv_out)
            self._cache['gelu_outputs'].append(gelu_out)
            x = pool.forward(gelu_out)

        # Adaptive avg pool
        self._cache['pre_pool_shape'] = x.shape
        x = _adaptive_avg_pool2d(x, self.grid_size)

        # Per-position linear projection
        n, c, gh, gw = x.shape
        n_pos = gh * gw
        features = x.reshape(c, n_pos).T  # (n_pos, C)
        self._cache['features'] = features

        logits = np.array([
            self.proj.forward(features[i:i + 1]).ravel()
            for i in range(n_pos)
        ])

        codes = _block_softmax(logits, self.k, self.l, self.temperature)
        if self.n_positions == 1:
            return codes.reshape(self.k, self.l)
        return codes

    def _forward_numpy(self, x: np.ndarray) -> np.ndarray:
        """CPU fallback forward."""
        for i in range(len(self.channels)):
            x = _gelu(_conv2d_numpy(x, self.conv_w[i], self.conv_b[i]))
            x = _maxpool2d_numpy(x)

        x = _adaptive_avg_pool2d(x, self.grid_size)
        n, c, gh, gw = x.shape
        n_pos = gh * gw
        features = x.reshape(c, n_pos).T
        self._cache['features'] = features

        logits = (features @ self.proj_w.T + self.proj_b)
        codes = _block_softmax(logits, self.k, self.l, self.temperature)
        if self.n_positions == 1:
            return codes.reshape(self.k, self.l)
        return codes

    # ── Backward ────────────────────────────────────────────────────────

    def backward(self, grad_output: np.ndarray) -> None:
        """Full backward through conv stack using grilly.nn.Conv2d.backward().

        Args:
            grad_output: (k, l) or (n_pos, k, l) gradient w.r.t. block-codes.
        """
        if grad_output.ndim == 2:
            g = grad_output.ravel().reshape(1, -1)
        else:
            g = grad_output.reshape(grad_output.shape[0], -1)

        features = self._cache.get('features')
        if features is None:
            return

        # ── Backward through linear projection ──
        n_pos, c = features.shape
        g_features = np.zeros_like(features)

        if self._use_grilly:
            for i in range(n_pos):
                g_features[i] = self.proj.backward(
                    g[i:i + 1], x=features[i:i + 1]
                ).ravel()
        else:
            # Numpy: manual proj gradients
            self._proj_w_grad = g.T @ features
            self._proj_b_grad = g.sum(axis=0)
            g_features = g @ self.proj_w

        if not self._use_grilly:
            return  # Numpy path: no conv backward

        # ── Backward through adaptive avg pool ──
        gh, gw = self.grid_size
        g_spatial = g_features.T.reshape(1, c, gh, gw)
        pre_pool_shape = self._cache.get('pre_pool_shape')
        if pre_pool_shape is None:
            return
        g = _adaptive_avg_pool2d_backward(g_spatial, pre_pool_shape, self.grid_size)

        # ── Backward through conv stack (reverse order) ──
        for i in range(len(self.convs) - 1, -1, -1):
            # Backward through MaxPool2d
            g = self.pools[i].backward(g)

            # Backward through GELU
            gelu_input = self._cache['gelu_inputs'][i]
            g = _gelu_backward(gelu_input, g)

            # Backward through Conv2d (computes weight/bias grads internally)
            g = self.convs[i].backward(g)

    # ── Optimizer interface ─────────────────────────────────────────────

    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        if self._use_grilly:
            for conv in self.convs:
                conv.zero_grad()
            self.proj.zero_grad()

    def step(self, lr: float = 0.001) -> None:
        """SGD update using accumulated gradients (in-place)."""
        if self._use_grilly:
            for conv in self.convs:
                self._sgd_update_param(conv, 'weight', '_grad_w', lr)
                self._sgd_update_param(conv, 'bias', '_grad_b', lr)
            self._sgd_update_param(self.proj, 'weight', '_grad_w', lr)
            self._sgd_update_param(self.proj, 'bias', '_grad_b', lr)
        else:
            if hasattr(self, '_proj_w_grad'):
                self.proj_w -= lr * self._proj_w_grad
                self.proj_b -= lr * self._proj_b_grad

    @staticmethod
    def _sgd_update_param(module, param_name: str, grad_name: str, lr: float):
        """Update a parameter in-place, preserving the original object type."""
        param = getattr(module, param_name, None)
        if param is None:
            return
        # Try .grad attribute first (Parameter objects)
        grad = getattr(param, 'grad', None)
        if grad is None:
            # Fall back to module-level gradient cache
            grad = getattr(module, grad_name, None)
        if grad is None:
            return
        # In-place update to preserve the object (and its .grad attribute)
        param_arr = np.asarray(param)
        grad_arr = np.asarray(grad)
        param_arr -= lr * grad_arr

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

    def get_parameters(self) -> list:
        if self._use_grilly:
            params = []
            for conv in self.convs:
                params.append(conv.weight)
                if conv.bias is not None:
                    params.append(conv.bias)
            params.append(self.proj.weight)
            if self.proj.bias is not None:
                params.append(self.proj.bias)
            return params
        return [self.proj_w, self.proj_b]

    def __repr__(self) -> str:
        backend = "grilly" if self._use_grilly else "numpy"
        return (
            f"CNNEncoder(k={self.k}, l={self.l}, d={self.d_vsa}, "
            f"grid={self.grid_size}, ch={self.channels}, "
            f"temp={self.temperature:.3f}, backend={backend})"
        )
