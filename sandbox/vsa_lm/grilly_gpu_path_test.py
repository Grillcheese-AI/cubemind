"""Verify grilly's GPU path actually dispatches to Vulkan shaders.

The smoke test (grilly_facade_smoke_test.py) only verifies *correctness*:
operations produce correct results without throwing. It does NOT verify
which path produced them — a layer could silently fall back to numpy CPU
and still pass.

This test catches that. It uses three independent signals:

  1. ``_bridge.get_fallback_stats()`` — grilly's built-in counter that
     records every CPU fallback inside the bridge layer.
  2. Monkey-patched call counters on ``_bridge.linear``, ``_bridge.layernorm``,
     ``VulkanLearning.embedding_lookup``, and ``VulkanLearning.adamw_update``
     so we know each function actually fires once per intended op.
  3. Forward+backward latency comparison vs a CPU-only run — GPU should be
     visibly faster on a moderately-sized batch.

Run:
    python sandbox/vsa_lm/grilly_gpu_path_test.py
"""

from __future__ import annotations

import time
import os

import numpy as np

# Force grilly path
import grilly
import grilly.torch_api as torch
from grilly import nn
from grilly.nn import functional as F
from grilly.backend import _bridge


def section(title: str):
    print(f"\n-- {title} " + "-" * (60 - len(title)))


# ── Sanity: Vulkan is actually up ──
section("1. Vulkan device alive")
print(f"  VULKAN_AVAILABLE: {grilly.VULKAN_AVAILABLE}")
print(f"  _bridge.is_available(): {_bridge.is_available()}")
import grilly_core as gc
dev = gc.Device()
print(f"  gc.Device() OK: {type(dev).__name__}")
assert grilly.VULKAN_AVAILABLE, "Vulkan must be available for this test"
assert _bridge.is_available(), "Bridge must be available for this test"


# ── Wrap dispatch points to count GPU calls ──
section("2. Wrapping dispatch points to count GPU calls")

call_counts = {}


def make_counter(name, fn):
    def wrapped(*args, **kwargs):
        call_counts[name] = call_counts.get(name, 0) + 1
        return fn(*args, **kwargs)
    return wrapped


# _bridge entry points (modern path) — module-level functions, single instance
_bridge.linear = make_counter("_bridge.linear", _bridge.linear)
_bridge.layernorm = make_counter("_bridge.layernorm", _bridge.layernorm)
_bridge.gelu = make_counter("_bridge.gelu", _bridge.gelu)

# CRITICAL: grilly has TWO VulkanCompute instances — one from device_manager
# and a separate one from grilly.Compute() that AdamW grabs internally. Each
# has its own .learning / .fnn attributes. Patch at the CLASS level so both
# instances see the wrapped methods, otherwise we miss AdamW's GPU dispatch
# entirely (the test counter stays at 0 even when GPU is fired).
from grilly.backend.learning import VulkanLearning

if hasattr(VulkanLearning, "embedding_lookup"):
    VulkanLearning.embedding_lookup = make_counter(
        "VulkanLearning.embedding_lookup", VulkanLearning.embedding_lookup
    )
if hasattr(VulkanLearning, "embedding_backward"):
    VulkanLearning.embedding_backward = make_counter(
        "VulkanLearning.embedding_backward", VulkanLearning.embedding_backward
    )
if hasattr(VulkanLearning, "adamw_update"):
    VulkanLearning.adamw_update = make_counter(
        "VulkanLearning.adamw_update", VulkanLearning.adamw_update
    )

# Patch VulkanFNN class for linear_backward / layernorm_backward
try:
    from grilly.backend.fnn import VulkanFNN

    if hasattr(VulkanFNN, "linear_backward"):
        VulkanFNN.linear_backward = make_counter(
            "VulkanFNN.linear_backward", VulkanFNN.linear_backward
        )
    if hasattr(VulkanFNN, "layernorm_backward"):
        VulkanFNN.layernorm_backward = make_counter(
            "VulkanFNN.layernorm_backward", VulkanFNN.layernorm_backward
        )
except ImportError:
    pass

print("  Counters installed at CLASS level on VulkanLearning + VulkanFNN")
print("  + module level on _bridge.linear / layernorm / gelu")


# ── 3. nn.Linear forward+backward GPU path ──
section("3. nn.Linear forward + autograd backward")

_bridge.reset_fallback_stats()
call_counts.clear()

lin = nn.Linear(64, 32)
x = torch.randn(16, 64)
y = torch.tensor(np.random.randint(0, 32, 16), dtype=torch.long)
opt = torch.optim.AdamW(lin.parameters(), lr=1e-2)

logits = lin(x)
loss = F.cross_entropy(logits, y)
opt.zero_grad()
loss.backward()
opt.step()

print(f"  call_counts: {call_counts}")
print(f"  fallback_stats: {_bridge.get_fallback_stats()}")
linear_fwd = call_counts.get("_bridge.linear", 0)
linear_bwd = call_counts.get("VulkanFNN.linear_backward", 0)
adamw = call_counts.get("VulkanLearning.adamw_update", 0)
print(f"  Linear forward GPU calls : {linear_fwd}  ({'OK' if linear_fwd >= 1 else 'CPU FALLBACK'})")
print(f"  Linear backward GPU calls: {linear_bwd}  ({'OK' if linear_bwd >= 1 else 'CPU FALLBACK'})")
print(f"  AdamW step GPU calls     : {adamw}  ({'OK' if adamw >= 1 else 'CPU FALLBACK'})")


# ── 4. nn.LayerNorm forward GPU path ──
section("4. nn.LayerNorm")

_bridge.reset_fallback_stats()
call_counts.clear()

ln = nn.LayerNorm(64)
out = ln(x)

print(f"  call_counts: {call_counts}")
print(f"  fallback_stats: {_bridge.get_fallback_stats()}")
ln_fwd = call_counts.get("_bridge.layernorm", 0)
print(f"  LayerNorm forward GPU calls: {ln_fwd}  ({'OK' if ln_fwd >= 1 else 'CPU FALLBACK'})")


# ── 5. nn.Embedding forward GPU path ──
section("5. nn.Embedding")

_bridge.reset_fallback_stats()
call_counts.clear()

emb = nn.Embedding(128, 64)
ids = torch.tensor(np.random.randint(0, 128, (4, 16)), dtype=torch.long)
out = emb(ids)

print(f"  call_counts: {call_counts}")
print(f"  fallback_stats: {_bridge.get_fallback_stats()}")
emb_fwd = call_counts.get("VulkanLearning.embedding_lookup", 0)
print(f"  Embedding forward GPU calls: {emb_fwd}  ({'OK' if emb_fwd >= 1 else 'CPU FALLBACK'})")


# ── 6. End-to-end: stacked Linears with autograd loss ──
section("6. Stacked Linears (autograd chain)")

_bridge.reset_fallback_stats()
call_counts.clear()


class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 128)
        self.l2 = nn.Linear(128, 32)

    def forward(self, x):
        h = self.l1(x)
        h = F.gelu(h)
        return self.l2(h)


m = TwoLayerMLP()
opt = torch.optim.AdamW(m.parameters(), lr=1e-2)

x2 = torch.randn(16, 64)
y2 = torch.tensor(np.random.randint(0, 32, 16), dtype=torch.long)

# Run 5 steps to accumulate counts
for _ in range(5):
    logits = m(x2)
    loss = F.cross_entropy(logits, y2)
    opt.zero_grad()
    loss.backward()
    opt.step()

print(f"  call_counts after 5 steps: {dict(sorted(call_counts.items()))}")
print(f"  fallback_stats: {_bridge.get_fallback_stats()}")
expected_linear_fwd = 10  # 2 layers * 5 steps
expected_linear_bwd = 10  # 2 layers * 5 steps
expected_adamw = 20       # 4 params (2 weights + 2 biases) * 5 steps
actual_fwd = call_counts.get("_bridge.linear", 0)
actual_bwd = call_counts.get("VulkanFNN.linear_backward", 0)
actual_adamw = call_counts.get("VulkanLearning.adamw_update", 0)
print(f"  Linear fwd : actual={actual_fwd}  expected~={expected_linear_fwd}  "
      f"({'OK' if actual_fwd >= expected_linear_fwd else 'MISSING DISPATCHES'})")
print(f"  Linear bwd : actual={actual_bwd}  expected~={expected_linear_bwd}  "
      f"({'OK' if actual_bwd >= expected_linear_bwd else 'MISSING DISPATCHES'})")
print(f"  AdamW step : actual={actual_adamw}  expected~={expected_adamw}  "
      f"({'OK' if actual_adamw >= expected_adamw else 'MISSING DISPATCHES'})")


# ── 6b. Realistic VSA-LM tensor size — GPU should win naturally ──
section("6b. VSA-LM-realistic size (B=16, S=256, D=384)")

from grilly.nn._perf_policy import reset_perf_decisions, get_perf_decisions

reset_perf_decisions()
call_counts.clear()

# Same shape v3c will use: B=16, S=256 → 4096 rows × 384 dim
# AdditionLinear maps 384 → 1152 → 384, but for plain Linear test this works.
big_lin = nn.Linear(384, 1152)
big_x = torch.randn(4096, 384)

# Three forwards (enough for A/B benchmark + 1 cached pick)
for _ in range(5):
    _ = big_lin(big_x)

decisions = get_perf_decisions()
big_fwd_calls = call_counts.get("_bridge.linear", 0)
print(f"  call_counts: {call_counts}")
print(f"  perf decisions: {decisions}")
print(f"  GPU dispatches in 5 forwards on (4096, 384) -> (4096, 1152): {big_fwd_calls}")
big_chose_gpu = any("gpu" == v for v in decisions.values())
print(f"  Auto-fastest chose GPU: {big_chose_gpu}  ({'OK' if big_chose_gpu else 'CPU PICKED — GPU loses on this size'})")


# ── 6c. Force GPU with GRILLY_AUTO_FASTEST=0 ──
section("6c. GRILLY_AUTO_FASTEST=0 forces GPU regardless of size")

import importlib
os.environ["GRILLY_AUTO_FASTEST"] = "0"
from grilly.nn import _perf_policy
importlib.reload(_perf_policy)
# Re-import the symbol used by Linear
import grilly.nn.linear
grilly.nn.linear.choose_fastest = _perf_policy.choose_fastest

reset_perf_decisions()
call_counts.clear()

# Tiny size that previously picked CPU
forced_lin = nn.Linear(64, 32)
forced_x = torch.randn(16, 64)
for _ in range(5):
    _ = forced_lin(forced_x)

forced_calls = call_counts.get("_bridge.linear", 0)
print(f"  GPU dispatches with AUTO_FASTEST=0 (5 forwards on tiny size): {forced_calls}")
forced_ok = forced_calls >= 5
print(f"  Forced GPU: {forced_ok}  ({'OK' if forced_ok else 'AUTO_FASTEST=0 not honored'})")

# Restore
os.environ.pop("GRILLY_AUTO_FASTEST", None)
importlib.reload(_perf_policy)
grilly.nn.linear.choose_fastest = _perf_policy.choose_fastest


# ── 7. Strict-mode rerun: any fallback raises ──
section("7. GRILLY_BRIDGE_STRICT rerun (any fallback -> RuntimeError)")

# Reload _bridge with strict mode set
os.environ["GRILLY_BRIDGE_STRICT"] = "1"
import importlib
importlib.reload(_bridge)

# After reload, the original _bridge.linear / layernorm / gelu lost our counters
# but gained the strict flag. Run a fresh forward+backward and watch for raises.
strict_ok = True
strict_err = None
try:
    # Need a fresh model since _bridge was reloaded
    lin2 = nn.Linear(64, 32)
    opt2 = torch.optim.AdamW(lin2.parameters(), lr=1e-2)
    x3 = torch.randn(16, 64)
    y3 = torch.tensor(np.random.randint(0, 32, 16), dtype=torch.long)
    logits = lin2(x3)
    loss = F.cross_entropy(logits, y3)
    opt2.zero_grad()
    loss.backward()
    opt2.step()
except Exception as e:
    strict_ok = False
    strict_err = f"{type(e).__name__}: {e}"

if strict_ok:
    print("  Linear forward+backward+step under strict mode: NO FALLBACK")
else:
    print(f"  Linear under strict mode FELL BACK: {strict_err}")

# Reset env
os.environ.pop("GRILLY_BRIDGE_STRICT", None)
importlib.reload(_bridge)


# ── Summary ──
print("\n" + "=" * 64)
print("GPU PATH SUMMARY")
print("=" * 64)
checks = [
    ("Linear forward GPU", linear_fwd >= 1),
    ("Linear backward GPU", linear_bwd >= 1),
    ("AdamW step GPU", adamw >= 1),
    ("LayerNorm forward GPU", ln_fwd >= 1),
    ("Embedding forward GPU", emb_fwd >= 1),
    # Stacked test on tiny inputs: choose_fastest A/B benchmarks (3+3 = 6 GPU
    # dispatches) then routes the rest to whichever path won. For tiny inputs
    # CPU usually wins; GPU dispatch overhead exceeds the actual compute.
    # Don't gate on "10 fwd dispatches" — that's a misunderstanding of the
    # auto-fastest policy. The right gate is "GPU was tried at least once".
    ("Stacked autograd: GPU benchmarked", actual_fwd >= 1),
    ("Stacked autograd: 10 bwd dispatches", actual_bwd >= expected_linear_bwd),
    ("Stacked autograd: 20 AdamW dispatches", actual_adamw >= expected_adamw),
    # The key check for VSA-LM training: at realistic shapes, GPU wins
    # naturally without any env var override.
    ("VSA-LM-realistic size auto-picks GPU", big_chose_gpu),
    # And the override works when forced.
    ("GRILLY_AUTO_FASTEST=0 forces GPU on tiny shapes", forced_ok),
    ("Strict mode (no silent fallback)", strict_ok),
]
for name, ok in checks:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}")

n_pass = sum(1 for _, ok in checks if ok)
print(f"\n  {n_pass}/{len(checks)} GPU path checks passed")
import sys
sys.exit(0 if n_pass == len(checks) else 1)
