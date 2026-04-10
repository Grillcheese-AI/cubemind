"""Grilly torch_api facade smoke test for VSA-LM workload.

Exercises every API the v3c training loop touches with a tiny model so we
find every gap in seconds, not after a 4-hour training run.

Each test wraps a single API in try/except and reports PASS / FAIL / GAP.
The output is a checklist of what works against the real workload.

Usage:
    python sandbox/vsa_lm/grilly_facade_smoke_test.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np

# Force grilly-only path: do NOT touch real torch
import grilly.torch_api as torch
from grilly import nn
from grilly.nn import functional as F


# ── Test harness ──
class Result:
    PASS = "PASS"
    FAIL = "FAIL"
    GAP = "GAP "  # API doesn't exist at all


_results: list[tuple[str, str, str, str]] = []


def _safe(s: str) -> str:
    """Strip non-ASCII so Windows cp1252 console can print without crashing."""
    return s.encode("ascii", "replace").decode("ascii")


def check(category: str, name: str, fn):
    try:
        fn()
        _results.append((category, name, Result.PASS, ""))
        print(f"  [{Result.PASS}] {name}")
    except (AttributeError, ImportError, NotImplementedError) as e:
        msg = _safe(str(e))
        _results.append((category, name, Result.GAP, msg))
        print(f"  [{Result.GAP}] {name} -- {type(e).__name__}: {msg}")
    except Exception as e:
        msg = _safe(str(e))
        _results.append((category, name, Result.FAIL, f"{type(e).__name__}: {msg}"))
        print(f"  [{Result.FAIL}] {name} -- {type(e).__name__}: {msg}")


def section(title: str):
    print(f"\n-- {title} " + "-" * (60 - len(title)))


# ── 1. Imports + device ──
section("1. Imports & device")

check("imports", "import grilly.torch_api as torch", lambda: torch)
check("imports", "from grilly import nn", lambda: nn)
check("imports", "from grilly.nn import functional as F", lambda: F)
check("imports", "torch.vulkan.is_available()", lambda: torch.vulkan.is_available())
check("imports", "torch.device('vulkan')", lambda: torch.device("vulkan"))
check("imports", "torch.device('cpu')", lambda: torch.device("cpu"))

# ── 2. Tensor factories ──
section("2. Tensor factories")

check("factories", "torch.empty(4, 8)", lambda: torch.empty(4, 8))
check("factories", "torch.zeros(4, 8)", lambda: torch.zeros(4, 8))
check("factories", "torch.ones(4, 8)", lambda: torch.ones(4, 8))
check("factories", "torch.randn(4, 8)", lambda: torch.randn(4, 8))
check("factories", "torch.tensor([1,2,3], dtype=torch.long)",
      lambda: torch.tensor([1, 2, 3], dtype=torch.long))
check("factories", "torch.randperm(16)", lambda: torch.randperm(16))
check("factories", ".uniform_(-0.1, 0.1)",
      lambda: torch.empty(4, 8).uniform_(-0.1, 0.1))
check("factories", ".zero_()", lambda: torch.zeros(4, 8).zero_())

# ── 3. Tensor methods (the workload uses these heavily) ──
section("3. Tensor methods")


def _make_t():
    return torch.randn(2, 4, 8)


check("tensor", ".shape", lambda: _make_t().shape)
check("tensor", ".ndim", lambda: _make_t().ndim)
check("tensor", ".reshape(-1, 8)", lambda: _make_t().reshape(-1, 8))
check("tensor", ".unsqueeze(0)", lambda: _make_t().unsqueeze(0))
check("tensor", ".squeeze(0)", lambda: torch.zeros(1, 4, 8).squeeze(0))
check("tensor", ".mean(dim=1)", lambda: _make_t().mean(dim=1))
check("tensor", ".mean(dim=(0, 1))", lambda: _make_t().mean(dim=(0, 1)))
check("tensor", ".item() on scalar", lambda: torch.zeros(1).mean().item())
check("tensor", ".detach()", lambda: _make_t().detach())
check("tensor", ".numel()", lambda: _make_t().numel())
check("tensor", ".to(device)", lambda: _make_t().to("cpu"))
check("tensor", "x @ y (matmul)",
      lambda: torch.randn(4, 8) @ torch.randn(8, 4))

# ── 4. nn.init ──
section("4. nn.init")

check("init", "nn.init.normal_(tensor, 0, 0.02)",
      lambda: nn.init.normal_(torch.empty(4, 8), 0.0, 0.02))
check("init", "nn.init.uniform_(tensor, -0.1, 0.1)",
      lambda: nn.init.uniform_(torch.empty(4, 8), -0.1, 0.1))

# ── 5. Module / Parameter / register_buffer / ModuleList ──
section("5. Module infrastructure")


def _module_basics():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.empty(4, 8).uniform_(-0.1, 0.1))
            self.register_buffer("h", torch.zeros(8))

    m = M()
    assert "w" in dict(m.named_parameters()) or hasattr(m, "w")
    return m


check("module", "nn.Module subclass + Parameter + register_buffer", _module_basics)
check("module", "nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])",
      lambda: nn.ModuleList([nn.Linear(8, 8) for _ in range(2)]))


def _state_dict_roundtrip():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 4)
            self.register_buffer("h", torch.zeros(4))

    m1 = M()
    sd = m1.state_dict()
    m2 = M()
    m2.load_state_dict(sd)


check("module", "state_dict() / load_state_dict() roundtrip", _state_dict_roundtrip)

# ── 6. Layer modules ──
section("6. Layer modules")

check("layer", "nn.Linear(8, 4)", lambda: nn.Linear(8, 4)(torch.randn(2, 8)))
check("layer", "nn.LayerNorm(8)", lambda: nn.LayerNorm(8)(torch.randn(2, 8)))
check("layer", "nn.Embedding(16, 8)",
      lambda: nn.Embedding(16, 8)(torch.tensor([0, 1, 2, 3], dtype=torch.long)))

# ── 7. Math ops the workload uses ──
section("7. Math ops")

check("math", "torch.cdist(x1, x2, p=1)",
      lambda: torch.cdist(torch.randn(1, 4, 8), torch.randn(1, 6, 8), p=1))
check("math", "torch.tanh(x)", lambda: torch.tanh(torch.randn(4, 8)))
check("math", "torch.sign(x)", lambda: torch.sign(torch.randn(4, 8)))
check("math", "torch.clamp(x, max=2.0)",
      lambda: torch.clamp(torch.randn(4, 8), max_val=2.0))
check("math", "F.softplus(x)", lambda: F.softplus(torch.randn(4, 8)))
check("math", "F.gelu(x)", lambda: F.gelu(torch.randn(4, 8)))
check("math", "F.cross_entropy(logits, target, reduction='mean')",
      lambda: F.cross_entropy(
          torch.randn(8, 16),
          torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long),
          reduction="mean",
      ))
check("math", "F.cross_entropy(..., reduction='sum')",
      lambda: F.cross_entropy(
          torch.randn(8, 16),
          torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long),
          reduction="sum",
      ))

# ── 8. no_grad context ──
section("8. no_grad")


def _run_no_grad():
    with torch.no_grad():
        _ = torch.randn(4, 8) + torch.randn(4, 8)


check("ctx", "no_grad() works as a context manager", _run_no_grad)

# ── 9. AMP ──
section("9. AMP (autocast / GradScaler)")


def _run_autocast():
    with torch.amp.autocast("vulkan", enabled=True, dtype=torch.float16):
        _ = torch.randn(4, 8) @ torch.randn(8, 4)


check("amp", "torch.amp.autocast('vulkan', dtype=torch.float16)", _run_autocast)
check("amp", "torch.amp.GradScaler('vulkan', enabled=True)",
      lambda: torch.amp.GradScaler("vulkan", enabled=True))

# ── 10. Optimizer ──
section("10. Optimizer + scheduler")


def _adamw_step():
    lin = nn.Linear(8, 4)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=0.01)
    x = torch.randn(2, 8)
    y_target = torch.tensor([0, 1], dtype=torch.long)
    logits = lin(x)
    loss = F.cross_entropy(logits, y_target)
    opt.zero_grad()
    loss.backward()
    opt.step()


check("optim", "AdamW.zero_grad / backward / step", _adamw_step)


def _scheduler():
    lin = nn.Linear(8, 4)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
    sch.step()
    _ = sch.get_last_lr()


check("optim", "CosineAnnealingLR + get_last_lr()", _scheduler)

# ── 11. clip_grad_norm_ ──
section("11. nn.utils.clip_grad_norm_")


def _clip_grad():
    lin = nn.Linear(8, 4)
    x = torch.randn(2, 8)
    y_target = torch.tensor([0, 1], dtype=torch.long)
    loss = F.cross_entropy(lin(x), y_target)
    loss.backward()
    nn.utils.clip_grad_norm_(lin.parameters(), 1.0)


check("clip", "nn.utils.clip_grad_norm_(params, 1.0)", _clip_grad)

# ── 12. Save/load (.grl) ──
section("12. Checkpoint save/load (.grl)")


def _save_load():
    lin = nn.Linear(8, 4)
    with tempfile.NamedTemporaryFile(suffix=".grl", delete=False) as fh:
        path = fh.name
    try:
        torch.save({"model": lin.state_dict(), "step": 42, "best_ppl": 1.23}, path)
        ck = torch.load(path)
        assert "model" in ck
        assert ck.get("step") == 42
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


check("ckpt", "torch.save / torch.load (.grl) roundtrip with metadata", _save_load)

# ── 13. End-to-end tiny VSA-LM model ──
section("13. End-to-end tiny VSA-LM forward+backward")


class TinyAdditionLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in).uniform_(-0.1, 0.1))
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.d_in = d_in

    def forward(self, x):
        orig = x.shape
        x_flat = x.reshape(-1, orig[-1])
        dist = torch.cdist(x_flat.unsqueeze(0), self.weight.unsqueeze(0), p=1).squeeze(0)
        out = -dist + self.bias
        return out.reshape(*orig[:-1], -1)


class TinyVSALayer(nn.Module):
    def __init__(self, d, d_ffn):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.ffn_up = TinyAdditionLinear(d, d_ffn)
        self.ffn_down = TinyAdditionLinear(d_ffn, d)

    def forward(self, x):
        h = self.ln(x)
        h_up = self.ffn_up(h) / math.sqrt(self.ffn_up.d_in)
        h_up = F.gelu(h_up)
        h_ffn = self.ffn_down(h_up) / math.sqrt(self.ffn_down.d_in)
        return x + h_ffn


class TinyVSALM(nn.Module):
    def __init__(self, vocab=16, d=8, d_ffn=16, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.out = nn.Linear(d, vocab, bias=False)
        self.layers = nn.ModuleList([TinyVSALayer(d, d_ffn) for _ in range(n_layers)])

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


def _e2e_step():
    model = TinyVSALM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    logits = model(ids)  # (B, S, V)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


check("e2e", "TinyVSALM forward+backward+step (no exception)", _e2e_step)


# ── 14. Effect verification (catches false positives from "no exception" tests) ──
section("14. Gradient connectivity & state_dict effect verification")


def _adamw_actually_updates_weights():
    """AdamW.step() must actually move the weight, not just run without raising.

    Without gradient connectivity from loss.backward() back to nn.Linear's
    weight, this is a silent training failure: every call passes but no
    learning happens.
    """
    lin = nn.Linear(8, 4)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2)  # large lr so any update is visible
    x = torch.randn(4, 8)
    y_target = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    w_before = np.array(lin.weight, copy=True)
    logits = lin(x)
    loss = F.cross_entropy(logits, y_target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    w_after = np.array(lin.weight, copy=True)
    delta = float(np.abs(w_after - w_before).max())
    if delta == 0.0:
        raise AssertionError(
            f"AdamW.step() did not change nn.Linear.weight (max delta = {delta:.3e}). "
            "Gradient connectivity from autograd loss to Linear.weight is broken."
        )


check("effect", "AdamW.step() actually updates nn.Linear.weight", _adamw_actually_updates_weights)


def _state_dict_preserves_linear_weight():
    """state_dict() must include the 'weight' key for nn.Linear."""
    lin = nn.Linear(8, 4)
    sd = lin.state_dict()
    if "weight" not in sd:
        raise AssertionError(
            f"nn.Linear.state_dict() missing 'weight' key. Got keys: {list(sd.keys())}"
        )
    if sd["weight"].shape != (4, 8):
        raise AssertionError(
            f"state_dict['weight'].shape == {sd['weight'].shape}, expected (4, 8)"
        )


check("effect", "nn.Linear.state_dict() contains 'weight'", _state_dict_preserves_linear_weight)


def _grl_roundtrip_preserves_linear_weight():
    """torch.save → torch.load (.grl) must preserve real nn.Linear weights."""
    lin = nn.Linear(8, 4)
    w_orig = np.array(lin.weight, copy=True)
    with tempfile.NamedTemporaryFile(suffix=".grl", delete=False) as fh:
        path = fh.name
    try:
        torch.save({"model": lin.state_dict()}, path)
        ck = torch.load(path)
        if "model" not in ck or "weight" not in ck["model"]:
            raise AssertionError(
                f"Loaded checkpoint missing keys. ck keys: {list(ck.keys())}, "
                f"model keys: {list(ck.get('model', {}).keys())}"
            )
        loaded = np.asarray(ck["model"]["weight"])
        err = float(np.abs(loaded - w_orig).max())
        if err > 1e-6:
            raise AssertionError(f"Weight roundtrip max abs err = {err:.3e}")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


check("effect", "torch.save/load .grl preserves nn.Linear.weight bytes",
      _grl_roundtrip_preserves_linear_weight)

# ── Summary ──
print("\n" + "=" * 64)
print("SUMMARY")
print("=" * 64)

by_status = {Result.PASS: 0, Result.FAIL: 0, Result.GAP: 0}
for _, _, status, _ in _results:
    by_status[status] += 1

print(f"  PASS: {by_status[Result.PASS]}")
print(f"  FAIL: {by_status[Result.FAIL]}  (API exists but throws)")
print(f"  GAP : {by_status[Result.GAP]}  (API missing entirely)")
print(f"  TOTAL: {len(_results)}")

if by_status[Result.FAIL] or by_status[Result.GAP]:
    print("\nGaps and failures (action items for grilly):")
    for cat, name, status, msg in _results:
        if status != Result.PASS:
            print(f"  [{status}] {cat}/{name}")
            if msg:
                print(f"         -> {msg[:160]}")
    sys.exit(1)
else:
    print("\nAll v3c-required APIs work. Ready to port vsa_lm_v3c to grilly.")
    sys.exit(0)
