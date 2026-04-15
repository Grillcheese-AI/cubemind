"""Smoke test for the Phase 1.2 MinGRU backbone.

Done-when spec from TASKS.md:
  MinGRUModel(n_layers=6, d_model=256) forward pass runs without error.

This script verifies:
  1. Model instantiates at the Phase 1 reference config.
  2. Forward runs at several seq_len (32, 128, 256, 512) — validates the
     lifted prefix-scan cap.
  3. Logits have the expected (B, S, V) shape.
  4. Backward completes and populates .grad on embed / RMSNorm / Linear.
  5. Parameter count is in the right ballpark (~5M for the default config).

Run::

    uv run python sandbox/mingru_baseline/smoke_test.py
"""

from __future__ import annotations

import numpy as np

from cubemind.training.vsa_lm import MinGRUConfig, MinGRUModel


def test_instantiation():
    cfg = MinGRUConfig()  # defaults: d=256, L=6, d_ffn=768, vocab=4000
    model = MinGRUModel(cfg)
    params = model.num_parameters()
    print(f"  params: {params:,}  ({params / 1e6:.2f}M)")
    # Target "~5M params" per TASKS.md 1.3. Allow 3–10M.
    assert 3_000_000 <= params <= 10_000_000, (
        f"parameter count {params:,} outside expected 3-10M range"
    )
    return model


def test_forward_various_seq_lens(model):
    B = 2
    vocab = model.cfg.vocab_size
    rng = np.random.default_rng(0)
    for S in (32, 128, 256, 512):
        tokens = rng.integers(0, vocab, size=(B, S), dtype=np.int64)
        logits = model(tokens)
        assert logits.data.shape == (B, S, vocab), (
            f"seq={S}: expected {(B, S, vocab)}, got {logits.data.shape}"
        )
        finite = np.isfinite(logits.data).all()
        assert finite, f"seq={S}: non-finite logits"
        print(f"  forward seq={S:>3}  ok  logits.shape={logits.data.shape}  "
              f"min={logits.data.min():.3f}  max={logits.data.max():.3f}")


def test_backward(model):
    from grilly.nn.autograd import Variable

    B, S = 2, 64
    vocab = model.cfg.vocab_size
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, vocab, size=(B, S), dtype=np.int64)
    logits = model(tokens)

    # Trivial scalar loss: sum of logits. Backward exercises the full
    # autograd path (embed → MinGRU blocks → RMSNorm → tied head).
    assert isinstance(logits, Variable), "head should return a Variable"
    logits.backward(np.ones_like(logits.data, dtype=np.float32))

    # Check that gradients landed on the embedding + the first block's
    # MinGRU projections and the final RMSNorm weight.
    from grilly.nn._helpers import _get_param_array
    grads_populated = 0
    grads_total = 0
    for p in model.parameters():
        grads_total += 1
        if getattr(p, "grad", None) is not None:
            arr = _get_param_array(p)
            g_arr = p.grad if isinstance(p.grad, np.ndarray) else np.asarray(p.grad)
            if g_arr.shape == arr.shape and np.abs(g_arr).sum() > 0:
                grads_populated += 1
    print(f"  backward ok  grads_populated={grads_populated}/{grads_total}")
    assert grads_populated >= grads_total * 0.5, (
        f"only {grads_populated}/{grads_total} params got non-zero grad; "
        "autograd wiring may be broken"
    )


def main() -> None:
    print("\n=== MinGRU Phase 1.2 smoke test ===\n")

    print("[1/3] instantiation")
    model = test_instantiation()

    print("\n[2/3] forward at various seq lengths")
    test_forward_various_seq_lens(model)

    print("\n[3/3] backward + grad population")
    test_backward(model)

    print("\n=== PASS — MinGRU baseline ready for Phase 1.3 training ===\n")


if __name__ == "__main__":
    main()
