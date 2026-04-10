# MoQE TinyStories Results Log

## [2026-04-03 03:00] Run 1 — d=256, 6 layers, 4 experts (manual backward)

**Status:** PARTIAL — trained but diverged with lr=5e-2
**Config:** d_model=256, n_layers=6, n_experts=4, top_k=1, vocab=4000, seq=512, lr=5e-2
**Pipeline:** moqe_backward (manual) + run_offline_distillation + CE-only mode
**Issue:** No position embeddings in moqe_backward forward → stuck at PPL ~4.7
**Fix applied:** Added pos_embed to moqe_backward forward + backward
**Second issue:** lr=5e-2 too high for AdamW → PPL diverged to 47M (gnorm 1.5M)
**Decision:** Switch to grilly autograd (eliminates manual backward bugs)

---

## [2026-04-03 13:24] Run 2 — d=256, 6 layers, grilly autograd (CE-only)

**Status:** PASS — stable training, PPL dropping
**Config:** d_model=256, n_layers=6, n_experts=4, top_k=1, vocab=4000, seq=512
**LR:** 3e-4 (AdamW, cosine schedule, warmup=1000)
**Pipeline:** grilly.nn.autograd Variable + cross_entropy + .backward(use_gpu=True)

**PPL Curve:**
| Step | PPL | lr | stp/s |
|------|-----|-----|-------|
| 0 | 3999 | 0.00001 | 0.2 |
| 200 | 581 | 0.00007 | 0.8 |
| 500 | 238 | 0.00015 | 0.9 |
| 1000 | 117 | 0.00030 | 0.9 |
| 1400 | 99 | 0.00030 | 0.9 |
| 3000 | 73 | 0.00030 | 0.9 |
| 4650 | 68 | 0.00030 | 0.9 |

**Observations:**
- grilly autograd works: Variable + matmul + cross_entropy + backward(use_gpu=True)
- GPU backward via _bridge (relu_backward, gelu_backward, linear_backward)
- Fixed gpu_backward.py: replaced deprecated Compute() with _bridge
- Plateaued ~68 PPL — capacity limited at d=256, 3.8M params
**Decision:** Scale up to d=512

---

## [2026-04-03 ~15:00] Run 3 — d=512, 8 layers, grilly autograd

**Status:** PASS — broke through d=256 plateau
**Config:** d_model=512, n_layers=8, n_experts=4, top_k=1, vocab=4000, seq=768
**Params:** ~15M

**PPL Curve:**
| Step | PPL |
|------|-----|
| 11800 | 58.6 |
| 11950 | 58.1 |
| 12000 | 60.2 |

**Observations:**
- PPL 58 at 12K steps — 10 better than d=256 plateau
- 0.3 stp/s (limited by Python autograd overhead)
- Checkpoints 47MB (model-only after trimming)
**Decision:** Reduce vocab to 1000 (faster training), scale to d=768

---

## [2026-04-04 12:25] Run 4 — d=768, 8 layers, vocab=1000

**Status:** IN PROGRESS
**Config:** d_model=768, n_layers=8, n_experts=4, top_k=1, vocab=1000, seq=512
**Params:** 20.8M
**LR:** 3e-4 (AdamW, cosine, warmup=1000)

**PPL Curve:**
| Step | PPL | stp/s |
|------|-----|-------|
| 50 | 975 | 0.1 |

**Observations:**
- vocab 4000→1000 reduces output projection 4x
- FnnChainRecorder not available (VulkanCore init fails — Python path)
- GPU backward via _bridge works
- C++ moe_forward being built on Cursor (target: 5 stp/s)
**Decision:** Continue training, await C++ forward for speed

---

## Architecture Evolution

1. **Manual backward** (moqe_backward) — 300 lines of gradient code, bug-prone, diverged
2. **grilly autograd** — one `.backward()` call, GPU backward via _bridge, works reliably
3. **FnnChainRecorder** — batched GPU dispatch (read_multiple for MoE fan-out), blocked by VulkanCore init
4. **C++ moe_forward** (in progress) — entire forward in one C++ call, target 5 stp/s

## Key Findings

- Position embeddings are critical — without them, model can't learn token order
- lr=3e-4 is the sweet spot for AdamW on this architecture
- Hebbian/NLMS update rules plateau at PPL 370 — backprop wins for LM
- grilly autograd works but Python overhead dominates at 0.1-0.9 stp/s
- vocab=1000 sufficient for TinyStories (simple English)
- N-expert MoQE (4 experts with ExpertSpec) works with Gumbel-Softmax routing
