# LiquidMoE / OCH-MoE Results Log

## [2026-04-02 02:00] H1-H3 — Bandit Router Basics

**Status:** PASS
**Config:** 4 experts, d=16, top_k=2, beta=0.1
**Measurements:**
- Q-values differentiate: std(Q) > 0.01 ✅
- Weights sum to ~1.0 ✅
- Indices within valid range ✅
- Q-values update on reward ✅

**Decision:** Proceed to expert spawning.

---

## [2026-04-02 02:00] H4-H6 — Oja Spawning + Dynamic Growth

**Status:** PASS
**Config:** spawn_threshold=0.1, initial=2, max=8
**Measurements:**
- Residual EMA tracks reconstruction quality ✅
- Experts spawn when threshold exceeded ✅
- Spawned experts get routed to (UCB exploration) ✅
- Router W expands with new experts ✅

**Decision:** Proceed to eligibility traces.

---

## [2026-04-02 02:15] H7-H8 — Eligibility Traces + Consolidation

**Status:** PASS
**Config:** top_k=1, eta_consol=0.1
**Measurements:**
- Active experts accumulate a_trace > 0 ✅
- Inactive expert weights change via consolidation ✅

**Observations:** Eligibility trace products (a_trace * e_trace * w) provide
meaningful gradient-free update signal for offline learning.
**Decision:** Proceed to hippocampal capsules.

---

## [2026-04-02 02:15] H9-H10 — Hippocampal Capsules + Sleep

**Status:** PASS
**Config:** surprise_threshold=0.1, replay_interval=10
**Measurements:**
- Capsules stored when error > threshold ✅
- Max capsules respected ✅
- Sleep consolidation changes expert weights ✅
- Contrastive router updates change router W ✅

**Observations:** Sleep replay of high-error capsules reinforces experts
that were active during surprising events. This is biologically
plausible — hippocampal replay during sleep consolidates memories.
**Decision:** ALL HYPOTHESES CONFIRMED. Proceed to staging.

---

## [2026-04-02 02:30] Full Pipeline Integration Test

**Status:** PASS
**Config:** d=16, 200 steps, all features enabled
**Measurements:**
- Experts grew from 2 → N (N varies) ✅
- Capsules stored ✅
- Sleep consolidation functional ✅
- Integration with BlockCodes ✅
- Integration with GIFNeuron ✅
- Integration with HippocampalFormation ✅

**Decision:** Ready for staging with real CubeMind components.
