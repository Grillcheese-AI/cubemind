# Archived Experiments

Experiments that completed their hypothesis cycle but were not promoted to production.
Per GRL process section 10: failed/superseded experiments are archived, not deleted.

## he_moe/ (2026-04-02)
HE-MoE: Hilbert-Electrostatic Mixture of Experts. Coulomb force routing in RKHS.
All 12 hypotheses passed. Staging integration + stress tests passed (33 tests).
**Not promoted**: Oja/Hebbian learning plateaus at PPL ~370 on TinyStories.
Backprop-based MoQE achieves PPL <60 on same task. HE-MoE concepts (Coulomb routing,
hippocampal capsules) may be useful for non-LM tasks.

## he_moe_tinystories/ (2026-04-02 — 2026-04-03)
HE-MoE + LiquidMoE hybrid on TinyStories. Tested: Hebbian delta rule, NLMS experts,
UCB+cosine routing, LiquidStateCell. All variants plateaued at PPL 370-420.
Confirmed: no-backprop update rules cannot compete with gradient descent for LM.
