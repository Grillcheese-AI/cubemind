# HE-MoE (Hilbert-Electrostatic MoE) — Hypothesis Progression

## H1: RBF kernel similarity reflects input distance
**Claim:** k(x,x) = 1 (self-similarity), k(x,y) → 0 as ||x-y|| → ∞.
RKHS distance d²(x,y) = 2 - 2k(x,y) is a valid metric.
**Test:** Compute kernel for identical, similar, and distant vectors.
**Pass criteria:** k(x,x)=1, k(near)>0.5, k(far)<0.1, distance monotonic.

## H2: Random Fourier Features approximate the kernel
**Claim:** φ(x)·φ(y) ≈ k(x,y) where φ is the RFF mapping.
**Test:** Compare exact kernel to RFF dot product over 100 pairs.
**Pass criteria:** mean |exact - approx| < 0.15 with d_rff=256.

## H3: Charged experts produce asymmetric routing scores
**Claim:** Positive-charge expert near input gives high score.
Negative-charge expert near input gives low/negative score.
**Test:** Place expert at x, query at x. Compare +1 vs -1 charge.
**Pass criteria:** score(+1) > 0 > score(-1) for same position.

## H4: Coulomb force points toward/away from expert center
**Claim:** Force on trace from positive expert points toward μ.
Force from negative expert points away from μ.
**Test:** Compute force vectors, check dot product with (x - μ).
**Pass criteria:** dot(F, x-μ) > 0 for positive, < 0 for negative.

## H5: Force-based routing selects nearest compatible experts
**Claim:** Top-k by attraction score selects experts whose centers
are closest to the input AND have positive charge.
**Test:** Place 4 experts at known positions, query near one.
**Pass criteria:** nearest positive expert is always in top-k.

## H6: Expert positions move via Coulomb force updates
**Claim:** After N force updates, expert μ moves toward the centroid
of inputs it was attracted to.
**Test:** Feed clustered inputs, track expert position drift.
**Pass criteria:** expert μ moves toward cluster center.

## H7: Oja weight update keeps experts normalized
**Claim:** After Oja updates, expert weight row norms stay ≈ 1.
**Test:** Run 100 updates, check row norms.
**Pass criteria:** all row norms in [0.8, 1.2].

## H8: Charge flipping reflects error history
**Claim:** Expert that consistently reduces error keeps charge +1.
Expert that consistently increases error flips to -1.
**Test:** Create "good" and "bad" expert conditions, track charge.
**Pass criteria:** good expert stays +1, bad expert flips to -1.

## H9: Inactive experts consolidate via trace-bound states
**Claim:** Inactive expert weights change via eligibility trace products
(same as OCH-MoE H8 but in Hilbert space context).
**Test:** Set top_k=1, verify inactive expert W changes.
**Pass criteria:** inactive expert weights ≠ initial after 50 steps.

## H10: Hippocampal capsules store high-error events
**Claim:** Events exceeding error threshold get stored with full context.
**Test:** Generate high-error inputs, count stored capsules.
**Pass criteria:** n_capsules > 0, capsule contains context + error + expert_ids.

## H11: Sleep replay moves expert positions toward stored traces
**Claim:** After sleep_replay(), expert μ moves toward capsule contexts.
**Test:** Store capsules, record μ before sleep, run sleep, compare.
**Pass criteria:** Σ||μ_after - capsule_context|| < Σ||μ_before - capsule_context||.

## H12: Full HE-MoE system is stable under sustained training
**Claim:** 1000 steps of training with spawning, charge flipping,
consolidation, and replay produces no NaN/Inf and system remains functional.
**Test:** Run 1000 steps, check all losses finite, n_experts ≤ max.
**Pass criteria:** all(isfinite(loss)), n_experts ≤ max_experts, system outputs valid vectors.
