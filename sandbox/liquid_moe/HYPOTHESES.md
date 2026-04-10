# LiquidMoE — Hypothesis Progression

## H1: Bandit Q-values converge from per-token reward
**Claim:** A simple EMA update Q ← (1-β)Q + β*reward, using -loss as reward,
causes expert Q-values to differentiate after N steps.
**Test:** Feed 200 random inputs, check Q-values are not all equal.
**Pass criteria:** std(Q) > 0.01 after 200 steps.

## H2: UCB exploration prevents expert collapse
**Claim:** UCB bonus sqrt(ln(t)/N_e) ensures all experts get selected
at least 10% of the time over 500 steps.
**Test:** Track per-expert selection count over 500 steps.
**Pass criteria:** min(N_e) / max(N_e) > 0.3 (no expert starved).

## H3: Top-k routing produces valid blended output
**Claim:** Selecting top-k experts and normalizing weights produces
a valid output that is a weighted combination of expert outputs.
**Test:** Compare blended output to individual expert outputs.
**Pass criteria:** output = sum(w_i * expert_i(x)), weights sum to 1.

## H4: Oja residual EMA tracks reconstruction quality
**Claim:** Residual EMA increases when inputs are outside the experts'
representation capacity and decreases when inputs are familiar.
**Test:** Train on domain A, then switch to domain B.
**Pass criteria:** residual_ema spikes on domain switch.

## H5: Expert spawning occurs when residual exceeds threshold
**Claim:** A new expert is created when residual_ema > threshold,
initialized along the residual direction.
**Test:** Set low threshold, feed diverse data.
**Pass criteria:** n_experts increases from initial count.

## H6: Spawned experts get routed to by the bandit
**Claim:** A newly spawned expert receives tokens within 50 steps
of creation (UCB gives it exploration bonus from low N_e).
**Test:** Spawn an expert, continue training, check its n_uses.
**Pass criteria:** spawned expert n_uses > 0 within 50 steps.

## H7: Eligibility traces accumulate on active experts
**Claim:** Active experts accumulate a_trace (activity) and e_trace (error)
that reflect their recent engagement and error history.
**Test:** Run 50 steps, check that active experts have non-zero traces.
**Pass criteria:** active experts have a_trace > 0 and ||e_trace|| > 0.

## H8: Inactive experts consolidate via trace products
**Claim:** When an expert is not in top-k, its weights change via
Δw = η * a_trace * e_trace * w (offline learning, no energy wasted).
**Test:** Set top_k=1 (3 inactive per step), verify weight changes.
**Pass criteria:** at least one inactive expert's W2 changes after 20 steps.

## H9: Hippocampal capsules store surprising events
**Claim:** Events with error > threshold are stored with full context
(input, expert_ids, error, timestamp). Retrieval finds similar events.
**Test:** Create high-error events, verify storage and retrieval.
**Pass criteria:** capsules stored, retrieved by similarity.

## H10: Sleep replay consolidates long-term patterns
**Claim:** Replaying stored capsules during a "sleep" phase moves
expert weights toward patterns seen during waking, improving
performance on previously-seen domains.
**Test:** Train on domain A, store capsules. Switch to domain B.
Sleep replay. Test domain A performance vs no-replay baseline.
**Pass criteria:** post-sleep domain A loss < no-replay domain A loss.
