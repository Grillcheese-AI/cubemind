# HE-MoE TinyStories — Real Text Benchmark

## H1: HE-MoE achieves finite perplexity on real text
**Claim:** HE-MoE produces valid next-token probabilities on TinyStories.
**Pass:** perplexity < 100 (finite, not random chance).

## H2: HE-MoE perplexity decreases over training
**Claim:** Loss drops during 5000 steps of training.
**Pass:** final_ppl < initial_ppl.

## H3: HE-MoE is competitive with MLP baseline
**Claim:** No-backprop HE-MoE within 3x of MLP perplexity.
**Pass:** he_moe_ppl < 3 * mlp_ppl.

## H4: Each HE-MoE component helps on real text
**Claim:** Ablating charges/force/consolidation/sleep hurts perplexity.
**Pass:** at least 2 of 4 ablations have worse ppl than full.

## H5: HE-MoE generates coherent text
**Claim:** Sampling from HE-MoE produces recognizable English words.
**Pass:** manual inspection of 5 generated sequences.
