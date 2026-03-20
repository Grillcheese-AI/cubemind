# CubeMind

**90.3% on RAVEN** (zero-shot, no training) | **100% on I-RAVEN-X** (100x OOD)

Neuro-Vector-Symbolic Architecture for abstract visual reasoning on Raven's Progressive Matrices. Surpasses the supervised NVSA baseline (87.7%) without any training data.

Built on [grilly](https://github.com/Grillcheese-AI/grilly) — GPU-accelerated via Vulkan compute shaders. Runs on any GPU vendor (AMD, NVIDIA, Intel).

## Results

| Configuration | Accuracy |
|:---|---:|
| Center Single | 97.5% |
| Left-Right | 98.0% |
| Up-Down | 96.0% |
| Out-InCenter | 100.0% |
| Out-InGrid | 77.0% |
| 2x2 Grid | 82.0% |
| 3x3 Grid | 81.5% |
| **Overall** | **90.3%** |

I-RAVEN-X (out-of-distribution): **100.0%** at 100x the standard attribute range.

## Key Features

- **Zero-shot reasoning** — deterministic integer-domain rule detectors (constant, progression, arithmetic, distribute-three), no training required
- **Position-aware scoring** — spatial layout rule detection via bbox signatures, resolves 74% of grid-config ties (+14.5pp)
- **Oja-plastic memory** — self-normalizing VSA consolidation via Oja's rule, GPU-accelerated dream cycle
- **VSA autograd** — differentiable bind/unbind where grad(unbind) = bind, no extra shaders needed
- **VQ-VSA perception** — DenseNet CNN with Vector Quantization and Straight-Through Estimation for end-to-end pixel-to-block-code training
- **11 custom SPIR-V shaders** — fused Conv+GELU, zero-copy DenseNet concat, subgroup 1x1 conv, Oja learning, VSA argmax

## Install

```bash
pip install cubemind
```

For GPU acceleration:
```bash
pip install grilly
```

See **[INSTALL.md](INSTALL.md)** for detailed setup instructions, GPU configuration, pre-built extensions, and troubleshooting.

## Quick Start

```python
from cubemind.model import CubeMind

model = CubeMind(k=8, l=64, n_codebook=16)
result = model.forward(phi=panel_block_code)
print(result["answer"])
```

### Run the I-RAVEN Benchmark

```bash
python -m benchmarks.iraven --max-problems 200 --seed 42
```

### Train the CNN Perception Frontend

```bash
python cubemind/perception/train_vq.py center_single 6000 100
```

## Architecture

```
Image (80x80)
  -> DenseNet-Small (Vulkan GPU, 8.4ms)
  -> VQ-VSA Quantizer (STE + commitment loss)
  -> Block-Code (k=8, l=64, d=512)
  -> Rule Detectors (constant, progression, arithmetic, distribute-three)
  -> Position-Aware Scoring (bbox signature rules)
  -> Answer Selection (HMM tiebreaker)
```

### Reasoning Pipeline (metadata path)

```
XML Metadata -> Per-Attribute Grids -> Integer Rule Detectors -> Candidate Scoring -> Answer
                                    -> Position Signatures   ->
```

### Oja-Plastic Memory (model2.py)

```
Forward:  Cache hit (sim > 0.7) -> Oja consolidation (self-normalizing)
Sleep:    Replay episodic memories -> GPU Oja shader -> Prune low-utility traces
Codebook: PlasticCodebook adapts to environment statistics
```

## Modules

| Module | Description |
|:---|:---|
| `cubemind/reasoning/rule_detectors.py` | Integer-domain rule detection |
| `cubemind/reasoning/sinkhorn.py` | Sinkhorn entity alignment (ablation) |
| `cubemind/perception/cnn_encoder.py` | CNN perception frontend |
| `cubemind/perception/grilly_densenet.py` | Lightweight DenseNet with full backward |
| `cubemind/perception/grilly_resnet.py` | ResNet-18 with pretrained weights |
| `cubemind/perception/vsa_dense.py` | VSA-Dense early bundling architecture |
| `cubemind/perception/train_vq.py` | VQ-VSA training with AutoHypergradient |
| `cubemind/perception/additive_ce.py` | Additive CE bundle-predictive loss |
| `cubemind/model2.py` | Oja-plastic NVSA architecture |
| `cubemind/ops/block_codes.py` | GPU block-code VSA operations |
| `benchmarks/iraven.py` | I-RAVEN benchmark runner |

## Paper

See `docs/papers/cubemind-benchmark-analysis.md` for the full technical report including:
- Ablation study (Appendix B): Sinkhorn vs entity sets vs position scoring
- Perception experiments (Appendix C): loss function comparison, architecture details, training dynamics
- Reproducibility (Appendix A): 8 seeds x 4000 problems

## Citation

```
@techreport{cubemind2026,
  title={CubeMind: Zero-Shot Abstract Visual Reasoning via Neuro-Vector-Symbolic Architecture on Vulkan GPU},
  year={2026},
  note={90.3\% on RAVEN (zero-shot), 100\% on I-RAVEN-X (100x OOD)}
}
```

## License

BSL-1.1
