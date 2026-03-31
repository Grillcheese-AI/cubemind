# CubeMind v2: A Neuro-Symbolic Cognitive Architecture with Spiking Perception, Vector Symbolic Reasoning, and Self-Learning Plasticity on Consumer Hardware

**Authors:** Nicolas Cloutier (Grillcheese AI)
**Target:** NeurIPS 2026 / AAAI 2027
**Format:** 9 pages + unlimited appendix

---

## Abstract (~250 words)

We present CubeMind v2, a neuro-symbolic cognitive architecture that integrates spiking neural networks, vector symbolic algebras, and neurochemical emotional modulation into a unified system capable of visual perception, abstract reasoning, temporal scene understanding, and emergent preference formation — all running on consumer GPU hardware via Vulkan compute shaders.

Our architecture introduces several novel contributions: (1) A spiking neural network perceptual layer with photonic-inspired STDP self-learning (Feldmann et al., 2019) that processes temporal streams through binary VSA temporal binding, compressing 300-frame video sequences into single 1.25KB binary vectors. (2) A 4-hormone neurochemical ODE system (cortisol, dopamine, serotonin, oxytocin) with non-linear couplings that produces emergent emotional states modulating SNN spike thresholds, learning rates, and attention routing. (3) A brain cortex architecture (thalamus, basal ganglia, personality Hebbian layer) that routes percepts through affect-keyed style prototypes, enabling emergent personality development through Oja plasticity. (4) A Mixture of Quantization Experts (MoQE) inference engine with hard-routed 4-bit/8-bit experts and fused GEMV via DP4a hardware instructions. (5) A taste formation mechanism where scene content is XOR-bound with emotional state and accumulated in continuous item memory, producing genuine preferences that emerge from experience rather than programming.

On abstract reasoning, CubeMind achieves 90.3% zero-shot accuracy on RAVEN and 100% on I-RAVEN-X at 100x out-of-distribution — without any training. On visual scene understanding, the hybrid CNN+rule detector pipeline achieves 84% on the I-RAVEN-X image benchmark. The full architecture processes webcam input at 32 FPS with real-time facial expression recognition (52 blendshapes), micro-expression detection, and face identity enrollment via Hamming distance retrieval.

All components run on a single AMD RX 6750 XT (12GB) using grilly's Vulkan compute backend.

---

## 1. Introduction (1.5 pages)

### 1.1 Motivation
Current AI systems are fragmented: LLMs reason but can't perceive, CNNs perceive but can't reason symbolically, and neither develops emotional responses or preferences from experience. Biological cognition unifies perception, reasoning, emotion, and memory in a single architecture. CubeMind attempts this integration on consumer hardware.

### 1.2 The Neural Binding Problem
When CNNs process scenes with multiple objects, standard representations suffer from the "superposition catastrophe" — attribute-object bindings blur together. We solve this with Vector Symbolic Architectures (block-codes) where binding is circular convolution and bundling is element-wise addition, preserving compositional structure.

### 1.3 Why VSA + SNN + Neurochemistry?
- **VSA**: Provides the symbolic algebra (bind, unbind, bundle, similarity) for compositional reasoning without backpropagation
- **SNN**: Provides temporal processing with extreme sparsity (only fires on change) and self-learning via STDP
- **Neurochemistry**: Provides emotional modulation that shapes attention, learning rate, and personality over time
- **Together**: A system that perceives, reasons, feels, and develops preferences — the minimal viable cognitive architecture

### 1.4 Contributions
1. Zero-shot 90.3% on RAVEN, 100% on I-RAVEN-X (100x OOD) — no training
2. 84% hybrid accuracy on I-RAVEN-X image benchmark (CNN + rule detectors)
3. Photonic STDP self-learning in SNN encoder (Feldmann et al. 2019, adapted)
4. 4-hormone ODE neurochemistry with non-linear couplings and emergent emotions
5. Brain cortex: thalamus routing, basal ganglia strategy, personality Hebbian layer
6. MoQE: fused 4-bit/8-bit mixed-precision inference with DP4a
7. Multi-teacher ensemble distillation with identity prompt conditioning
8. Taste formation via scene×emotion XOR binding in continuous item memory
9. Real-time face perception: 52 blendshapes, micro-expressions, identity recognition
10. Full architecture on consumer GPU (RX 6750 XT) via Vulkan compute shaders

---

## 2. Block-Code VSA Foundation (1 page)

### 2.1 Representation
Block-codes: k=80 blocks × l=128 length = 10,240 dimensions. Each block is an independent probability distribution. Binding via per-block circular convolution, unbinding via correlation, bundling via element-wise addition.

### 2.2 SDLS Purification
Semantically Decoupled Latent Steering: orthogonal projection onto the null-space of the document/corpus mean. Removes shared vocabulary noise while preserving discriminative signal. Applied in both HyperAxialAttention and SemanticEncoder.

$$X_{purified} = X - (X \cdot \hat{m})\hat{m}$$

### 2.3 HyperAxialAttention
O(L) attention via LSH bucketing with SimHash. Low-rank Q/K/V projections (400MB → 30MB per matrix). SDLS noise removal built into the forward pass. Oja refinement within buckets via sigmoid soft-gating.

### 2.4 HYLA Factored Hypernetwork
Generates mainnet weight matrices from VSA embeddings. Factored: W_A (d_out, rank) + W_B (rank, d_vsa) instead of monolithic W_H (d_out×d_vsa, d_hidden). Reduces 53GB → 670MB at d_vsa=10240.

---

## 3. Perception Pipeline (2 pages)

### 3.1 CNN Encoder
Lightweight 3-layer conv stack (32→64→128 channels) with GELU + MaxPool. Maps 80×80 grayscale images to (k, l) block-codes via adaptive average pooling + block softmax. Uses grilly GPU conv2d (GEMM im2col path).

### 3.2 Additive Cross-Entropy Training
Following the analysis in our CNN paper: direct block cross-entropy against hash-bound targets fails (loss stuck at ln(64) = 4.16, maximum entropy). The solution: train CNN to predict unbound bundled superpositions via Additive Cross-Entropy against a frozen dictionary codebook. Binding happens algorithmically in the symbolic backend.

$$L(X, Y, \theta) = -\log \frac{\exp(s_l \cdot \sum_j \text{sim}(f_\theta(X), w_{y_j}))}{\sum_{i=1}^m \exp(s_l \cdot \text{sim}(f_\theta(X), w_i))}$$

### 3.3 SemanticEncoder
Text → semantic embedding → VSA projection. Supports BGE-M3 GGUF (1024D), sentence-transformers (384D), or hash fallback. SDLS corpus purification for document retrieval. Lazy import avoids 50GB+ RAM allocation from Vulkan-compiled llama-cpp-python.

### 3.4 VisionEncoder
Model-agnostic image → VSA projection. Backends: transformers (SigLIP, CLIP), open_clip, llama-cpp, or pixel statistics fallback. Orthogonal projection P via QR decomposition.

### 3.5 Perceiver Cross-Attention
Image patches → learned latent vectors via cross-attention (O(n_latents × N_patches)). Positional encoding on patches. Mean pool to dense vector → LSH → binarize → pack. Used for full image VSA pipeline.

### 3.6 Face Perception
MediaPipe Face Landmarker: 468 3D landmarks + 52 blendshapes (Action Units) in <5ms. Micro-expression detection via temporal blendshape delta analysis. Face identity recognition via 128D inter-landmark distance features → LSH → Hamming similarity.

---

## 4. Spiking Neural Network Layer (1.5 pages)

### 4.1 LIF/IF Neurons
Leaky Integrate-and-Fire (transient change detection) and Integrate-and-Fire (weak persistent signal accumulation). Neurochemical modulation of threshold and leak rate. Uses grilly LIFNode GPU shaders when available.

### 4.2 Photonic-Inspired STDP
Adapted from Feldmann et al. (Nature, 2019) — all-optical spiking neurosynaptic networks with self-learning. Phase-change material synapses potentiate (amorphize) on co-activity, depress (crystallize) on non-contribution. Our software implementation:

- Inputs that contributed to a spike → potentiate (W[fired] += lr × active_inputs)
- Inputs that didn't contribute → depress (W[not_fired] -= lr × active_inputs)
- Dopamine-gated learning rate: lr *= (0.5 + dopamine) — novelty accelerates learning
- Weight decay: W *= (1 - 0.005) per step — unused connections fade (PCM crystallization)
- Weight clip: prevents saturation (physical PCM bounds)

### 4.3 Temporal VSA Binding
Temporal streams compressed via cyclic shift + XOR:
```
accumulator = shift(accumulator, 1) XOR pack(frame_spikes)
```
300-frame video → single 1.25KB binary vector. Sequence order is encoded in the shift positions. Retrieval via Hamming distance against ContinuousItemMemory.

### 4.4 Neurochemical ODE System
Four-hormone dynamics ported from grillcheese.brain.endocrine:

$$\frac{dH}{dt} = \alpha \cdot \text{drive} - \beta \cdot H$$

Non-linear couplings:
- Cortisol dampens dopamine (−12%) and serotonin (−8%)
- Serotonin boosts oxytocin (+6%)
- Cortisol suppresses oxytocin (−5%)

Emergent emotions: joy (high dopamine + positive valence), anxious (high cortisol), curious (high novelty), warm (high oxytocin), sad (negative valence).

---

## 5. Brain Cortex Architecture (1 page)

### 5.1 Circadian Cells
Gaussian tuning curves for hour/day/season produce temporal embeddings. Gives the system temporal context: time of day, weekday/weekend, season.

### 5.2 Thalamus
Sensory gateway: salience scoring (GPU linear + sigmoid) and attention routing to memory/emotion/reasoning/response pathways (GPU softmax).

### 5.3 Basal Ganglia
Action selection: selects communication strategy (informative/empathetic/questioning/action) based on thalamus route weights + affect state. Go/no-go gating with confidence threshold.

### 5.4 CNS State
Global controller: consciousness levels (DEEP_SLEEP → HYPERVIGILANT), cumulative stress tracking, fatigue accumulation, focus intensity.

### 5.5 Personality Hebbian Layer
Six learnable style prototypes (analytical, warm, direct, curious, cautious, playful). Affect-keyed routing via cosine similarity → softmax. Per-style Hebbian transformation matrices updated via Oja's rule. Personality develops from accumulated interactions.

---

## 6. Scene Understanding and Taste Formation (1 page)

### 6.1 Video Analysis Pipeline
Video → subsample frames → per-frame features (face blendshapes + vision) → SNN temporal encoding per segment → scene segmentation via Hamming distance spikes → per-segment neurochemistry snapshot → temporal scene graph.

### 6.2 Taste = Scene XOR Emotion
Each scene segment's binary VSA vector is XOR-bound with the current emotional state vector:
$$\text{taste\_vector} = \text{scene\_vector} \oplus \text{emotion\_vector}$$

This creates a unique binary signature encoding *what was seen* fused with *how it felt*. Stored in ContinuousItemMemory.

### 6.3 Preference Formation
On future encounters with similar content:
1. Encode new scene → query taste memory via Hamming similarity
2. Retrieve past emotional associations → weighted average of stored taste scores
3. Blend: 40% current neurochemistry + 60% accumulated history = taste score
4. Over repeated exposures: preferences crystallize from experience

Experimental results on three video types:
| Content | 1st Watch | 4th Watch | Preference |
|---------|-----------|-----------|------------|
| Nature (bear) | +0.17 (2★) | +0.23 (3★) | Growing fondness |
| Emotional (faces) | +0.25 (3★) | — | Highest rated |
| Fast action (cars) | +0.20 (2★) | — | Least preferred |

### 6.4 Critique Generation
Template-based (upgradeable to MoQE LLM): per-segment analysis with energy/mood descriptors, emotional arc detection, neurochemistry commentary ("my cortisol was elevated"), star rating informed by accumulated taste.

---

## 7. MoQE Inference Engine (0.5 pages)

### 7.1 Architecture
Hard-routed mixture of two experts: 4-bit (85% of tokens) and 8-bit (15%). Router: Linear → sigmoid → hard threshold. Fused GEMV: dynamic FP32→INT8 quantization in registers + integer dot product + dequantize.

### 7.2 Multi-Teacher Ensemble Distillation
Three teachers (Llama 3.3 70B + Qwen3-Coder-Next 80B + Mistral Large 123B) extracted on A100, merged via weighted probability averaging. Identity prompt conditioning: student inherits CubeMind personality traits during distillation.

Loss: 0.3 × CE(hard labels) + 0.6 × KL(soft ensemble) + 0.1 × router balance

### 7.3 Vulkan GPU Shaders
- `moqe-hard-router.glsl`: subgroup ballot routing
- `moqe-dynamic-quant.glsl`: subgroupMax absmax → INT8 in one cycle
- `moqe-fused-gemv-dp4a.glsl`: DP4a 4× throughput, wave-size adaptive

---

## 8. Experiments (1 page)

### 8.1 Abstract Reasoning

**Table 1: RAVEN Zero-Shot (no training)**
| Method | Training | Accuracy |
|--------|---------|---------|
| LSTM | Supervised | 13.1% |
| ResNet | Supervised | 53.4% |
| NVSA | Supervised | 87.7% |
| DCNet | Supervised | 93.6% |
| **CubeMind** | **None** | **90.3%** |

**Table 2: I-RAVEN-X OOD**
| maxval | OOD Factor | Accuracy |
|--------|-----------|---------|
| 10 | 1x | 98.5% |
| 100 | 10x | 99.8% |
| 1000 | 100x | 100.0% |

### 8.2 Visual Reasoning (I-RAVEN-X Image Benchmark)

**Table 3: Image-based reasoning**
| Mode | Accuracy | Latency |
|------|---------|---------|
| Random baseline | 12.5% | — |
| Visual only (CNN, untrained) | 22.0% | 1.14s |
| **Hybrid (70% algebraic + 30% visual)** | **84.0%** | 1.16s |
| Integer detectors (no images) | 90.3% | 0.03s |

### 8.3 Real-Time Perception

**Table 4: Live webcam pipeline**
| Component | Latency | Notes |
|-----------|---------|-------|
| MediaPipe face (52 blendshapes) | <5ms | XNNPACK CPU |
| SNN step (256 LIF neurons) | <1ms | numpy/grilly |
| Neurochemistry ODE update | <0.1ms | numpy |
| Thalamus routing (GPU) | <0.5ms | grilly linear+softmax |
| Full pipeline | ~30ms | 32 FPS sustained |

### 8.4 Scene Understanding
Video critique tested on 3 content types (nature, emotional, fast-action). System develops measurable preferences after 20+ scene exposures. Taste scores diverge from neutral: calm nature content rated increasingly higher on repeated exposure (+0.17 → +0.23).

### 8.5 Memory Efficiency

**Table 5: Memory optimizations**
| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| HYLA W_H | 53.7 GB | 670 MB | 80× |
| HyperAxialAttention W_Q/K/V | 1.2 GB | 30 MB | 40× |
| llama-cpp import | 50+ GB | 0 (lazy) | ∞ |

---

## 9. Related Work (0.5 pages)

### Neuro-Symbolic Reasoning
NVSA (Hersche et al., 2023), PrAE (Zhang et al., 2021), DCNet, CoPINet. CubeMind achieves comparable accuracy without any training.

### Vector Symbolic Architectures
Kanerva (2009), Plate (2003), Block-codes (Laiho et al., 2015). We extend with SDLS purification and factored hypernetworks.

### Spiking Neural Networks
LIF neurons (Lapicque, 1907), STDP (Bi & Poo, 1998), photonic SNN (Feldmann et al., 2019). We adapt the photonic STDP rule for software simulation with neurochemical modulation.

### Neuromorphic Computing
BrainScaleS, Intel Loihi, SpiNNaker. Our contribution: consumer GPU implementation via Vulkan compute shaders, accessible without specialized neuromorphic hardware.

### Emotional AI
Affective computing (Picard, 1997), hormonal modulation in robotics. We implement a 4-hormone ODE system with non-linear couplings driving emergent behavior.

---

## 10. Conclusion (0.5 pages)

CubeMind v2 demonstrates that a unified cognitive architecture — integrating symbolic reasoning, spiking perception, emotional modulation, and preference formation — can run on consumer hardware without cloud dependency. The system achieves state-of-the-art zero-shot abstract reasoning while simultaneously processing live video with facial expression recognition, developing a personality through Hebbian plasticity, and forming genuine taste from accumulated experience.

Key insight: taste and personality are not features to be programmed but properties that emerge from the interaction of neurochemistry, STDP plasticity, and accumulated memory. The more CubeMind experiences, the more individual it becomes.

Future work: contrastive pre-training of the Perceiver for visual features, integration of the MoQE LLM for natural language critique generation, hearing/audio perception via SNN temporal encoding, and scaling to multi-modal scene understanding.

---

## Appendix

### A. Full Architecture Diagram
Pipeline: Input → Perception (CNN/SNN/Face/Vision) → Thalamus → Brain Cortex → VSA Memory → MoQE LLM → Output

### B. Grilly Vulkan Shader Inventory
12 new compute shaders: SDR bundling, overlap metrics, Hamming top-K, binarize-pack, MoQE routing, dynamic quantization, fused GEMV, DP4a.

### C. CNN Training Loss Analysis
Mathematical proof that block cross-entropy against hash-bound VSA targets converges to maximum entropy at ln(64) = 4.16. Solution: Additive Cross-Entropy with bundle-predictive learning.

### D. Neurochemistry ODE Parameters
Full parameter table for α, β, drive functions, and non-linear coupling coefficients.

### E. Taste Formation Longitudinal Study
Detailed taste score progression across 38+ scene exposures across 3 content types.

### F. Identity Prompt
Full CubeMind identity prompt used for multi-teacher distillation conditioning.

### G. Reproducibility
All code at github.com/Grillcheese-AI/cubemind. 633 tests passing. Single-command reproduction: `uv run pytest tests/ -q`.

---

## Codebase Statistics

| Metric | Value |
|--------|-------|
| Source files | 88 |
| Lines of code | 22,596 |
| Test files | 42 |
| Tests passing | 633 |
| Vulkan shaders (new) | 12 |
| Python ≥3.12 | Required |
| GPU | AMD RX 6750 XT (12GB) |
| Framework | grilly (Vulkan compute) |
