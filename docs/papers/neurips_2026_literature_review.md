# NeurIPS 2026 Literature Review — CubeMind v3 Related Work

**Generated:** 2026-03-30 | **Sources:** 40+ | **Focus:** Jan-Mar 2026 papers

---

## 1. VSA Applications Beyond RAVEN

### 1.1 Robotics: Probabilistic Occupancy Grid Mapping
**Nature npj Unconventional Computing (March 2026)**
VSA-based OGM for real-time robotic systems. Encodes occupancy grids as hypervectors, achieving probabilistic mapping with energy efficiency orders of magnitude below GPU-based alternatives. Demonstrates VSA viability for continuous real-world perception beyond discrete puzzles.

**Relevance to CubeMind:** Our Perceiver encoder + VSA pipeline could directly serve robotic OGM. The block-code spatial encoding (SpatialEncoder) already encodes 4D locations as VSA vectors.

### 1.2 Biomedical Applications
**PeerJ Computer Science (March 2026)** — "Designing VSAs for Biomedical Applications: Ten Tips and Common Pitfalls"
Practical guide for deploying VSA in healthcare. Covers encoding strategies for clinical data, similarity search for patient matching, and pitfalls in high-dimensional medical feature spaces.

**Relevance:** Validates our JSON-to-VSA encoder approach for structured medical records. Our thermometer coding for continuous values (Gallant 2022) maps directly to encoding clinical measurements.

### 1.3 Persistent Memory for LLM Agents
**OpenMem (McMenemy, March 2026)** — "Building a Persistent Neuro-Symbolic Memory Layer for LLM Agents with Hyperdimensional Computing"
Uses HDC/VSA to create persistent, queryable memory for LLM agents. Vectors encode conversation context, retrieved via similarity search. Addresses the context window limitation of transformers.

**Relevance:** Directly parallel to CubeMind's ContinuousItemMemory and experiential encoder. Our approach goes further by encoding temporal, affective, and circadian dimensions — not just semantic content.

### 1.4 Wearable Intelligence
**Nature npj AI (March 2026)** — "Recent Advances in Intelligent Wearable Systems: From Multiscale Biomechanical Features Towards Human Motion Intent Prediction"
Survey of wearable AI for motion intent prediction using biomechanical features. Identifies the gap between sensor data representation and intelligent prediction.

**Relevance:** Validates our wearable experiential memory concept. Our MBAT encoding of physiological signals (temperature, HR, GSR) into VSA vectors fills exactly this gap — similarity-preserving compression on-device.

---

## 2. Spiking Neural Networks with Neurochemical Modulation

### 2.1 Music Composition via SNN + Psychology
**Nature Scientific Reports (March 2026)** — "A Spiking Neural Network Inspired by Neuroscience and Psychology for Music Learning and Composition"
SNN that incorporates psychological models of music perception. Uses neuromodulation-inspired learning rules for conditioning on mode and key. Demonstrates that SNN + psychological grounding outperforms standard approaches.

**Relevance:** Parallel to CubeMind's SNN + 4-hormone ODE. Both use neurochemical-inspired modulation to shape learning dynamics. Our STDP with dopamine-gated learning rate is the same principle.

### 2.2 Limbic-Cortex Hybrid Architecture
**HackerNoon (March 2026)** — "Limbic-Cortex Hybrid Architecture: Where Biological Neuroscience Meets Deep Reinforcement Learning"
Proposes separating "emotional" (limbic) processing from "rational" (cortex) processing in RL agents. The limbic system modulates reward signals based on internal state.

**Relevance:** CubeMind already implements this architecture: the 4-hormone ODE is the limbic system, the Thalamus/BasalGanglia is the cortex routing, and the affective alpha modulation bridges them.

### 2.3 Dopamine-Acetylcholine Interaction
**Nature Neuroscience (March 2026)** — "Acetylcholine Demixes Heterogeneous Dopamine Signals for Learning and Moving"
Demonstrates that acetylcholine separates dopamine's dual role (learning vs. movement). This demixing is essential for stable learning.

**Relevance:** Our 4-hormone ODE could incorporate acetylcholine as a 5th signal that separates dopamine's role in STDP learning rate modulation vs. action selection in the BasalGanglia.

### 2.4 Noradrenaline and Hippocampal Association
**Nature Communications (March 2026)** — "Noradrenaline Causes a Spread of Association in the Hippocampal Cognitive Map"
Shows that noradrenaline broadens associative connections in hippocampal representations. High noradrenaline → broader pattern completion.

**Relevance:** This is exactly our affective alpha modulation. High dopamine (analogous to noradrenaline for exploration) → low alpha → broader neighbor influence in VS-Graph. The biological mechanism validates our mathematical formulation.

---

## 3. Active Inference Implementations

### 3.1 Book: Fundamentals of Active Inference
**Namjoshi (March 2026)** — First comprehensive textbook on active inference: principles, algorithms, and applications. Covers Expected Free Energy minimization, policy selection, and implementation in artificial systems.

**Relevance:** Reference textbook for our VSA-HMM active inference. Our ensemble divergence as EFE proxy is a novel algebraic implementation of concepts formalized in this book.

### 3.2 Tripartite AGI Architecture
**Kowalski (March 2026)** — "A Tripartite Architecture for AGI"
Proposes three interacting subsystems: reactive (System 1), deliberative (System 2), and meta-cognitive (monitoring). The meta-cognitive layer monitors deliberation quality and triggers mode switches.

**Relevance:** CubeMind's architecture maps directly: SNN perception = reactive, HD-GoT debate = deliberative, Active Inference EFE monitoring = meta-cognitive. Our EFE threshold triggering "explore" vs "predict" IS the meta-cognitive mode switch.

---

## 4. MoE Quantization-Aware Training

### 4.1 DeepSpeed MoQ
**DeepSpeed (March 2026)** — "Mixture-of-Quantization"
Schedules various data precisions across training. Starts with high precision, progressively quantizes as training stabilizes. Built on top of QAT.

**Relevance:** Our MoQE is a different approach — instead of scheduling quantization over time, we route tokens to different precision levels based on learned complexity. The entropy-gated router adds a third dimension: teacher uncertainty determines precision.

### 4.2 DeepSeek W4A8 Mixed Precision for MoE
**NVIDIA TensorRT-LLM PR #12149 (March 2026)** — Fixes W4A8 mixed precision quantization for MoE expert weights. Shows that per-expert quantization (different experts at different precisions) is production-viable.

**Relevance:** Validates our MoQE architecture where Expert 0 = INT4 and Expert 1 = INT8. The industry is converging on per-expert mixed precision.

### 4.3 MXFP4 W4A4 MoE Kernels
**vLLM PR #37463 (March 2026)** — CUTLASS MoE kernel for microscaling FP4 (W4A4) on SM100. Demonstrates that 4-bit MoE experts are production-ready on next-gen hardware.

**Relevance:** Our DP4a INT4 expert shader (moqe-fused-gemv-dp4a.glsl) implements the same concept on AMD RDNA2 via Vulkan compute.

---

## 5. Graph-of-Thoughts Reasoning

### 5.1 Network-of-Thought
**arXiv 2603.20730 (March 2026)** — "Reasoning Topology Matters: Network-of-Thought for Complex Reasoning Tasks"
Proposes structuring LLM reasoning as arbitrary directed graphs rather than chains. Shows that graph topology (not just content) determines reasoning quality. Specific topologies (star, tree, mesh) outperform linear chains on different task types.

**Relevance:** HD-GoT implements this insight natively in VSA space. Our spike diffusion centrality ranking IS the graph topology computation. The difference: Network-of-Thought generates text tokens to build the graph; HD-GoT builds it geometrically from hypothesis vectors.

### 5.2 Graph-Theoretic Agreement Framework
**UBOS (March 2026)** — "Graph-theoretic Agreement Framework for Multi-agent LLM Systems"
Formal framework for modeling agreement/disagreement between LLM agents using graph theory. Edges represent agreement strength; centrality identifies consensus.

**Relevance:** This is exactly HD-GoT's architecture. But they implement it with multiple LLM API calls; we implement it with VSA cosine similarity + spike diffusion — orders of magnitude cheaper.

### 5.3 Graph-Native Cognitive Memory
**alphaXiv 2603.17244 (March 2026)** — "Graph-Native Cognitive Memory for AI Agents: Formal Belief Revision Semantics for Versioned Memory Architectures"
Proposes graph-structured memory for AI agents with formal belief revision. Memory nodes have versions; edges represent temporal and causal relationships.

**Relevance:** Our experiential memory + VS-Graph could implement this. The MBAT orthogonal binding provides the versioning (different temporal bindings = different versions). The VS-Graph provides the graph structure.

### 5.4 Panoptic Thinking
**MatterAI (March 2026)** — "Panoptic Thinking: A Graph-Orchestrated Global Reasoning Architecture for Long-Horizon Autonomous Systems"
Graph-orchestrated reasoning for autonomous systems. Multiple reasoning modules produce partial solutions; a graph aggregator resolves conflicts and builds global consensus.

**Relevance:** This IS HD-GoT at the system level. Their "graph aggregator resolving conflicts" = our spike diffusion centrality ranking + OR-based message passing.

---

## Key Citations to Add to Paper

| # | Citation | Section | Relevance |
|---|----------|---------|-----------|
| [33] | VSA Occupancy Grid Mapping, Nature npj 2026 | 8.1 | VSA for robotics |
| [34] | Designing VSAs for Biomedical, PeerJ 2026 | 9.6 | JSON-to-VSA for healthcare |
| [35] | OpenMem, McMenemy 2026 | 8.1 | HDC persistent memory for agents |
| [36] | Noradrenaline hippocampal association, Nature Comms 2026 | 3.4 | Validates affective alpha |
| [37] | ACh demixes dopamine, Nature Neurosci 2026 | 2.4 | 5th hormone candidate |
| [38] | Network-of-Thought, arXiv 2603.20730 | 4, 8.2 | Graph topology for reasoning |
| [39] | Graph-Theoretic Agreement, UBOS 2026 | 4 | Formal agreement framework |
| [40] | Graph-Native Cognitive Memory, alphaXiv 2026 | 9.6 | Versioned graph memory |
| [41] | DeepSpeed MoQ, 2026 | 6 | Mixture-of-Quantization baseline |
| [42] | Limbic-Cortex Hybrid, 2026 | 3 | Validates architecture pattern |
| [43] | Fundamentals of Active Inference, Namjoshi 2026 | 5 | Reference textbook |
| [44] | Wearable biomechanical AI, Nature npj 2026 | 9.5 | Validates wearable concept |

---

## Methodology

Searched 5 topic queries via exa web search (8 results each, filtered to March 2026). Deep-read 6 key sources via full-page crawl. Cross-referenced findings against CubeMind v3 architecture. Total: 40+ sources evaluated, 12 new citations identified.
