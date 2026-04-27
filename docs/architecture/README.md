# CubeMind Architecture Documentation

This directory contains comprehensive technical architecture documentation for the CubeMind
multi-language system.

## Documents

| File | Description |
|------|-------------|
| [01-overview.md](01-overview.md) | Executive summary and system overview |
| [02-vsa-foundations.md](02-vsa-foundations.md) | VSA mathematical foundations shared across all three runtimes |
| [03-rust-engine.md](03-rust-engine.md) | opcode-vsa-rs: the Rust compute engine |
| [04-cpp-vulkan-grilly.md](04-cpp-vulkan-grilly.md) | grilly: C++/Vulkan GPU acceleration layer |
| [05-python-orchestration.md](05-python-orchestration.md) | CubeMind Python: orchestration, training, MindForge |
| [06-polyglot-integration.md](06-polyglot-integration.md) | Cross-language integration, protobuf design, and Go API plan |
| [07-migration-roadmap.md](07-migration-roadmap.md) | Migration path from Python-heavy to Rust-centric architecture |
| [08-cubemind-lm.md](08-cubemind-lm.md) | CubeMind-LM (MinGRU hybrid): HybridBlock, VSABindingHead, MindForgeLoRAHead, Heinsen scan, three-stage H200 training protocol |
| [09-continuous-learning.md](09-continuous-learning.md) | Wake/sleep loop, neurochemistry, STDP, hippocampal replay, LiveAdapter sandbox→live bridge, stage 1.5 corpora |
| [10-mowm.md](10-mowm.md) | Mixture of World Models (MoWM): planning and RL track over the shared VSA infrastructure |

### C4 model diagrams (Mermaid)

| File | Level | Description |
|------|-------|-------------|
| [c4-context.md](c4-context.md) | 1 — Context | CubeMind + sibling repos (grilly / opcode-vsa-rs / cubelang / optimum-grilly) + external systems (H200 RunPod, HuggingFace Hub, AMD RX 6750 XT) |
| [c4-containers.md](c4-containers.md) | 2 — Container | Internal containers (Orchestrator, VSA-VM, MindForge, VSA-LM trainer, Sandbox MinGRU trainer, LiveAdapter, FastAPI, Hippocampus, Web UI) + external integrations |
| [c4-components-vsa-lm.md](c4-components-vsa-lm.md) | 3 — Component | CubeMind-LM hybrid stack: TokenEmbedding → HybridBlock ×12 (MinGRU / MoE / LocalAttn / Memory / GLU / RMSNorm) + SurpriseTracker → VSABindingHead + 5 × MindForgeLoRAHead (file name predates rename; content is current) |
| [c4-dynamic-training.md](c4-dynamic-training.md) | Dynamic | H200 stage-1 training flow per 500-step eval interval (16 numbered steps) |

See also `docs/papers/cubemind_lm_h200_training.md` for the in-progress paper covering the
H200 training run that exercises the architecture described in 08 and 09.

## Reading Paths

**New contributor**: Start with 01, skim 02, then read the module(s) you are working on.

**Architect / reviewer**: Read all documents in order.

**Operations / deployment**: Read 01, then 06 (integration points), then 07 (migration status).

**ML researcher**: Read 01, 02 (VSA algebra), 08 (CubeMind-LM), 09 (continuous learning),
05 (MindForge + orchestration), skim 03 (Rust encoder). The companion paper
`docs/papers/cubemind_lm_h200_training.md` has the current H200 training results.
