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
| [08-vsa-lm.md](08-vsa-lm.md) | VSA-LM architecture: MinGRU hybrid backbone, VSA binding head, MindForge heads, H200 training protocol |
| [09-continuous-learning.md](09-continuous-learning.md) | Wake/sleep loop, neurochemistry, STDP, hippocampal replay, sandbox→live bridge |
| [10-mowm.md](10-mowm.md) | Mixture of World Models (MoWM): planning and RL track over the shared VSA infrastructure |

See also `docs/papers/cubemind_lm_h200_training.md` for the in-progress paper covering the
H200 training run that exercises the architecture described in 08 and 09.

## Reading Paths

**New contributor**: Start with 01, skim 02, then read the module(s) you are working on.

**Architect / reviewer**: Read all documents in order.

**Operations / deployment**: Read 01, then 06 (integration points), then 07 (migration status).

**ML researcher**: Read 01, 02 (VSA algebra), 08 (VSA-LM), 09 (continuous learning),
05 (MindForge + orchestration), skim 03 (Rust encoder). The companion paper
`docs/papers/cubemind_lm_h200_training.md` has the current H200 training results.
