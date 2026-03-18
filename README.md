# CubeMind v2

Neuro-vector-symbolic architecture for compositional reasoning on consumer hardware.

Built on [grilly](https://github.com/Grillcheese-AI/grilly) — GPU-accelerated via Vulkan compute shaders.

## Install

```bash
pip install grilly cubemind
```

## Architecture

```
Input → Perception → Routing → Memory → Detection → Execution → Answer
         (VSA)       (MoE)    (Cache)    (HMM)      (HYLA+CVL)  (Decode)
```

97.5% accuracy on iRaven with interpretable block-code reasoning.
