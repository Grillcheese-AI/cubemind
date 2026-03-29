Here is the complete, unified Markdown document for your updated **MoWM v0.4.0 Design Spec**. It integrates the neuro-symbolic graph, the VSA latent space, the SymPy executor, the multimodal DenseNet/News layers, and the neuromorphic language realizer. 

You can copy and paste this directly into your `2026-03-28-mowm-v04-reasoning-decoder.md` file.

***

# MoWM v0.4.0 Design Spec — Neuro-Symbolic Reasoning Engine

**Date:** 2026-03-28  
**Author:** Nicolas Cloutier + Claude + Gemini  
**Status:** Draft  
**Module:** `mowm/` (extends v0.3.0)  
**Depends on:** v0.3.0 (QAPipeline, DirectSolver, AxiomLibrary, LogicEngine)

## Overview

v0.3.0 could answer formulaic questions directly but punted open-ended questions (EXPLAIN, COMPARE, GENERAL) to an external LLM. v0.4.0 replaces that dependency with a **Neuro-Symbolic Reasoning Engine**. 

MoWM now generates its own "intuition" via Vector Symbolic Architectures (VSA), proves its hypotheses using a deterministic symbolic math engine, and translates its reasoning into fluid speech using a highly constrained neuromorphic language layer. Furthermore, it introduces multimodal capabilities, allowing it to perceive live news and visual data and map them directly into its mathematical axiom graph.

This is a **knowledge compiler** paired with a **surface realizer**—it separates the *act of reasoning* from the *act of speaking*, entirely eliminating hallucinated facts.

## Architecture Flow

```text
Multimodal Input: "Why is the hurricane so destructive?" + [Live Radar Image]
    |
    v
1. Perception & Context Injection
   - DenseNet CNN -> Projects image into VSA space (Fluid Dynamics)
   - LiveNewsInjector -> Parses headlines -> Injects ephemeral causal edges
    |
    v
2. Knowledge Search (VSA + Graph)
   - Maps inputs to Semantic Graph nodes via cosine similarity.
   - Extracts localized subgraph (KnowledgeContext) with EdgeTypes (e.g., [CAUSES], [INCREASES]).
    |
    v
3. Reasoning & Proof (The "Brain")
   - CausalTracer: Finds opposing pathways (cancellation detection).
   - ChainBuilder: Explores logical graph paths.
   - SafeFormulaExecutor: Proves paths via SymPy (e.g., substituting equations).
    |
    v
4. Semantic Compilation
   - Composer compiles verified logic into a strict JSON SemanticPayload.
    |
    v
5. Surface Realization (The "Voice")
   - NeuromorphicLanguageLayer -> Converts payload to fluent English.
   - VocalTract (TTS) -> Synthesizes speech with mathematically-derived prosody.
```

## Module Structure

```text
mowm/
    reasoning/
        __init__.py                 # ReasoningDecoder Orchestrator
        graph_ontology.py           # EdgeType Enum, SemanticEdge, KnowledgeContext
        vsa_library.py              # Dense vector embeddings for concepts
        knowledge_search.py         # Subgraph extraction & VSA edge creation
        chain_builder.py            # Graph traversal for logical derivations
        causal_tracer.py            # Variable influence and cancellation detection
        comparator.py               # Structural formula diffing
        safe_executor.py            # SymPy-based symbolic math verifier
    perception/
        vision_projector.py         # DenseNet to VSA coordinate mapping
        news_injector.py            # Spacy NLP causal edge extraction
    generation/
        composer.py                 # Compiles math/logic into SemanticPayload
        neuromorphic_realizer.py    # Constrained language generation (SLM/SNN)
        vocal_tract.py              # TTS with prosody mapping
```

## Component Design

### 1. Graph Ontology (`graph_ontology.py`)
Defines the relationships between concepts, transforming the library into a Semantic Graph.

```python
class EdgeType(Enum):
    INCREASES = "INCREASES"
    DECREASES = "DECREASES"
    CAUSES = "CAUSES"
    IS_A = "IS_A"
    PROPORTIONAL_TO = "PROPORTIONAL_TO"
    DERIVATIVE_OF = "DERIVATIVE_OF"
    SIMILAR_TO_VEC = "SIMILAR_TO_VEC" # VSA latent edge

@dataclass
class SemanticEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

@dataclass
class KnowledgeContext:
    primary_axioms: list[Axiom]
    edges: list[SemanticEdge]
    adjacency_list: dict[str, list[SemanticEdge]]
```

### 2. Multimodal Perception (`perception/`)
Bridges unstructured real-world data to the rigid mathematical graph.

* **`VisionProjector`:** Uses a DenseNet CNN paired with a Contrastive Projection Matrix to convert image arrays (like charts or weather maps) into the 384D VSA vector space, allowing mathematical queries based on visual input.
* **`NewsInjector`:** Uses a lightweight dependency parser (spaCy) to extract subject-verb-object relationships from live news, mapping human verbs ("crush", "surge") to `EdgeType` relationships and injecting them as ephemeral graph nodes.

### 3. VSA & Knowledge Search (`vsa_library.py`, `knowledge_search.py`)
Replaces LLM semantic understanding with Vector Symbolic Architecture. Encodes every axiom's domain, description, and variables into a dense vector space. 

When answering open-ended `GENERAL` questions, it calculates cosine similarity to dynamically generate `SIMILAR_TO_VEC` edges, pulling highly relevant axioms without hardcoded links.

### 4. Logic & Proof (`causal_tracer.py`, `safe_executor.py`)
The deterministic core of the engine.

* **`CausalTracer`:** Uses Depth-First Search on the `adjacency_list` to find causal paths. It specifically looks for structural cancellations (e.g., when variable $A$ affects target $B$ via a `[INCREASES]` edge on one path, but a `[DECREASES]` edge on another).
* **`SafeFormulaExecutor`:** Wraps `SymPy` to securely parse and prove these hypotheses. It performs algebraic rearrangement and symbolic substitution (e.g., substituting $F=mg$ into $a=F/m$ to mathematically prove mass cancels out to $a=g$).

### 5. Surface Realization (`generation/`)
Separates the "thinking" from the "speaking" to eliminate hallucinations.

```python
@dataclass
class SemanticPayload:
    intent: str
    core_claim: str
    causal_chain: list[dict]
    math_proof: list[str]
    tone: str
```

* **`Composer`:** Compiles the logical proofs into a strict JSON `SemanticPayload`.
* **`NeuromorphicLanguageLayer`:** A highly distilled, constrained sequence-to-sequence model (or SNN). It has a strict system prompt to *only* translate the `SemanticPayload` into fluent English, adding syntactic variety without inventing facts.
* **`VocalTract`:** Maps the logical intent to TTS prosody (e.g., `BREAKING_NEWS` intent generates a faster, more urgent voice profile).

## Example Flow: "Why do heavier objects fall at the same rate?"

1.  **Search:** Pulls nodes for `Mass`, `Gravity`, `Acceleration`.
2.  **Causal Trace:** Graph detects $m$ has a `[PROPORTIONAL_TO]` path to force, and an `[INVERSELY_PROPORTIONAL]` path to acceleration.
3.  **Proof:** `SafeFormulaExecutor` isolates $a$ via substitution: $a = (m \cdot g)/m \rightarrow a = g$.
4.  **Payload:** `Composer` generates strict JSON stating mass cancels out.
5.  **Language:** `NeuromorphicLayer` outputs: *"Because mass increases gravitational pull but equally increases resistance to acceleration, the two effects cancel each other out mathematically. When we substitute the equations, mass drops out entirely, leaving us with exactly $a=g$ for all objects."*

## Key Design Decisions

1.  **Neuro-Symbolic Split:** The neural networks (VSA, DenseNet, SLM) handle *perception* and *language*. The symbolic graph (SymPy, Dijkstra's) handles *truth and logic*.
2.  **No Hallucinations by Design:** Because the language model only sees a rigid JSON payload of mathematically proven equations, it cannot confidently invent false physical laws.
3.  **Real-Time Context:** By isolating news and vision as "ephemeral edges," the core axiom library remains mathematically pristine while still reasoning about the present moment.

***
