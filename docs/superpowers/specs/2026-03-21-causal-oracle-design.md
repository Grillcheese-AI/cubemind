# Decision Oracle — Causal Training & Interactive Decision Tree

**Date:** 2026-03-21
**Status:** Draft
**Goal:** Ground the Decision Oracle in real-world causal data from 5 sources (~7M+ records), enabling interactive branching exploration of plausible futures.

---

## 1. Problem

The Decision Oracle currently operates cold — random personality vectors, untrained HYLA, placeholder Q-values. It has no grounding in real-world facts or causal relationships. To be useful for the cloud API demo and beyond, it needs to:

1. Understand cause → effect relationships from historical data
2. Predict plausible futures grounded in real-world patterns
3. Present predictions interactively as a branching decision tree
4. Cite supporting evidence from the corpus

---

## 2. Data Sources

| Source | Records | Location | Key Value |
|--------|---------|----------|-----------|
| Qdrant corpus | 5.3M | localhost:6333 collection `corpus` | 384-dim MiniLM embeddings, 46 categories — codebook learning base |
| Historical events | 49K + 122K precursor links | `C:\Users\grill\Desktop\GrillCheese\data_learning\temporal\historical\epub_pdf_consolidated.json` | Explicit causal chains, quantum metrics, sentiment, emotion |
| NYT archive | ~2M articles (294 files) | `D:\grillcheese_training_data\temporal\nyt_data` | 173 years of dated events (1851–2024), entity-tagged (persons/orgs/glocations/subjects) |
| Movie annotations | 365 scenes / 5 films | `H:\AURA_GENESIS\datasets\movie_annotated` | Emotion timelines, Plutchik scores, narrative causality |
| Knowledge texts | 1,086 books/docs (2.5 GB) | `C:\Users\grill\Documents\GitHub\grillcheese\training_data\knowledgetxt` | Deep domain knowledge (history, military, law, psychology, science) |

### 2.1 Historical Events Schema

Each record has 34 fields:
- `event_id`, `source_filename`, `source_text`, `title`, `summary`
- `dates`, `cleaned_dates`, `earliest_date_year`, `latest_date_year`, `date_span_years`, `num_dates_mentioned`, `has_date_range`
- `category` (14 types: war_conflict, political_event, historical_event, etc.)
- `event_type` (300+ fine-grained types)
- `confidence`
- `entities` (JSON with PERSON, ORG, GPE, LOC), plus `person_count`, `org_count`, `gpe_count`, `loc_count`
- `sentiment` (POSITIVE/NEGATIVE/NEUTRAL), `sentiment_score`
- `emotion`, `emotion_score`
- `q_entropy`, `q_coherence`, `q_interference`, `q_entanglement` (quantum metrics)
- `precursor_events` — list of `{description, year_parsed, month_parsed, season_parsed}` (avg 3.8 per record, 65% coverage)

### 2.2 NYT Archive Schema

- `headline` (main, kicker), `abstract`, `snippet`, `lead_paragraph`
- `pub_date` (ISO datetime, precise to day)
- `keywords` — typed: `persons`, `organizations`, `glocations`, `subject`, `creative_works`
- `section_name` (16 sections), `document_type`, `type_of_material`
- `web_url`, `_id`, `uri`, `word_count`

### 2.3 Movie Annotations Schema

- `movie_title`, `scene_number`, `original_scene_text`, `full_dialogue_context`
- `main_base_emotion`, `plutchik_score`, `plutchik_label`
- `emotion_timeline` — ordered list of emotional states through scene
- `annotated_dialogue` — keyword-tagged dialogue lines

---

## 3. Architecture — Four Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: INTERACTIVE DECISION TREE                         │
│  Prompt → futures → user selects → branch → repeat         │
│  Backtrack, depth limit, grounding citations                │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: CAUSAL GRAPH ENGINE                               │
│  Directed temporal graph — tiered edge linking              │
│  Training: direct pairs → graph walks → Q-values            │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: FACTORED CODEBOOK (80 axes × 128 values)          │
│  32 explicit axes (LLM-extracted attributes)                │
│  48 learned axes (PCA on corpus embeddings)                 │
│  Every record → native (80, 128) discrete block-code        │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: UNIFIED DATA LAKE + LLM AUGMENTATION              │
│  5 sources → normalized event schema                        │
│  LLM extracts 32 structured attributes per record           │
│  All embedded in Qdrant (384-dim MiniLM)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Layer 1 — Unified Data Lake + LLM Augmentation

### 4.1 Normalized Event Schema

All five sources are mapped to a common schema:

```python
@dataclass
class UnifiedEvent:
    event_id: str
    text: str                    # primary text content
    date: datetime | None        # precise date when available
    year: int | None             # fallback year
    category: str                # normalized category
    entities: dict[str, list[str]]  # {PERSON: [], ORG: [], GPE: [], LOC: []}
    source_type: str             # "corpus" | "historical" | "nyt" | "movie" | "knowledge"
    source_ref: str              # original file/ID
    precursors: list[str]        # event_ids of causal predecessors
    attributes: dict[str, float] # 32 LLM-extracted attributes (populated in augmentation)
```

### 4.2 LLM Attribute Extraction (32 Explicit Axes)

Each record is enriched with 32 structured attributes via LLM:

**Causality (axes 1–6):**
1. `reversibility` — can this be undone? (0.0 = irreversible, 1.0 = fully reversible)
2. `agency_type` — who/what drives this? (quantized: human=0.0, institutional=0.33, systemic=0.66, natural=1.0)
3. `chain_length_potential` — how many downstream effects? (0.0 = terminal, 1.0 = massive cascade)
4. `cause_certainty` — how certain is the causal link? (0.0 = speculative, 1.0 = established fact)
5. `effect_magnitude` — scale of the effect (0.0 = negligible, 1.0 = civilization-altering)
6. `feedback_loop` — does the effect reinforce its cause? (0.0 = no, 1.0 = strong positive feedback)

**Temporal (axes 7–12):**
7. `urgency` — time pressure (0.0 = no deadline, 1.0 = immediate)
8. `duration` — how long does this last? (0.0 = momentary, 1.0 = permanent)
9. `periodicity` — does this recur? (0.0 = one-time, 1.0 = cyclical)
10. `temporal_distance` — how far in the future are effects felt? (0.0 = immediate, 1.0 = generational)
11. `decay_rate` — how fast does relevance fade? (0.0 = never fades, 1.0 = instant decay)
12. `recurrence_risk` — probability of happening again (0.0 = unique, 1.0 = near-certain recurrence)

**Impact (axes 13–18):**
13. `geographic_scope` — spatial reach (0.0 = local, 0.33 = regional, 0.66 = continental, 1.0 = global)
14. `population_scale` — people affected (0.0 = individual, 1.0 = billions)
15. `domain_breadth` — how many domains touched (0.0 = single domain, 1.0 = all domains)
16. `institutional_depth` — government/institution involvement (0.0 = none, 1.0 = deep structural)
17. `economic_weight` — economic significance (0.0 = none, 1.0 = massive)
18. `cultural_reach` — cultural/social significance (0.0 = none, 1.0 = defining)

**Counterfactual (axes 19–24):**
19. `n_plausible_alternatives` — how many other outcomes were possible? (0.0 = inevitable, 1.0 = many paths)
20. `pivotality` — did this event change the trajectory? (0.0 = no change, 1.0 = inflection point)
21. `contingency` — how dependent on chance? (0.0 = deterministic, 1.0 = pure chance)
22. `path_dependence` — does this lock in future paths? (0.0 = no, 1.0 = strong lock-in)
23. `fragility` — how easily could this have not happened? (0.0 = robust/inevitable, 1.0 = extremely fragile)
24. `determinism_score` — overall predictability (0.0 = chaotic, 1.0 = fully predictable)

**Narrative (axes 25–30):**
25. `tension_level` — dramatic tension (0.0 = calm, 1.0 = peak crisis)
26. `stakes` — what's at risk? (0.0 = nothing, 1.0 = existential)
27. `resolution_type` — how does this resolve? (0.0 = unresolved, 0.5 = compromise, 1.0 = decisive)
28. `moral_complexity` — ethical nuance (0.0 = clear right/wrong, 1.0 = deep moral ambiguity)
29. `perspective_count` — how many valid viewpoints? (0.0 = one, 1.0 = many competing)
30. `ambiguity` — clarity of interpretation (0.0 = unambiguous, 1.0 = highly contested)

**Semantic (axes 31–32):**
31. `category` — normalized event category (quantized across all sources)
32. `event_type` — fine-grained event classification

**Learned (axes 33–80):**
48 axes derived from PCA on 384-dim MiniLM corpus embeddings. These capture latent semantic dimensions not covered by explicit attributes.

### 4.3 Augmentation Pipeline

- **Test (1,000 records):** OpenRouter API — structured JSON extraction, ~$1–3 total cost
- **Scale (7M+ records):** GGUF model on Colab A100/G4 — batch processing, ~500–1000 records/min

LLM extraction prompt outputs structured JSON per record:
```json
{
  "reversibility": 0.2,
  "agency_type": 0.66,
  "chain_length_potential": 0.8,
  "urgency": 0.9,
  "geographic_scope": 0.66,
  "tension_level": 0.85,
  "..."
}
```

---

## 5. Layer 2 — Factored Codebook

### 5.1 Codebook Construction

1. **PCA training:** Sample ~100K records from the 5.3M corpus. Extract 384-dim embeddings from Qdrant. Run PCA → top 48 principal components. These become axes 33–80.
2. **Quantization:** For each of the 80 axes (32 explicit + 48 learned), apply vector quantization to create 128 discrete bins. Each bin corresponds to one-hot position in that block.
3. **Encoding:** Any record → project onto 80 axes → quantize each → (80, 128) discrete block-code.

### 5.2 Integration with Existing Code

- Extends `WorldEncoder` with `encode_from_attributes(attributes: dict) → np.ndarray`
- New `CausalCodebook` class handles PCA fitting, VQ bin boundaries, encode/decode
- Compatible with all existing `BlockCodes` operations (bind, unbind, similarity, bundle)

---

## 6. Layer 3 — Causal Graph + Oracle Training

### 6.1 Graph Construction

Nodes = unified events. Edges = causal links with three tiers:

| Tier | Criteria | Edge Weight |
|------|----------|-------------|
| **Strong** | Explicit precursor link (historical dataset) | 1.0 |
| **Medium** | Shared entities + same category + temporal proximity (<5 years) | 0.6 |
| **Weak** | Semantic similarity (Qdrant top-k) + temporal order | 0.3 |

NYT articles with shared `keywords.persons` or `keywords.organizations` within temporal windows become bridge nodes linking historical events.

Movie emotion timelines create sequential edges within each film (scene N → scene N+1) for narrative causality training.

### 6.2 Training Phases

**Phase 1 — Direct Pair Training:**
- Each (precursor → event) = (cause_state, effect_state) block-code pair
- Train HYLA hypernetwork: given cause block-code, predict effect block-code delta
- Train CVL: plausibility of predicted future vs actual outcome as reward signal

**Phase 2 — Graph Walk Training:**
- Random walks along causal chains (2–5 hops)
- Oracle learns multi-step prediction
- CVL Q-values trained on trajectory rewards (higher confidence/coherence events = better outcomes)
- Movie emotion trajectories provide narrative path training

**Phase 3 — Contrastive Training:**
- Positive pairs: real causal chains from data
- Negative pairs: temporally shuffled or cross-domain non-causal pairs
- Sharpens the Oracle's ability to distinguish plausible from implausible futures

---

## 7. Layer 4 — Interactive Decision Tree

### 7.1 User Flow

```
User prompt: "What if the EU passes strict AI regulation in 2026?"
    │
    ▼
WorldEncoder + AttributeExtractor → (80, 128) state block-code
    │
    ▼
Decision Oracle (128 parallel futures, top-k=5)
    │
    ▼
FutureDecoder → 5 plausible futures in natural language
    │
    ├── 1. Tech companies relocate R&D...          (plausibility: 0.82)
    ├── 2. Open-source AI accelerates...            (plausibility: 0.74)
    ├── 3. EU becomes global standard setter...     (plausibility: 0.71)
    ├── 4. Innovation slowdown in regulated...      (plausibility: 0.65)
    └── 5. Two-tier AI market emerges...            (plausibility: 0.58)
    │
User selects #3
    │
    ▼
Selected future → new state → Oracle branches → 5 new futures
    │
    ├── 3.1 China matches EU rules for trade...     (plausibility: 0.79)
    ├── 3.2 US tech lobby pushes back...            (plausibility: 0.73)
    └── ...
    │
User selects 3.2 → depth=3 → continues...
```

### 7.2 Core Data Structures

```python
class DecisionTree:
    root: TreeNode           # initial prompt state
    current: TreeNode        # where the user is now
    history: list[TreeNode]  # path taken (for backtrack)

class TreeNode:
    state: np.ndarray        # (80, 128) block-code
    prompt_text: str         # NL description
    futures: list[Future]    # Oracle predictions at this node
    selected: int | None     # which future the user picked
    parent: TreeNode | None
    children: list[TreeNode]
    depth: int

class Future:
    state: np.ndarray        # predicted future block-code
    description: str         # NL readable prediction
    plausibility: float      # Oracle score
    q_value: float
    grounding: list[str]     # corpus records that support this prediction
```

### 7.3 Key Behaviors

- **Branch:** User selects a future → becomes new state → Oracle generates new futures
- **Backtrack:** User returns to any previous node and picks a different path
- **Ground:** Each future cites supporting corpus records (historical events, NYT articles)
- **Depth limit:** Configurable, default 10 levels
- **Export:** Full tree serializable as JSON for the cloud API

### 7.4 NL Translation (FutureDecoder)

- **Test/MVP:** Template-based — map 32 explicit axes back to NL fragments
- **Production:** Local LLM decodes block-code attributes + grounding records into natural language descriptions

### 7.5 API Endpoints

Update `cubemind/cloud/api.py`:
- `POST /predict` — initial prompt → returns tree root with top-k futures
- `POST /choose` — select a future → returns new node with branched futures
- `POST /backtrack` — return to previous node
- `GET /tree/{session_id}` — full tree state

---

## 8. New Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `DataNormalizer` | `cubemind/execution/data_normalizer.py` | Unify 5 sources → `UnifiedEvent` schema |
| `AttributeExtractor` | `cubemind/execution/attribute_extractor.py` | LLM prompt + JSON schema for 32 attributes (OpenRouter + GGUF) |
| `CausalCodebook` | `cubemind/execution/causal_codebook.py` | PCA + VQ pipeline, 384-dim → (80, 128) block-codes |
| `CausalGraph` | `cubemind/execution/causal_graph.py` | Directed temporal graph, tiered linking, graph walks |
| `OracleTrainer` | `cubemind/execution/oracle_trainer.py` | Direct pairs + graph walks + contrastive → HYLA/CVL training |
| `DecisionTree` | `cubemind/execution/decision_tree.py` | Interactive tree state, branch/backtrack/export |
| `FutureDecoder` | `cubemind/execution/future_decoder.py` | Block-code → NL description (template + LLM modes) |
| Tests | `tests/test_causal_codebook.py`, `tests/test_causal_graph.py`, `tests/test_decision_tree.py` | End-to-end on test dataset |

---

## 9. Test-First Validation Plan

| Step | What | Size | Validates |
|------|------|------|-----------|
| 1 | Select 1,000 historical events with good precursor chains | 1,000 records | Data selection criteria |
| 2 | Run LLM attribute extraction via OpenRouter | 1,000 enriched records | Extraction prompt quality, attribute distributions |
| 3 | Build codebook from 10K corpus embedding sample | 10K embeddings | PCA axis quality, VQ bin distribution |
| 4 | Encode 1,000 events → block-codes | 1,000 block-codes | Codebook fidelity, similarity preservation |
| 5 | Build mini causal graph | ~1,000 nodes, ~3,800 edges | Graph construction, tiered linking |
| 6 | Train Oracle on mini-graph (direct pairs + walks) | Training run | HYLA learns cause→effect, CVL Q-values converge |
| 7 | Interactive decision tree on 3 test prompts | Manual evaluation | End-to-end: prompt → plausible futures → branch |
| 8 | Scale decision: proceed to full 7M+ pipeline | Go/no-go | Quality threshold met? |

---

## 10. Scale Pipeline (After Test Validation)

1. **Augment full datasets** — GGUF on Colab A100/G4, batch all 7M+ records
2. **Rebuild codebook** — PCA on full 5.3M corpus, VQ on enriched attributes
3. **Construct full graph** — all sources linked with tiered edges
4. **Train Oracle** — full multi-phase training on complete graph
5. **Deploy** — updated cloud API with interactive decision tree endpoints

---

## 11. Dependencies

- `qdrant-client` — Qdrant vector search (already in use via elephant-coder)
- `sentence-transformers` — MiniLM embeddings for new records
- `scikit-learn` — PCA + KMeans/VQ for codebook
- `networkx` — causal graph construction and traversal
- `httpx` or `openai` — OpenRouter API calls for test augmentation
- Existing: `BlockCodes`, `HYLA`, `CVL`, `WorldEncoder`, `DecisionOracle`
