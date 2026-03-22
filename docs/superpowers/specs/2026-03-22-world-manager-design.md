# WorldManager — Self-Organizing Specialist World Models

**Date:** 2026-03-22
**Status:** Draft
**Goal:** Replace fixed 128 random personality vectors with dynamically spawning, self-organizing specialist world models that emerge from historical data via anomaly detection and Oja's plasticity consolidation.

---

## 1. Problem

The Decision Oracle currently uses 128 random personality vectors to generate diverse futures. These personalities are meaningless — they don't represent real-world causal patterns. The result is diverse but ungrounded predictions.

We need world models that are **specialists** — each one an expert in a specific type of causal pattern (economic cascades, military escalation, technological disruption, cultural shifts). These specialists must:

1. Emerge from data, not be hardcoded
2. Self-organize via anomaly detection (spawn new specialist when OOD detected)
3. Self-refine via Oja's plasticity (sharpen over repeated observations)
4. Be human-readable (VSA Translator probes what each specialist learned)
5. Support counterfactual reasoning (remove entities, simulate alternate timelines)

---

## 2. Architecture Overview

```
Historical JSONL Events
        |
        v
JSON-to-VSA Encoder
  (entities, relations, affect_tags, causal_links -> block-codes)
        |
        v
WorldManager (Pre-allocated VRAM Arena)
  |
  |-- For each event transition:
  |     1. Extract rule: R_obs = unbind(State_t+1, State_t)
  |     2. Compare R_obs against all active specialists (batch similarity)
  |     3. max_sim < tau (0.65)? -> SPAWN new specialist
  |        max_sim >= tau?       -> ROUTE to best match + Oja consolidation
  |
  |-- Result: N self-organized specialist world models
        |
        v
VSA Translator ("Particle Accelerator")
  (probe each specialist with known codebook -> English descriptions)
        |
        v
Decision Oracle (uses specialists instead of random personalities)
  (DSelect-k gating selects which specialists apply per query)
```

---

## 3. Data Pipeline

### 3.1 Generate ~15K Historical Events

Batch prompt to Gemini Flash Lite / OpenRouter, 30-50 year windows:

```
"Generate 100 real historical events from {year_start} to {year_end} as JSONL.
Each event must have: event_id, title, summary, source_text,
earliest_date_year, latest_date_year, category, event_type,
sentiment, sentiment_score, energy, pleasantness,
latitude, longitude, participants [{name, entity_type}],
affect_tags [{id, score}], topic_tags [str],
precursor_events [{description, year, causal_strength}],
causal_link {effect_score, influence_factors [{factor, description, score}],
  next_event_summary},
similar_events [{event_summary, reasoning, similarity_score}]"
```

Coverage: -5000 BCE to 2026, ~150 windows x 100 events = ~15,000 events.
Pre-1000 BCE: 50 events per window (sparser records).
Cost: ~$15 on Gemini Flash Lite.

### 3.2 Convert Existing 49K Records

Transform epub_pdf_consolidated.json records into the rich JSONL format.
Fields already present: event_id, summary, dates, category, entities, precursors, sentiment, emotion, quantum metrics.
Fields to add via LLM enrichment: causal_link, similar_events, affect_tags, energy, pleasantness.

### 3.3 JSON-to-VSA Encoder

For each event, build a state vector by binding entity-role pairs and bundling:

```
V_event = (Entity_1 * Role_1) + (Entity_2 * Role_2) + ... + (w_1 * Tag_1) + (w_2 * Tag_2)
```

Where:
- Entities from `participants` bound with their `entity_type` role
- Tags from `affect_tags` scaled by their `score` before bundling
- `energy` and `pleasantness` encoded as quantized block-code dimensions
- `topic_tags` bound with a `HasTopic` role vector

Causal edges encoded as directed bindings:
```
E = effect_score * (V_event * Causes * V_downstream)
```

Influence factors add weighted sub-edges:
```
E_factor = factor_score * (V_event * InfluenceFactor * V_factor_concept)
```

---

## 4. WorldManager

### 4.1 Pre-Allocated Arena

On boot, allocate a buffer for max_worlds (default 1024) specialist vectors:
- Buffer size: 1024 x D_VSA x 4 bytes = 1024 x 10240 x 4 = ~40 MB
- Active counter: `uint32_t active_worlds = 0`
- No reallocation ever needed

### 4.2 Anomaly Detection (The Trigger)

For each historical transition (event_t -> event_t+1):

1. Extract the transformation rule:
   ```
   R_observed = unbind(State_t+1, State_t)
   ```

2. If `active_worlds == 0`: spawn first specialist, return.

3. Compute batch similarity of R_observed against all active specialists.

4. Find max_sim and best_world_idx.

### 4.3 The tau Threshold (tau = 0.65)

Mathematical derivation for block-codes with k blocks of length l:

- **Random noise floor:** mu = 1/l (for l=128: ~0.0078)
- **Noisy correct match (6/8 blocks correct):** ~0.753
- **Compositional overlap:** 0.30-0.50
- **Decision boundary:** tau = 0.65

This provides massive safety margin above noise and compositional overlap, while tolerating perception noise.

### 4.4 Creation Event (Spawn New Specialist)

When max_sim < tau:
```
1. Copy R_observed into arena at offset active_worlds * D_VSA
2. active_worlds++
3. Log: "Anomalous pattern detected. Specialist {N} instantiated."
```

Zero allocation — just a memcpy into the pre-allocated buffer.

### 4.5 Consolidation (Oja's Plasticity)

When max_sim >= tau (rule recognized):
```
1. Route to best matching specialist
2. Apply Oja's learning rule:
   w_new = w_old + eta * y * (x - y * w_old)
   where:
     x = R_observed (the new observation)
     y = similarity score (the activation)
     w_old = current specialist vector
     eta = learning rate (0.01)
3. Normalize w_new to unit length per block
```

Over repeated observations, each specialist crystallizes into a mathematically pure representation of its causal pattern. Noisy first observations get refined. This is the biological "sleep cycle" analog.

---

## 5. VSA Translator ("Particle Accelerator")

After training, probe each specialist to generate human-readable descriptions.

### 5.1 Identity Check (Constant Rule)

For each concept in codebook:
```
Out = bind(Concept, W_specialist)
Match = argmax similarity(Out, codebook)
If Match == Concept: "Constant {concept_name}"
```

### 5.2 Shift Check (Progression Rule)

```
Out = bind(Size_2, W_specialist)
If best match is Size_3:
  Out2 = bind(Size_3, W_specialist)
  If best match is Size_4: "Progression (+1) Size"
```

### 5.3 Compositional Rules

When probing reveals multiple attribute changes:
```
Probe(Economic_Stability) -> Economic_Crisis
Probe(Military_Tension) -> Military_Conflict
>> "Economic destabilization triggers military escalation"
```

### 5.4 Output Format

```
[VSA Translator] Analyzing N Discovered Specialists...

--- Specialist 0 ---
Probe(Economic_Growth) -> Economic_Crisis (sim: 0.89)
Probe(Trade_Volume) -> Trade_Decline (sim: 0.85)
Probe(Military_Tension) -> Military_Tension (sim: 0.92, constant)
>> English: "Economic collapse pattern — growth -> crisis, trade declines, military unchanged"

--- Specialist 1 ---
Probe(Military_Tension) -> Military_Conflict (sim: 0.91)
Probe(Diplomatic_Relations) -> Diplomatic_Breakdown (sim: 0.88)
Probe(Economic_Growth) -> Economic_Growth (sim: 0.94, constant)
>> English: "Military escalation pattern — tension -> conflict, diplomacy breaks, economy unchanged"
```

---

## 6. Counterfactual Simulation

### 6.1 State Construction

Bundle all facts about a time period into a single state vector:
```
W_1939 = (Germany * RuledBy * Nazi_Party) +
         (Germany * AtWarWith * Poland) +
         (Britain * AlliedWith * France) + ...
```

### 6.2 Extract Era Physics

```
R_era = unbind(W_1945, W_1939)
```

This single vector captures the mathematical signature of WWII.

### 6.3 Counterfactual Query

"What if the US never entered WWII?"
```
W_1941_alt = W_1941 - (US * AtWarWith * Axis)
W_1945_alt = bind(W_1941_alt, R_era)
```

### 6.4 Probe Alternate Timeline

```
Probe(Germany * Status) in W_1945_alt
-> If "Active" scores higher than "Defeated": "Without US entry, Germany survives longer"
```

The binding interference math makes this work: removing a key entity disrupts the causal chain for outcomes that were entangled with that entity.

---

## 7. Integration with Decision Oracle

### 7.1 Replace Random Personalities

Current: `self.world_personalities = [bc.random_discrete(seed=i) for i in range(128)]`

New: `self.world_personalities = world_manager.get_active_specialists()`

The number of specialists is dynamic (not fixed at 128). Could be 15, could be 200 — depends on what the data reveals.

### 7.2 DSelect-k Gating

When evaluating futures for a user query, the gate selects which specialists are relevant:
```
gate_weights = DSelect_k(state, action, specialists)
# Only top-k specialists generate futures
# Each specialist predicts through its own causal lens
```

### 7.3 Decision Tree with Specialist Attribution

Each future in the decision tree now carries which specialist(s) generated it:
```
Future {
  state: block-code
  description: "Economic destabilization leads to trade collapse"
  specialist: "Specialist 3: Economic collapse pattern"
  plausibility: 0.82
  grounding: ["Great Depression 1929", "Asian Financial Crisis 1997"]
}
```

---

## 8. New Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `WorldManager` | `cubemind/execution/world_manager.py` | Pre-allocated arena, anomaly detection, spawn/consolidate |
| `OjaConsolidator` | `cubemind/execution/oja_consolidator.py` | Oja's learning rule for specialist refinement |
| `VSATranslator` | `cubemind/execution/vsa_translator.py` | Probe specialists against codebook, generate English |
| `EventEncoder` | `cubemind/execution/event_encoder.py` | JSONL -> block-code state vectors with causal edges |
| `HistoricalGenerator` | `scripts/generate_historical_events.py` | Batch LLM prompt to generate JSONL events |
| `RecordConverter` | `scripts/convert_epub_to_jsonl.py` | Transform 49K epub records to rich JSONL |
| `CounterfactualEngine` | `cubemind/execution/counterfactual.py` | State subtraction + era physics for alt-timeline queries |
| Tests | `tests/test_world_manager.py`, etc. | TDD for all components |

---

## 9. Empirical Validation Against Original Benchmarks

The WorldManager must prove that self-organizing specialists match or exceed the handcoded rule detectors from the original CubeMind paper. This validates the architecture for the NeurIPS submission.

### 9.1 I-RAVEN Benchmark (Original Paper Results)

Feed RAVEN problems to the WorldManager instead of the hardcoded constant/progression/arithmetic/distribute-three detectors. The WorldManager should:

1. **Discover the 4 rule types automatically** — by processing panel transitions, the system should spawn ~4 specialists that correspond to constant, progression, arithmetic, and distribute-three.
2. **VSA Translator confirms** — probing the discovered specialists should produce descriptions matching the known rule types.
3. **Target: >= 90.3% overall accuracy** (matching the paper's zero-shot result).

| Configuration | Paper Result | WorldManager Target |
|---------------|-------------|-------------------|
| Center Single | 97.5% | >= 95% |
| Left-Right | 98.0% | >= 95% |
| Up-Down | 96.0% | >= 93% |
| Out-InCenter | 100.0% | >= 97% |
| Grid configs | 82.0% | >= 78% |
| **Overall** | **90.3%** | **>= 88%** |

A small accuracy drop is acceptable since the specialists are discovered, not engineered. If they match or exceed, that's a headline result.

### 9.2 I-RAVEN-X Out-of-Distribution (100% Zero-Shot)

The critical test: does the WorldManager maintain **perfect OOD generalization**?

Feed I-RAVEN-X problems at maxval=10, 100, 1000. The discovered specialists operate on algebraic relationships (equality, arithmetic difference, set membership) — they should be invariant to operand magnitude just like the handcoded detectors.

| OOD Factor | Paper Result | WorldManager Target |
|-----------|-------------|-------------------|
| 1x (maxval=10) | 98.5% | >= 96% |
| 10x (maxval=100) | 99.8% | >= 98% |
| 100x (maxval=1000) | 100.0% | >= 99% |

If the WorldManager achieves >= 99% at 100x OOD, it proves that self-organized algebraic specialists generalize as well as handcoded ones — without any human engineering of the rule detectors.

### 9.3 Historical Causal Validation

After training on 15K+ historical events:

1. **Specialist count** — expect 20-80 specialists to emerge across economic, military, technological, diplomatic, cultural, scientific patterns.
2. **Specialist coherence** — VSA Translator descriptions should be semantically meaningful, not noise.
3. **Counterfactual sanity** — known historical counterfactuals should produce plausible alternate timelines:
   - "What if the printing press was never invented?" -> delayed Renaissance, slower scientific revolution
   - "What if Rome never fell?" -> delayed feudalism, continued centralized governance
   - "What if the Industrial Revolution started in China?" -> shifted global power dynamics
4. **Cross-temporal analogy** — specialists trained on ancient patterns should activate for modern events with similar dynamics (e.g., "economic collapse" specialist fires for both 1929 and 2008).

### 9.4 Ablation Study

| Configuration | Expected Result |
|--------------|----------------|
| WorldManager (tau=0.65) | Baseline: N specialists, accuracy X% |
| WorldManager (tau=0.50) | More specialists (finer-grained), slight accuracy change |
| WorldManager (tau=0.80) | Fewer specialists (coarser), possible accuracy drop |
| WorldManager without Oja | Noisier specialists, lower accuracy |
| WorldManager with Oja (eta=0.001 vs 0.01 vs 0.1) | Optimal consolidation rate |
| Handcoded detectors (paper baseline) | 90.3% / 100% OOD |

---

## 10. Test-First Implementation Validation

| Step | What | Validates |
|------|------|-----------|
| 1 | Run WorldManager on I-RAVEN dataset | Specialists emerge matching 4 rule types |
| 2 | Measure accuracy vs paper baseline | >= 88% overall, >= 99% OOD |
| 3 | Run VSA Translator on RAVEN specialists | Descriptions match constant/progression/arithmetic/distribute-three |
| 4 | Generate 100 historical events (1800-1850) via LLM | JSONL format + prompt quality |
| 5 | Encode 100 events to block-codes | EventEncoder pipeline |
| 6 | Run WorldManager on 100 historical transitions | Domain specialists emerge (expect 5-10) |
| 7 | Run VSA Translator on historical specialists | Human-readable domain descriptions |
| 8 | Run counterfactual ("What if no railways?") | Binding interference produces sensible alt-timeline |
| 9 | Integrate with DecisionOracle | Specialist-attributed futures in decision tree |
| 10 | Scale to 15K+ events | Full self-organization + ablation study |

---

## 10. Dependencies

- Existing: BlockCodes, HYLA, CVL, WorldEncoder, DecisionOracle, CausalCodebook, CausalGraph
- New: None (pure numpy/grilly, no new packages)
