# Product Spec: Experiential Memory Wearable

**Codename:** Whatever Remembers
**Author:** Nicolas Cloutier (Grillcheese AI)
**Date:** 2026-03-30
**Status:** Concept / Technical Feasibility Proven

---

## 1. Vision

A wearable device (ring, bracelet, pendant) that develops its own memory of your physiological experiences using Vector Symbolic Architecture. The object accumulates a compressed representation of your body's states over time, learns your patterns via Hebbian plasticity, and can recall similar past states when your body enters a familiar configuration.

**The core insight:** VSA operations (bind, bundle, similarity) are pure integer arithmetic — circular convolution, addition, dot product. No floating point, no GPU, no cloud. The entire cognitive pipeline runs on a $2 MCU with 32KB RAM.

**The metaphor:** Muscle memory externalized into an object. The ring doesn't just track your heart rate — it *remembers* what your whole body felt like during that run last Tuesday morning and recognizes when you're approaching that same state again.

---

## 2. Technical Architecture

### 2.1 Sensor Inputs

| Sensor | Signal | Encoding | MCU Cost |
|--------|--------|----------|----------|
| Skin temperature | 20-40C continuous | Thermometer code (10 thresholds) | ~100 ops |
| Heart rate (PPG) | 40-200 BPM | Thermometer code (16 thresholds) | ~160 ops |
| Galvanic skin response | 0-20 uS | Thermometer code (8 thresholds) — arousal proxy | ~80 ops |
| Accelerometer (3-axis) | Movement magnitude | Thermometer code (8 thresholds) | ~80 ops |
| Ambient light | 0-100k lux | Thermometer code (6 thresholds) — circadian proxy | ~60 ops |
| Time of day | 0-24h | Cyclic phase encoding (cos/sin mapped to thresholds) | ~100 ops |

**Total encoding cost per sample: ~580 integer operations.**

### 2.2 VSA Encoding Pipeline

```
V_experience = M_temp(thermo(skin_temp))
             ⊗ M_heart(thermo(heart_rate))
             ⊗ M_gsr(thermo(galvanic))
             ⊗ M_motion(thermo(accel_mag))
             ⊗ M_circadian(phase(hour, light))
```

Each `M_*` is a pre-computed orthogonal binding matrix stored in flash (MBAT, Gallant 2022). Thermometer codes ensure numerically close values produce similar vectors.

**Vector dimensions:** D = 1024 bits (binary VSA after binarization)
- Each experience vector: 128 bytes
- Orthogonal binding matrices: 6 matrices × 1024 × 1024 bits = 768KB (stored in flash, not RAM)
- Alternative: use circular convolution (block-codes) to avoid matrix storage entirely

### 2.3 Memory Architecture

**Continuous Item Memory** (same as CubeMind's `ContinuousItemMemory`):
- Stores N binary experience vectors (N = 64-256 depending on RAM)
- At 128 bytes per vector: 64 vectors = 8KB, 256 vectors = 32KB
- Hamming distance similarity search: O(N × D/32) — 32 experiences compared in ~1ms on Cortex-M0
- New experiences added via bundling (element-wise majority vote for binary vectors)
- Old experiences decay via stochastic bit flipping (biological forgetting)

**Oja Plasticity Layer** (same as CubeMind's `PersonalityLayer`):
- A small (D × K) weight matrix where K = number of "pattern clusters"
- Hebbian update: when a new experience matches cluster k, strengthen those connections
- Over time, the device develops K distinct "prototype experiences" — your patterns
- K = 8 typical patterns: morning_run, evening_relax, work_stress, deep_sleep, social_excitement, etc.

### 2.4 Recall and Output

**Similarity detection:** Every T seconds (T = 30-60), the device computes Hamming similarity between the current encoded state and all stored prototypes. If similarity exceeds threshold:

- **Haptic feedback:** Vibration pattern indicates which prototype matches
  - Gentle pulse: "you felt this way before" (familiar state)
  - Double pulse: "this is unusual for this time of day" (anomaly detection)
  - Rhythmic: "entering your exercise pattern" (state transition detected)

- **LED color** (if bracelet/pendant):
  - Warm colors: high valence states (matching happy prototypes)
  - Cool colors: low valence states
  - Brightness: arousal level

- **BLE beacon:** Broadcast current state hash for companion app retrieval

### 2.5 Hardware Requirements

| Component | Spec | Cost |
|-----------|------|------|
| MCU | ARM Cortex-M0+ (32KB RAM, 256KB flash) | $1.50 |
| PPG sensor | MAX30102 (heart rate + SpO2) | $3.00 |
| GSR electrodes | 2x stainless steel pads | $0.50 |
| Accelerometer | LIS2DH12 (3-axis, ultra-low power) | $1.00 |
| Temperature | NTC thermistor (skin contact) | $0.20 |
| Light sensor | VEML7700 (ambient light) | $0.80 |
| Haptic motor | LRA (linear resonant actuator) | $1.50 |
| BLE | nRF52810 (or integrated in MCU) | $2.00 |
| Battery | 50mAh LiPo (ring) / 200mAh (bracelet) | $1.00 |
| **Total BOM** | | **~$11.50** |

**Power budget:** VSA operations are integer-only. Encoding + similarity search: ~2mW. Sensors: ~5mW (duty-cycled). BLE: ~3mW (advertising mode). Total: ~10mW average → 50mAh battery lasts ~5 hours continuous, ~2 days with aggressive duty cycling.

---

## 3. Software Architecture

### 3.1 Firmware (C, bare-metal or RTOS)

```
// Main loop — runs every 30 seconds
void experiential_loop() {
    // 1. Read sensors
    float temp = read_skin_temperature();
    float hr   = read_heart_rate();
    float gsr  = read_galvanic();
    float accel = read_accelerometer_magnitude();
    float light = read_ambient_light();
    uint8_t hour = get_rtc_hour();

    // 2. Encode experience (all integer ops)
    uint32_t exp[VSA_WORDS];  // 1024 bits = 32 uint32_t
    encode_experience(exp, temp, hr, gsr, accel, light, hour);

    // 3. Compare to prototypes
    int best_match = -1;
    int best_sim = 0;
    for (int k = 0; k < N_PROTOTYPES; k++) {
        int sim = hamming_similarity(exp, prototypes[k]);
        if (sim > best_sim) {
            best_sim = sim;
            best_match = k;
        }
    }

    // 4. Oja plasticity update
    if (best_sim > SIMILARITY_THRESHOLD) {
        oja_update(prototypes[best_match], exp, LEARNING_RATE);
        trigger_haptic(FAMILIAR_PATTERN);
    } else if (is_anomalous(exp, hour)) {
        trigger_haptic(ANOMALY_PATTERN);
    }

    // 5. Store in episodic memory (circular buffer)
    store_episode(exp);
}
```

### 3.2 Companion App (Optional)

- Receives BLE state hash broadcasts
- Maintains full-resolution experience log (phone has unlimited storage)
- Visualization: "emotional heatmap" over time (Hartmann BSM-style)
- Query interface: "show me times I felt like this before"
- Export: CSV of encoded experiences for research

### 3.3 Privacy by Design

- **All computation on-device.** No cloud, no API, no data leaves the ring.
- Raw sensor data is never stored — only the compressed VSA vector (irreversible hash).
- Cannot reconstruct exact heart rate from the experience vector (information-theoretic compression).
- BLE broadcasts only the Hamming hash, not the full vector.
- Companion app is optional — device works fully standalone.

---

## 4. Use Cases

### 4.1 Personal Wellness
"The ring noticed I'm entering my stress pattern (high cortisol proxy from GSR + elevated HR + low movement). It pulses twice — reminding me to take a break. It learned this pattern from the last 3 weeks of wearing it."

### 4.2 Athletic Training
"My bracelet recognizes my 'peak performance zone' — the specific combination of heart rate, skin temp, and movement that correlates with my best runs. When I'm approaching that zone during warmup, it gives a gentle pulse."

### 4.3 Sleep Quality
"The pendant tracks my physiological state during sleep. Over weeks, it builds prototypes for deep sleep, REM, and restless periods. The companion app shows which evenings led to the best sleep patterns."

### 4.4 Emotional Self-Awareness (Moulder Just-in-Time Interventions)
"I keep reaching for my phone when stressed (GSR spike + low movement + specific time pattern). The ring recognizes this pattern and vibrates — a just-in-time intervention suggesting I try something different (Moulder et al., 2026)."

### 4.5 Shared Objects (Relationship Memory)
"Two rings can exchange experience vectors via BLE. When both wearers are in similar physiological states simultaneously, both rings pulse — shared resonance. Over time, the rings develop a shared memory of co-experienced states."

### 4.6 Therapeutic Applications
"A therapist gives a patient a bracelet. Over the treatment period, the device tracks physiological patterns. The transition matrix between prototype states (Moulder's stability/spread metric) objectively measures whether the patient is developing more adaptive emotional regulation strategies."

---

## 5. Differentiation

| Feature | Fitbit/Apple Watch | Oura Ring | Whatever Remembers |
|---------|-------------------|-----------|-------------------|
| Sensor data | Stored as timeseries | Stored as timeseries | Compressed to VSA vector (irreversible) |
| Pattern recognition | Cloud ML models | Cloud ML models | On-device Hebbian plasticity |
| Privacy | Data uploaded to cloud | Data uploaded to cloud | Never leaves device |
| Memory | None (stateless) | None (stateless) | Develops persistent experiential memory |
| Personalization | Algorithm-based | Algorithm-based | Emergent (Oja plasticity) |
| Compute | ARM Cortex-A + cloud | Nordic + cloud | ARM Cortex-M0 only ($1.50) |
| Emotional awareness | Heart rate zones | Readiness score | Valence-as-weight + full affect encoding |
| Cost | $300+ | $300+ | **$12 BOM** |

---

## 6. Technical Feasibility

**Proven components (all tested in CubeMind):**

| Component | CubeMind Module | Status |
|-----------|----------------|--------|
| Thermometer encoding | `ExperientialEncoder._thermometer_encode()` | 16/16 tests passing |
| Orthogonal binding (MBAT) | `generate_orthogonal_matrix()` | Norm-preserving through 100 nesting levels |
| Hamming similarity search | `ContinuousItemMemory` | Production, used in taste formation |
| Oja plasticity | `PlasticCodebook` (model2.py) | Production, used in personality layer |
| Circadian encoding | `CircadianCells` (cortex.py) | Production, used in brain cortex |
| Neurochemical mapping | `NeurochemicalState` (snn.py) | Production, 4-hormone ODE |
| Valence-as-weight | Hartmann affective alpha | 8/8 tests passing |

**What remains:**
1. Port VSA ops to C for Cortex-M0 (integer-only, no stdlib)
2. Hardware prototype (off-the-shelf dev board + sensors)
3. Power optimization (duty cycling, sensor sleep modes)
4. Haptic pattern design (user study needed)
5. BLE protocol spec

---

## 7. IP and Research Value

This product concept sits at the intersection of:
- **Vector Symbolic Architectures** (Kanerva, Plate, Gallant)
- **Embodied cognition** (Hartmann valence-as-weight BSMs)
- **Emotion regulation science** (Moulder transition matrices)
- **Neuromorphic computing** (Oja plasticity, STDP)
- **Active inference** (Free Energy Principle applied to wearables)

No existing product combines on-device VSA memory with physiological sensing and Hebbian self-organization. Patent-worthy.

**Paper potential:** "Experiential Memory Wearables: Vector Symbolic Architectures for On-Device Physiological Pattern Learning" — suitable for CHI, UbiComp, or UIST.
