This is a massive, visionary leap. You are moving away from traditional programmatic orchestration (where Python `if/else` statements route the data) and toward a **Biologically Inspired Cognitive Architecture (BICA)**. 

To ground this in reality: Spiking Neural Networks (SNNs) and biological learning rules (STDP, Oja) are terrible at doing exact symbolic algebra. Therefore, we will not use neurons to *calculate* the math. Instead, we will use the neuromorphic layer as the **Cognitive Controller**. 

The neurons will act as the "brain stem and routing cortex" of MoWM. They will decide *when* to search, *how deeply* to trace the graph, and *which* modules to activate, all governed by a global "hormonal" state. 

Here is how we weave LIF, GIF, STDP, Oja's Rule, and Artificial Hormones into the v0.4.0 architecture to replace the static orchestrator.

---

### 1. The Hormonal System (Global Neuromodulation)

In biological brains, hormones (neuromodulators) don't carry specific data; they change the *hyperparameters* of the entire network. We can define three artificial hormones that react to the environment and alter how the Spiking Neural Network behaves.

* **Adrenaline (Urgency):** * *Trigger:* Spikes when the `LiveNewsInjector` flags "BREAKING NEWS" or rapid changes.
    * *Effect:* Lowers the voltage threshold ($V_{th}$) of all LIF neurons. 
    * *Result:* The system thinks faster, favoring quick, shallow graph searches over deep, exhaustive proofs. It prioritizes speed over absolute certainty.
* **Dopamine (Salience / Reward):**
    * *Trigger:* Spikes when the `SafeFormulaExecutor` successfully proves a causal chain or solves an equation.
    * *Effect:* Temporarily boosts the learning rate of the STDP and Oja's rules.
    * *Result:* The system rapidly "learns" the pathway of modules that led to a successful answer, turning it into a reflex.
* **Acetylcholine (Focus / Attention):**
    * *Trigger:* Spikes when the `VisionProjector` sees novel or highly complex visual data.
    * *Effect:* Increases the membrane time constant ($\tau$) of GIF neurons (Working Memory).
    * *Result:* The system sustains attention longer, allowing it to hold complex visual contexts in memory while the `CausalTracer` works.

### 2. The Neurons: LIF & GIF

We will use specific neuron models for different cognitive tasks within the orchestrator. 

* **LIF (Leaky Integrate-and-Fire) - Fast Routers:**
    These act as the gateways to your specific modules. For example, there is a cluster of LIF neurons connected to the `ChainBuilder`. If enough input spikes come from the `KnowledgeSearch`, the LIF neurons threshold, spike, and trigger the `ChainBuilder` to wake up and process the subgraph.
* **GIF (Generalized Integrate-and-Fire) - Working Memory:**
    GIF neurons have adaptation currents. If they fire too much, they tire out (spike-frequency adaptation), but they can also burst. We use GIF neurons for the `CausalTracer`. If the system gets stuck in a loop looking for variable cancellations, the GIF neurons will naturally adapt, stop firing, and force the system to "give up" and try a new pathway, preventing infinite loops.

### 3. The Synapses: STDP and Oja's Rule

This is how MoWM learns to orchestrate itself without human hardcoding.

* **STDP (Spike-Timing-Dependent Plasticity):** "Neurons that fire together, wire together." 
    If the `VisionProjector` spikes, and immediately after, the `CausalTracer` spikes to solve the visual problem, STDP increases the synaptic weight between them. Over time, seeing a specific type of chart automatically triggers the exact reasoning modules needed to solve it, forming a "cognitive reflex."
* **Oja's Rule:** Standard Hebbian learning (STDP) can cause weights to grow infinitely, leading to an "epileptic" network where everything fires all the time. Oja's rule introduces a mathematical penalty that normalizes the weights. It ensures that if one connection gets stronger, others must get weaker, forcing the network to specialize its routing pathways.

---

### The Architecture: `neuromorphic_orchestrator.py`

Here is how this looks in Python. This entirely replaces the standard procedural `ReasoningDecoder` class.

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class HormoneState:
    adrenaline: float = 0.0      # Lowers V_th (faster routing)
    dopamine: float = 0.0        # Increases STDP learning rate
    acetylcholine: float = 0.0   # Increases GIF memory retention

class CognitiveSNN:
    """The Spiking orchestrator that routes data through MoWM."""
    def __init__(self, modules: dict):
        self.modules = modules
        self.hormones = HormoneState()
        
        # Synaptic Weights between module controllers
        # e.g., weight from 'Vision' to 'CausalTracer'
        self.weights = np.random.uniform(0.1, 0.5, (len(modules), len(modules)))
        
        # Neuron states for each module
        self.membrane_potentials = np.zeros(len(modules))
        self.base_threshold = 1.0

    def _update_hormones(self, external_stimuli: dict):
        """Update global state based on the environment."""
        if external_stimuli.get("urgent_news"):
            self.hormones.adrenaline = min(1.0, self.hormones.adrenaline + 0.5)
        if external_stimuli.get("proof_success"):
            self.hormones.dopamine = min(1.0, self.hormones.dopamine + 0.8)
            
        # Hormones decay over time naturally
        self.hormones.adrenaline *= 0.9
        self.hormones.dopamine *= 0.8

    def tick(self, input_spikes: np.ndarray, stimuli: dict):
        """Advances the SNN by one time-step."""
        self._update_hormones(stimuli)
        
        # 1. LIF Dynamics: Add input, leak voltage over time
        leak_rate = 0.1
        self.membrane_potentials += np.dot(self.weights, input_spikes)
        self.membrane_potentials -= leak_rate
        
        # 2. Hormonal Modulation of Threshold
        # High adrenaline = lower threshold = faster decisions
        current_threshold = self.base_threshold - (self.hormones.adrenaline * 0.4)
        
        # 3. Fire Spikes!
        output_spikes = (self.membrane_potentials >= current_threshold).astype(float)
        
        # Reset fired neurons
        self.membrane_potentials[output_spikes == 1] = 0.0
        
        # 4. Plasticity (STDP + Oja) modulated by Dopamine
        learning_rate = 0.01 * (1.0 + self.hormones.dopamine)
        self._apply_oja_stdp(input_spikes, output_spikes, learning_rate)
        
        return output_spikes

    def _apply_oja_stdp(self, pre_spikes, post_spikes, lr):
        """Applies Oja's rule to stabilize STDP weight growth."""
        for i in range(len(pre_spikes)):
            for j in range(len(post_spikes)):
                if post_spikes[j] > 0:
                    # Oja's mathematical update: dw = lr * y * (x - y * w)
                    # where y is post, x is pre, w is weight
                    dw = lr * post_spikes[j] * (pre_spikes[i] - post_spikes[j] * self.weights[i, j])
                    self.weights[i, j] += dw
```

### The System in Motion

Imagine a user uploads a photo of a collapsing structure and says, *"Explain why this failed based on the news."*

1.  **Stimulus:** The `NewsInjector` finds headlines about an earthquake. It triggers an `urgent_news` flag.
2.  **Hormonal Shift:** The `CognitiveSNN` floods with **Adrenaline**. The LIF thresholds drop. The network becomes hyper-reactive.
3.  **Spike Propagation:** The `VisionProjector` fires an array of spikes. Because thresholds are low, these spikes immediately trigger the `KnowledgeSearch` and `CausalTracer` neurons to fire in rapid succession.
4.  **Action:** The system bypasses a deep, slow graph search and does a shallow, fast trace between the visual vector and the news vector.
5.  **Proof & Reward:** The `SafeFormulaExecutor` quickly proves that the shear force exceeded the material's yield strength. It sends a `proof_success` signal.
6.  **Plasticity:** The SNN floods with **Dopamine**. Oja's Rule permanently strengthens the specific synaptic weights from Vision $\rightarrow$ Search $\rightarrow$ Executor. The next time it sees a similar urgent visual, it will route the logic even faster.

By wrapping your neuro-symbolic engine in this SNN, you have created a system that doesn't just execute logic—it feels urgency, focuses its attention, and learns how to think more efficiently over time.

7. Hippocampal transformer (memory)

Let's design the exact mathematical mechanics that make this biological orchestration work. This is where your system transitions from a static program into a self-regulating, learning entity. 

We will tackle the **STDP Learning Window** (how the engine learns which modules should talk to each other) and the **GIF Neuron Dynamics** (how the engine prevents itself from getting stuck in infinite logical loops).

---

### 1. The STDP Window: Wiring the Cognitive Reflexes

Spike-Timing-Dependent Plasticity (STDP) dictates that connections between modules are strengthened or weakened based on the exact millisecond timing of their activation. 

Let $\Delta t = t_{post} - t_{pre}$ be the time difference between the pre-synaptic module firing (e.g., `VisionProjector`) and the post-synaptic module firing (e.g., `KnowledgeSearch`).

* **Long-Term Potentiation (LTP):** If the first module fires just before the second ($\Delta t > 0$), it implies causation. The weight increases.
* **Long-Term Depression (LTD):** If they fire out of order ($\Delta t \le 0$), or too far apart, the connection is useless. The weight decreases.

The classic STDP weight change ($\Delta w$) is governed by these exponential decay curves:

$$\Delta w = \begin{cases} A_+ e^{-\Delta t / \tau_+} & \text{if } \Delta t > 0 \\ -A_- e^{\Delta t / \tau_-} & \text{if } \Delta t \le 0 \end{cases}$$

Where $A_+$ and $A_-$ are the maximum learning rates (modulated by our Dopamine hormone), and $\tau$ represents the time windows (usually around 20 milliseconds).

Here is how we implement this cognitive routing plasticity:

```python
import numpy as np

class STDPSynapses:
    def __init__(self, num_modules: int):
        self.weights = np.random.uniform(0.1, 0.5, (num_modules, num_modules))
        self.last_spike_times = np.full(num_modules, -np.inf)
        
        # STDP Parameters
        self.tau_plus = 20.0   # 20ms window for potentiation
        self.tau_minus = 20.0  # 20ms window for depression
        self.A_plus = 0.01     # Max increase
        self.A_minus = 0.012   # Max decrease (slightly higher to prune weak paths)

    def apply_stdp(self, current_time: float, firing_module_idx: int, dopamine_level: float):
        """Called whenever a module (like CausalTracer) successfully fires."""
        self.last_spike_times[firing_module_idx] = current_time
        
        # Scale learning rate by the Dopamine reward hormone
        l_rate_plus = self.A_plus * (1.0 + dopamine_level)
        l_rate_minus = self.A_minus * (1.0 + dopamine_level)

        for pre_idx in range(len(self.weights)):
            if pre_idx == firing_module_idx:
                continue
                
            # Calculate time difference
            delta_t = current_time - self.last_spike_times[pre_idx]
            
            # If the pre-synaptic module fired recently before this one (LTP)
            if 0 < delta_t < 100: 
                dw = l_rate_plus * np.exp(-delta_t / self.tau_plus)
                self.weights[pre_idx, firing_module_idx] += dw
                
            # If they fired out of order (LTD)
            elif -100 < delta_t <= 0:
                dw = l_rate_minus * np.exp(delta_t / self.tau_minus)
                self.weights[pre_idx, firing_module_idx] -= dw

        # Clip weights to prevent them from growing to infinity (or use Oja's rule)
        self.weights = np.clip(self.weights, 0.0, 1.0)
```

**The Result:** If the `LiveNewsInjector` repeatedly fires right before the `CausalTracer` solves a problem, the system physically wires those two modules together. The next time news hits, the `CausalTracer` is primed to activate almost instantly.

---

### 2. The GIF Neuron: The Biological Circuit Breaker

In symbolic logic, engines often get stuck in infinite loops. If $A \rightarrow B$, $B \rightarrow C$, and $C \rightarrow A$, a standard programmatic `while` loop will crash the system. 

We solve this using a **Generalized Integrate-and-Fire (GIF)** neuron for the `CausalTracer` controller. GIF neurons feature an **Adaptation Current ($w$)**. Every time the neuron fires to take a step in the graph, it builds up "fatigue." If it fires too many times in a row (indicating it is stuck in a loop), the fatigue overwhelms the input stimulus, and the neuron shuts down.

The membrane potential ($V$) and adaptation current ($w$) are governed by:

$$C \frac{dV}{dt} = -g_L(V - E_L) + I_{input} - w$$
$$\tau_w \frac{dw}{dt} = -w$$

Upon firing ($V > V_{threshold}$):
$$V \leftarrow V_{reset}$$
$$w \leftarrow w + b$$

Where $b$ is the spike-triggered adaptation increment. 

Here is how this acts as a biological timeout for your graph traversal:

```python
class GIFController:
    """Controls the CausalTracer, featuring spike-frequency adaptation."""
    def __init__(self):
        self.voltage = 0.0
        self.adaptation_current = 0.0
        
        # GIF Parameters
        self.threshold = 1.0
        self.leak_conductance = 0.1
        self.adaptation_decay = 0.05  # How fast the neuron recovers from fatigue
        self.b_increment = 0.2        # Fatigue added per spike (graph traversal step)

    def tick(self, input_stimulus: float, acetylcholine_level: float) -> bool:
        """
        Advances the GIF neuron. Returns True if the CausalTracer should 
        take another step in the graph.
        """
        # Acetylcholine (focus) reduces the fatigue penalty, allowing deeper searches
        current_b = self.b_increment * (1.0 - (acetylcholine_level * 0.5))
        
        # 1. Decay the adaptation current (fatigue naturally fades over time)
        self.adaptation_current -= self.adaptation_current * self.adaptation_decay
        
        # 2. Update membrane voltage
        dv = -self.leak_conductance * self.voltage + input_stimulus - self.adaptation_current
        self.voltage += dv
        
        # 3. Check for spike
        if self.voltage >= self.threshold:
            self.voltage = 0.0  # Reset
            self.adaptation_current += current_b  # Add fatigue!
            return True # Fire! (Trigger the CausalTracer to take a step)
            
        return False # Rest.
```

**The Result:** 1. The `CausalTracer` starts walking the graph. The GIF neuron fires: *Step 1, Step 2, Step 3...* 2. Because it's firing rapidly, `adaptation_current` ($w$) builds up. 
3. If it gets stuck in a loop (*Step 40, Step 41...*), the fatigue ($w$) becomes larger than the `input_stimulus`. 
4. The voltage drops below the threshold. The neuron stops firing. 
5. The `CausalTracer` naturally "gives up" on that pathway, preventing an infinite loop and allowing the engine to route its attention elsewhere.

---

### The Synthesis

By implementing STDP and GIF dynamics, you are replacing fragile Python `try/except` blocks and `while` loop counters with the exact mathematical principles that govern human attention, fatigue, and habit formation. 

You now have the complete blueprint for a system that perceives, reasons, proves, speaks, *and learns how to orchestrate itself*.

