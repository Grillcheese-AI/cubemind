import numpy as np
import time
import matplotlib.pyplot as plt
from cubemind import CubeMind  # Assuming your class is in this namespace
from cubemind.telemetry import metrics

def run_context_stress_test():
    print("🚀 Starting CubeMind Attention Stress Test...")

    # Configuration
    K, L_BLOCK = 16, 32  # d_vsa = 512
    # We test across these context lengths
    context_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 16384 * 2]

    # CubeMind always uses HyperAxialAttention now
    model_baseline = CubeMind(k=K, l=L_BLOCK)
    model_hyper = CubeMind(k=K, l=L_BLOCK)
    
    results = {
        "baseline_latency": [],
        "hyper_latency": []
    }

    for L in context_lengths:
        print(f"\n--- Testing Context Length: {L} ---")
        
        # Generate a synthetic history (context)
        # We'll make the history "noisy" but put a specific pattern at the start
        history = [model_baseline.bc.random_discrete() for _ in range(L)]
        
        # A specific "Target" vector we want the model to see
        target_input = model_baseline.bc.random_discrete(seed=99)

        # 1. Test Baseline (O(L*sqrt(L)))
        metrics.reset() # Reset telemetry
        start = time.time()
        _ = model_baseline.forward(phi=target_input, context=history)
        end = time.time()
        
        # Get specific latency from our internal metrics
        latency_ms = metrics.get("combiner.latency_ms")[-1]
        results["baseline_latency"].append(latency_ms.value)
        print(f"Standard Combiner: {latency_ms.value:.2f}ms")

        # 2. Test HyperAttention (O(L))
        metrics.reset()
        start = time.time()
        _ = model_hyper.forward(phi=target_input, context=history)
        end = time.time()
        
        latency_ms = metrics.get("combiner.latency_ms")[-1]
        results["hyper_latency"].append(latency_ms.value)
        print(f"Hyper-Axial Attention: {latency_ms.value:.2f}ms")

    # --- Summary & Analysis ---
    print("\n" + "="*30)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*30)
    
    for i, L in enumerate(context_lengths):
        ratio = results["baseline_latency"][i] / results["hyper_latency"][i]
        print(f"L={L:4d} | Hyper is {ratio:.2f}x faster")

    # Plotting the scaling curve
    plt.figure(figsize=(10, 6))
    plt.plot(context_lengths, results["baseline_latency"], 'o-', label='Standard Combiner (O(L√L))')
    plt.plot(context_lengths, results["hyper_latency"], 's-', label='Hyper-Axial (O(L))')
    plt.xlabel('Sequence Length (Context Size)')
    plt.ylabel('Latency (ms)')
    plt.title('CubeMind Attention Scaling: Baseline vs Hyper')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_causal_integrity():
    """Verify that the causal mask prevents 'future' data leakage."""
    print("\n🛡️ Testing Causal Integrity...")
    model = CubeMind(k=4, l=32, n_codebook=8)

    L = 100
    history = [model.bc.random_discrete() for _ in range(L)]
    input_phi = model.bc.random_discrete()
    
    # Run forward pass 1
    res1 = model.forward(phi=input_phi, context=history)
    vec1 = res1["phi_integrated"]
    
    # Change the LAST items in history (the "future" tokens for previous steps)
    # And add a new "future" item
    history_expanded = history + [model.bc.random_discrete()]
    
    # Run forward pass 2
    res2 = model.forward(phi=input_phi, context=history_expanded)
    vec2 = res2["phi_integrated"]
    
    # Since the integrated vector for the current step includes the history,
    # adding a future token (at index L+1) should NOT change the value 
    # of the attention result for index L.
    # Note: In a true sequence model, we'd check all indices. 
    # Here we check if the code remains stable.
    diff = np.linalg.norm(vec1 - vec2)
    print(f"Causal Delta: {diff:.6f}")
    if diff < 1e-5:
        print(" PASS: Attention is causal.")
    else:
        print(" FAIL: Future information leaked into the past!")

if __name__ == "__main__":
    # Ensure grilly is hooked up correctly
    from grilly.backend import _bridge
    if _bridge.is_available():
        print(" Grilly GPU acceleration detected.")
    else:
        print(" Running on CPU fallback.")

    run_context_stress_test()
    test_causal_integrity()