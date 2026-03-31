# Colab cell: verify Vulkan / grilly GPU backend status
# Copy-paste into a notebook cell

import sys, os
print(f"Python: {sys.version}")
print(f"Platform: {sys.platform}")

# 1. Check grilly import
try:
    import grilly
    print(f"\n✓ grilly imported (v{getattr(grilly, '__version__', '?')})")
except ImportError as e:
    print(f"\n✗ grilly not installed: {e}")
    print("  Fix: !pip install grilly")

# 2. Check Vulkan bridge
try:
    from grilly.backend import _bridge
    if _bridge.is_available():
        print(f"✓ Vulkan bridge available")
        info = _bridge.device_info() if hasattr(_bridge, 'device_info') else {}
        if info:
            for k, v in info.items():
                print(f"  {k}: {v}")
    else:
        print("✗ Vulkan bridge loaded but NOT available")
except Exception as e:
    print(f"✗ Vulkan bridge failed: {e}")

# 3. Check block codes GPU fallback level
try:
    from cubemind.ops.block_codes import BlockCodes
    bc = BlockCodes(k=4, l=32)
    a = bc.random_discrete(seed=1)
    b = bc.random_discrete(seed=2)
    c = bc.bind(a, b)
    sim = bc.similarity(a, a)
    print(f"\n✓ BlockCodes working (bind + sim={sim:.3f})")
    # Check which backend is active
    if hasattr(bc, '_backend'):
        print(f"  Backend: {bc._backend}")
    else:
        print("  Backend: numpy fallback (no GPU)")
except Exception as e:
    print(f"\n✗ BlockCodes failed: {e}")

# 4. Check PyTorch + CUDA (for eval script)
try:
    import torch
    print(f"\n✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
except ImportError:
    print("\n✗ PyTorch not installed")

print("\n--- done ---")
