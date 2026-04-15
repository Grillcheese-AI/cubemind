"""CubeMind Live Brain Demo — webcam + audio + keyboard teaching.

Full perception→reasoning→memory→neurogenesis loop running live.
Shows: video feed, brain state overlay, neurochemistry, neuron count,
hippocampal memories, spike activity.

Controls:
  T — teach: type a label for what the camera sees
  R — recall: query memory with current visual input
  S — stats: print brain statistics
  Q — quit
  SPACE — freeze frame and show detailed brain state

Usage:
    python scripts/live_brain.py
    python scripts/live_brain.py --camera 0 --width 1920 --height 1080
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from cubemind import CubeMind, create_cubemind


def draw_overlay(frame: np.ndarray, result: dict, brain: CubeMind) -> np.ndarray:
    """Draw brain state overlay on the video frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent dark panel on the right
    panel_w = 320
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    x0 = w - panel_w + 10
    y = 25
    dy = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45
    white = (255, 255, 255)
    green = (0, 255, 128)
    cyan = (255, 200, 0)
    yellow = (0, 220, 255)
    red = (80, 80, 255)

    # Title
    cv2.putText(frame, "CubeMind Brain", (x0, y), font, 0.6, cyan, 2)
    y += dy + 5

    # Step count
    cv2.putText(frame, f"Step: {result['step']}", (x0, y), font, fs, white, 1)
    y += dy

    # Confidence
    conf = result.get("confidence", 0)
    color = green if conf > 0.5 else yellow if conf > 0.2 else red
    cv2.putText(frame, f"Confidence: {conf:.3f}", (x0, y), font, fs, color, 1)
    y += dy

    # Memories
    n_mem = result.get("memories_retrieved", 0)
    cv2.putText(frame, f"Memories retrieved: {n_mem}", (x0, y), font, fs, white, 1)
    y += dy

    total_mem = brain.hippocampus.memory_count
    cv2.putText(frame, f"Total memories: {total_mem}", (x0, y), font, fs, white, 1)
    y += dy + 5

    # Neurogenesis
    ng = result.get("neurogenesis", {})
    n_neurons = ng.get("neuron_count", "?")
    grew = ng.get("grew", False)
    pruned = ng.get("pruned", 0)
    residual = ng.get("residual_ema", 0)
    cv2.putText(frame, "-- Neurogenesis --", (x0, y), font, fs, cyan, 1)
    y += dy
    grow_str = f"Neurons: {n_neurons}"
    if grew:
        grow_str += " [GREW!]"
    cv2.putText(frame, grow_str, (x0, y), font, fs, green if grew else white, 1)
    y += dy
    if pruned:
        cv2.putText(frame, f"Pruned: {pruned}", (x0, y), font, fs, red, 1)
        y += dy
    cv2.putText(frame, f"Residual EMA: {residual:.4f}", (x0, y), font, fs, white, 1)
    y += dy + 5

    # Neurochemistry
    nc = result.get("neurochemistry", {})
    if nc:
        cv2.putText(frame, "-- Neurochemistry --", (x0, y), font, fs, cyan, 1)
        y += dy
        for name in ["dopamine", "serotonin", "cortisol", "noradrenaline", "oxytocin"]:
            val = nc.get(name, 0)
            bar_len = int(val * 150)
            colors = {
                "dopamine": (0, 200, 255),
                "serotonin": (255, 200, 0),
                "cortisol": (80, 80, 255),
                "noradrenaline": (0, 255, 0),
                "oxytocin": (255, 100, 255),
            }
            c = colors.get(name, white)
            cv2.putText(frame, f"{name[:4]}: {val:.2f}", (x0, y), font, 0.4, c, 1)
            cv2.rectangle(frame, (x0 + 80, y - 8), (x0 + 80 + bar_len, y + 2), c, -1)
            y += dy - 2
        y += 5

    # Spatial context
    ctx = result.get("spatial_context", {})
    loc = ctx.get("current_location", [0, 0])
    cv2.putText(frame, f"Location: ({loc[0]:.1f}, {loc[1]:.1f})", (x0, y), font, fs, white, 1)
    y += dy

    # FPS (bottom)
    return frame


def main():
    parser = argparse.ArgumentParser(description="CubeMind Live Brain")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--l", type=int, default=64)
    parser.add_argument("--d-hidden", type=int, default=64)
    parser.add_argument("--llm", type=str, default=None, help="Path to GGUF model")
    args = parser.parse_args()

    # Init brain
    print("Initializing CubeMind brain...")
    brain = create_cubemind(
        k=args.k, l=args.l, d_hidden=args.d_hidden,
        n_gif_levels=8, snn_timesteps=2, snn_ratio=0.3,
        enable_stdp=True,
        n_place_cells=500, n_time_cells=50, n_grid_cells=100,
        max_memories=50000, initial_neurons=32, max_neurons=2000,
        growth_threshold=0.3, enable_neurochemistry=True,
    )

    if args.llm:
        print(f"Attaching LLM: {args.llm}")
        brain.attach_llm(model_path=args.llm, n_ctx=2048, n_gpu_layers=-1)

    print(f"Brain ready: d_vsa={brain.d_vsa}, d_hidden={brain.d_hidden}")

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.camera}: {actual_w}x{actual_h}")

    print("\nControls: T=teach  R=recall  S=stats  Q=quit  SPACE=freeze")
    print("=" * 50)

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # Downsample for brain processing (keep display at full res)
        small = cv2.resize(frame, (160, 120))

        # Run brain forward with visual input
        result = brain.forward(image=small)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()

        # Draw overlay
        display = draw_overlay(frame, result, brain)

        # FPS + latency
        cv2.putText(display, f"{fps_display:.0f} FPS | {elapsed_ms:.0f}ms",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)

        cv2.imshow("CubeMind Brain", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('t'):
            # Teach mode: type a label
            label = input("\n  Teach label: ").strip()
            if label:
                result = brain.forward(text=label, image=small)
                print(f"  Stored: '{label}' (conf={result['confidence']:.3f}, "
                      f"neurons={result['neurogenesis']['neuron_count']})")

        elif key == ord('r'):
            # Recall mode: query by current visual
            results = brain.recall(brain.bc.to_flat(result["input_hv"]), k=5)
            print("\n  Recall top-5:")
            for mid, score in results:
                print(f"    {mid}: {score:.3f}")

        elif key == ord('s'):
            stats = brain.stats()
            print("\n  Brain Stats:")
            for k, v in stats.items():
                if isinstance(v, dict):
                    print(f"    {k}:")
                    for k2, v2 in v.items():
                        print(f"      {k2}: {v2}")
                else:
                    print(f"    {k}: {v}")

        elif key == ord(' '):
            print(f"\n  Frozen frame — Step {result['step']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Memories: {result['memories_retrieved']} retrieved, "
                  f"{brain.hippocampus.memory_count} total")
            ng = result['neurogenesis']
            print(f"  Neurons: {ng['neuron_count']} "
                  f"(grew={ng['grew']}, pruned={ng['pruned']})")
            cv2.waitKey(0)  # Wait for any key to unfreeze

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession complete: {brain.stats()['step']} steps, "
          f"{brain.hippocampus.memory_count} memories, "
          f"{brain.neurogenesis.neuron_count} neurons")


if __name__ == "__main__":
    main()
