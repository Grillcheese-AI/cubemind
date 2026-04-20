"""Live webcam SNN demo — see spikes, neurochemistry, and STDP in real-time.

Opens your webcam, feeds frames through the SNN encoder, and displays:
  - Live camera feed with spike overlay
  - Spike rate bar (how active the SNN is)
  - Neurochemistry levels (cortisol, dopamine, serotonin, oxytocin)
  - Dominant emotion
  - STDP learning indicator (weight drift)

Press 'q' to quit. Press 's' to toggle STDP learning on/off.
Press 'r' to reset SNN state. Press 'e' to show emotion details.

Usage:
    python scripts/webcam_snn_demo.py
    python scripts/webcam_snn_demo.py --device 1          # Different camera
    python scripts/webcam_snn_demo.py --neuron-type if     # IF neurons (more sensitive)
    python scripts/webcam_snn_demo.py --no-stdp            # Disable self-learning
"""

import argparse
import time

import cv2
import numpy as np

from cubemind.perception.snn import SNNEncoder
from cubemind.perception.live_vision import FramePreprocessor
from cubemind.perception.cnn_encoder import CNNEncoder
from cubemind.perception.face import FacePerception
from cubemind.ops.vsa_bridge import LSHProjector, ContinuousItemMemory, binarize_and_pack


def draw_bar(img, x, y, w, h, value, color, label=""):
    """Draw a horizontal bar with label."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    bar_w = int(w * min(1.0, max(0.0, value)))
    cv2.rectangle(img, (x, y), (x + bar_w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)
    if label:
        cv2.putText(img, f"{label}: {value:.2f}", (x + 5, y + h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)


def draw_spike_grid(img, spikes, x, y, grid_w, grid_h, n_cols=32):
    """Draw spikes as a grid of dots."""
    n = len(spikes)
    n_rows = (n + n_cols - 1) // n_cols
    cell_w = grid_w // n_cols
    cell_h = grid_h // max(n_rows, 1)

    for i in range(min(n, n_cols * n_rows)):
        row, col = i // n_cols, i % n_cols
        cx = x + col * cell_w + cell_w // 2
        cy = y + row * cell_h + cell_h // 2
        if spikes[i] > 0:
            cv2.circle(img, (cx, cy), max(1, cell_w // 3), (0, 255, 0), -1)
        else:
            cv2.circle(img, (cx, cy), max(1, cell_w // 4), (50, 50, 50), -1)


def main():
    parser = argparse.ArgumentParser(description="Live webcam SNN demo")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--neuron-type", choices=["lif", "if"], default="lif")
    parser.add_argument("--neurons", type=int, default=256, help="Number of SNN neurons")
    parser.add_argument("--no-stdp", action="store_true", help="Disable STDP learning")
    parser.add_argument("--feature-dim", type=int, default=128, help="Feature dimension")
    parser.add_argument("--use-cnn", action="store_true", default=False, help="Use CNN encoder")
    parser.add_argument("--no-face", action="store_true", help="Disable face perception (MediaPipe)")
    args = parser.parse_args()

    # Face perception: MediaPipe → 52 blendshapes + 52 deltas = 104D
    face = None
    use_face = not args.no_face
    if use_face:
        try:
            face = FacePerception(max_faces=1)
            args.feature_dim = 104  # 52 blendshapes + 52 deltas
            print(f"Face perception: MediaPipe 52 blendshapes + deltas = {args.feature_dim}D")
        except Exception as e:
            print(f"Face perception unavailable: {e}")
            face = None
            use_face = False

    # CNN fallback if no face perception
    cnn = None
    use_cnn = args.use_cnn and not use_face
    if use_cnn:
        cnn_k, cnn_l = 4, 32
        cnn = CNNEncoder(k=cnn_k, l=cnn_l, grid_size=(1, 1))
        args.feature_dim = cnn_k * cnn_l
        print(f"CNN encoder: k={cnn_k}, l={cnn_l}, feature_dim={args.feature_dim}")

    # Initialize
    preprocessor = FramePreprocessor(target_size=(80, 80), grayscale=True)
    snn = SNNEncoder(
        d_input=args.feature_dim,
        n_neurons=args.neurons,
        d_vsa=2048,
        neuron_type=args.neuron_type,
        tau=20.0,
        v_threshold=0.08,
    )
    snn.stdp_lr_potentiate = 0.00005
    snn.stdp_lr_depress = 0.00002
    snn.stdp_weight_clip = 0.2
    stdp_frame_counter = 0
    STDP_EVERY_N = 10  # Only update weights every 10th frame
    snn.stdp_enabled = not args.no_stdp

    # Use DirectShow on Windows — MSMF hangs when probing devices
    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Cannot open camera device {args.device}")
        return

    # Auto-load saved STDP weights if they exist
    SNN_SAVE_PATH = "cubemind_snn_brain.npz"
    import os
    if os.path.exists(SNN_SAVE_PATH):
        snn.load(SNN_SAVE_PATH)
        print(f"Loaded SNN brain from {SNN_SAVE_PATH} (drift preserved)")

    # Face recognition: LSH + ContinuousItemMemory for identity
    face_lsh = LSHProjector(d_input=128, d_output=2048, seed=77)
    face_memory = ContinuousItemMemory(d_vsa=2048, max_capacity=100)
    FACE_MEMORY_PATH = "cubemind_face_memory"
    if os.path.exists(f"{FACE_MEMORY_PATH}.npz"):
        face_memory.load(FACE_MEMORY_PATH)
        print(f"Loaded {face_memory.size} known faces from {FACE_MEMORY_PATH}")

    recognized_name = ""
    recognized_conf = 0.0
    enroll_samples = []  # Buffer for multi-frame enrollment

    feat_mode = "face" if use_face else ("CNN" if use_cnn else "grid")
    print(f"SNN Demo: {args.neurons} {args.neuron_type.upper()} neurons, "
          f"features={feat_mode}, "
          f"STDP={'ON' if not args.no_stdp else 'OFF'}")
    print("Keys: q=quit (saves), s=toggle STDP, r=reset, e=emotions")
    print("       f=enroll face (hold still, captures 10 frames)")
    print("       d=show known faces")

    show_emotions = False
    fps_counter = []
    initial_weight_norm = float(np.linalg.norm(snn.W_in))
    prev_feat = [None]

    while True:
        t0 = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess — use raw frame directly (phone cameras kill motion signal)
        processed = preprocessor.process(frame)
        processed_input = processed

        # Extract features
        face_result = None
        face_emotion = None
        micro_exprs = []

        if face is not None:
            # MediaPipe face path: frame → 52 blendshapes + 52 deltas = 104D
            face_result = face.process_frame(frame)
            if face_result is not None:
                feat = face.get_snn_features(frame)
                face_emotion = face_result["emotion"]
                micro_exprs = face_result.get("micro_expressions", [])
            else:
                feat = np.zeros(args.feature_dim, dtype=np.float32)
        elif cnn is not None:
            block_code = cnn.forward(processed_input)
            feat = block_code.ravel().astype(np.float32)
        else:
            h, w = processed_input.shape[:2]
            grid_n = max(2, int(np.sqrt(args.feature_dim / 4)))
            cell_h, cell_w = h // grid_n, w // grid_n
            features = []
            for i in range(grid_n):
                for j in range(grid_n):
                    cell = processed_input[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                    features.extend([float(np.mean(cell)), float(np.std(cell)),
                                     float(np.max(cell)), float(np.min(cell))])
            feat = np.array(features[:args.feature_dim], dtype=np.float32)
            if len(feat) < args.feature_dim:
                feat = np.pad(feat, (0, args.feature_dim - len(feat)))

        # Normalize to stable range for SNN
        feat_std = np.std(feat) + 1e-6
        feat = feat / feat_std * 0.3

        # Jitter
        feat += np.random.default_rng().standard_normal(len(feat)).astype(np.float32) * 0.03

        # Feature delta: how much the CNN features changed from last frame
        if prev_feat[0] is not None:
            motion_mag = float(np.mean(np.abs(feat - prev_feat[0])))
        else:
            motion_mag = 0.0
        prev_feat[0] = feat.copy()

        # Gate STDP to every Nth frame to prevent drift runaway
        stdp_frame_counter += 1
        if stdp_frame_counter % STDP_EVERY_N == 0:
            snn.stdp_enabled = not args.no_stdp
        else:
            snn.stdp_enabled = False
        stdp_display = not args.no_stdp  # For UI — always show "ON" if enabled

        # Face recognition: try to identify who's in the frame
        if face is not None and face_result is not None:
            id_feat = face.get_identity_features(frame)
            if id_feat is not None:
                packed_id = binarize_and_pack(face_lsh.project(id_feat))
                if face_memory.size > 0:
                    best_id, best_sim, best_label = face_memory.retrieve_best(packed_id)
                    if best_sim > 0.6:
                        recognized_name = best_label
                        recognized_conf = best_sim
                    else:
                        recognized_name = "Unknown"
                        recognized_conf = best_sim
                else:
                    recognized_name = "No faces enrolled"
                    recognized_conf = 0.0

        # SNN step
        spikes = snn.step(feat)
        spike_rate = float(np.mean(spikes))

        # Map face emotion to neurochemistry valence/threat signals
        face_valence = 0.0
        face_threat = 0.0
        if face_emotion == "happy":
            face_valence = 0.6
        elif face_emotion == "sad":
            face_valence = -0.4
        elif face_emotion == "angry":
            face_valence = -0.3
            face_threat = 0.4
        elif face_emotion == "fearful":
            face_threat = 0.6
        elif face_emotion == "surprised":
            face_valence = 0.2

        snn.neurochemistry.update(
            novelty=spike_rate + len(micro_exprs) * 0.2,
            valence=face_valence,
            threat=face_threat,
            focus=0.3 if face_result is not None else 0.0,
        )

        # Build display
        display = frame.copy()
        dh, dw = display.shape[:2]

        # Panel background
        panel_x = dw - 280
        cv2.rectangle(display, (panel_x, 0), (dw, dh), (20, 20, 20), -1)

        # Title
        if use_face:
            mode_str = f"FACE+{args.neuron_type.upper()}"
        elif use_cnn:
            mode_str = f"CNN+{args.neuron_type.upper()}"
        else:
            mode_str = args.neuron_type.upper()
        cv2.putText(display, f"CubeMind SNN [{mode_str}]",
                    (panel_x + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # Spike rate
        draw_bar(display, panel_x + 10, 40, 250, 18, spike_rate, (0, 255, 0), "Spike Rate")

        # Spike grid
        cv2.putText(display, "Spikes:", (panel_x + 10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        draw_spike_grid(display, spikes[:min(256, len(spikes))],
                        panel_x + 10, 80, 250, 80, n_cols=32)

        # Neurochemistry
        y_start = 175
        nc = snn.neurochemistry
        draw_bar(display, panel_x + 10, y_start, 250, 15, nc.cortisol, (0, 0, 200), "Cortisol")
        draw_bar(display, panel_x + 10, y_start + 20, 250, 15, nc.dopamine, (0, 200, 255), "Dopamine")
        draw_bar(display, panel_x + 10, y_start + 40, 250, 15, nc.serotonin, (200, 200, 0), "Serotonin")
        draw_bar(display, panel_x + 10, y_start + 60, 250, 15, nc.oxytocin, (200, 0, 200), "Oxytocin")

        # Face emotion (from blendshapes) + SNN emotion (from neurochemistry)
        if face_emotion and face_emotion != "neutral":
            face_emo_str = face_emotion.upper()
        else:
            face_emo_str = ""

        snn_emotion = nc.dominant_emotion.upper()
        emotion_colors = {
            "NEUTRAL": (180, 180, 180), "JOY": (0, 255, 255), "CURIOUS": (255, 200, 0),
            "ANXIOUS": (0, 0, 255), "SAD": (200, 100, 0), "WARM": (0, 180, 255),
            "HAPPY": (0, 255, 255), "ANGRY": (0, 0, 255), "SURPRISED": (255, 200, 0),
            "FEARFUL": (0, 100, 255), "DISGUSTED": (0, 180, 0),
        }
        # Show face emotion if available, otherwise SNN emotion
        if face_emo_str:
            display_emo = f"Face: {face_emo_str}  SNN: {snn_emotion}"
            ec = emotion_colors.get(face_emo_str, (180, 180, 180))
        else:
            display_emo = f"SNN: {snn_emotion}"
            ec = emotion_colors.get(snn_emotion, (180, 180, 180))
        cv2.putText(display, display_emo, (panel_x + 10, y_start + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ec, 1)

        # Micro-expressions detected
        if micro_exprs:
            me_text = ", ".join(m["blendshape"][:12] for m in micro_exprs[:3])
            cv2.putText(display, f"Micro: {me_text}", (panel_x + 10, y_start + 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        # Face recognition result
        if recognized_name:
            if recognized_name not in ("Unknown", "No faces enrolled"):
                id_color = (0, 255, 0)
                id_text = f"ID: {recognized_name} ({recognized_conf:.0%})"
            else:
                id_color = (0, 0, 200)
                id_text = f"ID: {recognized_name}"
            cv2.putText(display, id_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, id_color, 2)

        # Enrollment progress
        if enroll_samples:
            cv2.putText(display, f"Enrolling... {len(enroll_samples)}/10",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # STDP indicator
        weight_drift = abs(float(np.linalg.norm(snn.W_in)) - initial_weight_norm)
        stdp_text = f"STDP: {'ON' if stdp_display else 'OFF'} (drift: {weight_drift:.2f})"
        stdp_color = (0, 255, 0) if stdp_display else (0, 0, 200)
        cv2.putText(display, stdp_text, (panel_x + 10, y_start + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, stdp_color, 1)

        # Motion & input debug
        draw_bar(display, panel_x + 10, y_start + 135, 250, 15, min(motion_mag * 20, 1.0), (255, 255, 0), f"Delta: {motion_mag:.3f}")
        draw_bar(display, panel_x + 10, y_start + 155, 250, 15, min(float(np.mean(np.abs(feat))) / 3.0, 1.0), (100, 200, 255), f"Input: {float(np.mean(np.abs(feat))):.2f}")

        # FPS
        elapsed = time.monotonic() - t0
        fps_counter.append(elapsed)
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0
        cv2.putText(display, f"FPS: {fps:.0f}", (panel_x + 10, y_start + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Emotion details panel
        if show_emotions:
            cv2.putText(display, f"Valence: {nc.valence:.2f}", (panel_x + 10, y_start + 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(display, f"Arousal: {nc.arousal:.2f}", (panel_x + 10, y_start + 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(display, f"Stress:  {nc.stress:.2f}", (panel_x + 10, y_start + 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        cv2.imshow("CubeMind SNN Live", display)

        # Auto-enroll: capture frames when 'f' was pressed
        if enroll_samples is not None and len(enroll_samples) > 0 and len(enroll_samples) < 10:
            if face is not None:
                id_feat = face.get_identity_features(frame)
                if id_feat is not None:
                    enroll_samples.append(id_feat)
            if len(enroll_samples) >= 10:
                # Average the 10 samples for robust identity
                avg_feat = np.mean(np.stack(enroll_samples), axis=0)
                packed_id = binarize_and_pack(face_lsh.project(avg_feat))
                name = input("Enter name for this face: ").strip() or "Person"
                face_memory.learn(packed_id, name)
                face_memory.save(FACE_MEMORY_PATH)
                print(f"Enrolled '{name}' ({face_memory.size} faces stored)")
                enroll_samples = []

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            args.no_stdp = not args.no_stdp
            print(f"STDP: {'OFF' if args.no_stdp else 'ON'}")
        elif key == ord('r'):
            snn.reset()
            snn.neurochemistry = type(snn.neurochemistry)()
            print("SNN reset")
        elif key == ord('e'):
            show_emotions = not show_emotions
        elif key == ord('f'):
            # Start face enrollment: capture 10 frames
            print("Hold still — enrolling face (10 frames)...")
            enroll_samples = [face.get_identity_features(frame)] if face else []
            enroll_samples = [x for x in enroll_samples if x is not None]
        elif key == ord('d'):
            # Show known faces
            print(f"Known faces ({face_memory.size}):")
            for i in range(face_memory.size):
                print(f"  {i}: {face_memory._labels[i]}")

    cap.release()
    cv2.destroyAllWindows()
    if face is not None:
        face.close()

    # Save SNN brain (synaptic weights + neurochemistry)
    snn.save(SNN_SAVE_PATH)
    print(f"\nSNN brain saved to {SNN_SAVE_PATH}")

    # Save face memory
    if face_memory.size > 0:
        face_memory.save(FACE_MEMORY_PATH)
        print(f"Face memory saved: {face_memory.size} faces in {FACE_MEMORY_PATH}")

    # Print final stats
    print(f"Final neurochemistry: {snn.neurochemistry.to_dict()}")
    print(f"Weight drift: {abs(float(np.linalg.norm(snn.W_in)) - initial_weight_norm):.4f}")
    print(f"STDP was: {'enabled' if not args.no_stdp else 'disabled'}")


if __name__ == "__main__":
    main()
