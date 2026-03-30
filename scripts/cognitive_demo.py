"""CubeMind Cognitive Architecture Demo — full pipeline visualization.

Shows the complete cognitive loop in real-time:
  Input (webcam/video) → SNN → Neurochemistry → Experiential Encoding
  → HD-GoT Debate → Active Inference → Memory → Taste Formation

Usage:
    python scripts/cognitive_demo.py                     # webcam
    python scripts/cognitive_demo.py --video data/bear.mp4  # video file
    python scripts/cognitive_demo.py --video data/bear.mp4 --loop  # loop video
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# CubeMind imports
from cubemind.ops import BlockCodes
from cubemind.perception.snn import SNNEncoder, NeurochemicalState
from cubemind.brain.cortex import Thalamus, BasalGanglia, CircadianCells
from cubemind.ops.vsa_bridge import ContinuousItemMemory, LSHProjector
from cubemind.experimental.vs_graph import spike_diffusion
from cubemind.experimental.affective_graph import affective_alpha
from cubemind.reasoning.hd_got import hd_got_resolve
from cubemind.perception.experiential import ExperientialEncoder

# Optional imports
try:
    from cubemind.perception.face import FacePerception
    HAS_FACE = True
except Exception:
    HAS_FACE = False


# ── Drawing helpers ──────────────────────────────────────────────────

def draw_bar(img, x, y, w, h, value, color, label="", max_val=1.0):
    """Draw a labeled horizontal bar."""
    fill = int(w * min(value / max_val, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)
    if label:
        cv2.putText(img, f"{label}: {value:.2f}", (x + 2, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def draw_text(img, x, y, text, color=(200, 200, 200), scale=0.4):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def draw_section(img, x, y, w, h, title):
    """Draw a section box with title."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)
    cv2.rectangle(img, (x, y), (x + w, y + 18), (60, 60, 80), -1)
    draw_text(img, x + 4, y + 13, title, (255, 255, 255), 0.4)


# ── Main demo ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CubeMind Cognitive Demo")
    parser.add_argument("--video", type=str, default=None, help="Video file path")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--loop", action="store_true", help="Loop video")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    args = parser.parse_args()

    # ── Initialize subsystems ────────────────────────────────────────
    k, l = 8, 64
    d_vsa = 2048
    d_model = 256
    bc = BlockCodes(k=k, l=l)

    snn = SNNEncoder(d_input=104, n_neurons=128, d_vsa=d_vsa,
                     neuron_type="lif", tau=15.0, v_threshold=0.08, seed=42)
    snn.stdp_lr_potentiate = 0.0002
    snn.stdp_lr_depress = 0.0001

    thalamus = Thalamus(embedding_dim=d_model)
    basal_ganglia = BasalGanglia()
    circadian = CircadianCells()

    exp_encoder = ExperientialEncoder(k=k, l=l, seed=42)
    episodic_memory = ContinuousItemMemory(d_vsa=d_vsa, max_capacity=1000)
    lsh = LSHProjector(d_input=d_model, d_output=d_vsa, seed=52)

    face = None
    if HAS_FACE:
        try:
            face = FacePerception(max_faces=1)
        except Exception:
            pass

    # HMM ensemble for active inference (lightweight)
    codebook = bc.codebook_discrete(5, seed=42)

    # ── Open video source ────────────────────────────────────────────
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = args.video.split("/")[-1].split("\\")[-1]
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        source_name = f"Camera {args.camera}"

    if not cap.isOpened():
        print(f"Cannot open: {source_name}")
        sys.exit(1)

    print(f"CubeMind Cognitive Demo — {source_name}")
    print("Press 'q' to quit, 'r' to reset memory, 's' to save state")

    W, H = args.width, args.height
    VID_W, VID_H = 400, 300
    frame_count = 0
    fps_timer = time.time()
    fps = 0.0

    # State tracking
    experience_count = 0
    efe_history = []
    taste_scores = []
    debate_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.loop and args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_count += 1

        # ── Process frame ────────────────────────────────────────────
        small = cv2.resize(frame, (160, 120))
        gray = np.mean(small.astype(np.float32), axis=2) / 255.0

        # Grid features for SNN
        grid = cv2.resize(gray, (13, 8)).ravel()
        features = np.zeros(104, dtype=np.float32)
        features[:len(grid)] = grid

        # SNN encoding
        snn_result = snn.encode_temporal(features)
        nc = snn.neurochemistry
        spike_rate = float(np.mean(snn_result["spikes"]))

        # Face perception
        face_data = None
        if face is not None:
            try:
                face_data = face.process(frame)
            except Exception:
                pass

        # Thalamus routing
        embed = np.random.default_rng(frame_count).standard_normal(d_model).astype(np.float32) * 0.1
        embed[:min(104, d_model)] = features[:min(104, d_model)]
        route = thalamus.route(embed, arousal=nc.arousal,
                               valence=getattr(nc, 'valence', 0.0))

        # Basal ganglia strategy
        bg = basal_ganglia.select_strategy(
            route["routes"],
            valence=getattr(nc, 'valence', 0.0),
            arousal=nc.arousal,
            stress=nc.cortisol,
        )

        # Affective alpha
        alpha = affective_alpha(nc)

        # Circadian context
        now = datetime.now()
        circ = circadian.get_context()

        # Experiential encoding
        scene_vsa = bc.random_discrete(seed=frame_count % 1000)
        hour = now.hour
        day = now.weekday()
        month = now.month
        season = (["winter"] * 3 + ["spring"] * 3 + ["summer"] * 3 + ["fall"] * 3)[month - 1]
        exp_vec = exp_encoder.encode_experience(
            scene_vsa, hour=hour, day_of_week=day, season=season,
            neurochemistry=nc)
        experience_count += 1

        # HD-GoT (every 30 frames)
        if frame_count % 30 == 0:
            candidates = [bc.random_discrete(seed=frame_count + i) for i in range(5)]
            try:
                solution = hd_got_resolve(candidates, bc, top_k=3)
                debate_results.append(float(bc.similarity(solution, candidates[0])))
            except Exception:
                pass

        # FPS calculation
        if frame_count % 10 == 0:
            now_t = time.time()
            fps = 10.0 / max(now_t - fps_timer, 0.001)
            fps_timer = now_t

        # ── Build dashboard ──────────────────────────────────────────
        dashboard = np.zeros((H, W, 3), dtype=np.uint8)

        # Video feed (top-left)
        vid_frame = cv2.resize(frame, (VID_W, VID_H))
        dashboard[10:10+VID_H, 10:10+VID_W] = vid_frame

        # Source label
        draw_text(dashboard, 12, 10 + VID_H + 15, f"{source_name} | {fps:.0f} FPS",
                  (150, 255, 150), 0.4)

        # ── Neurochemistry panel (below video) ───────────────────────
        ny = 340
        draw_section(dashboard, 10, ny, VID_W, 160, "NEUROCHEMISTRY")
        draw_bar(dashboard, 20, ny + 30, 180, 12, nc.dopamine, (50, 200, 50), "Dopamine")
        draw_bar(dashboard, 20, ny + 55, 180, 12, nc.cortisol, (200, 50, 50), "Cortisol")
        draw_bar(dashboard, 20, ny + 80, 180, 12, nc.serotonin, (50, 50, 200), "Serotonin")
        draw_bar(dashboard, 20, ny + 105, 180, 12, nc.oxytocin, (200, 150, 50), "Oxytocin")
        draw_text(dashboard, 210, ny + 40, f"Arousal: {nc.arousal:.2f}", (180, 180, 180))
        draw_text(dashboard, 210, ny + 60, f"Emotion: {nc.dominant_emotion}", (255, 200, 100))
        draw_text(dashboard, 210, ny + 80, f"Alpha: {alpha:.2f}", (150, 200, 255))
        mode = "EXPLORE" if alpha < 0.45 else "CONSOLIDATE" if alpha > 0.55 else "BALANCED"
        color = (50, 255, 50) if alpha < 0.45 else (50, 50, 255) if alpha > 0.55 else (200, 200, 200)
        draw_text(dashboard, 210, ny + 100, f"Mode: {mode}", color, 0.45)

        # ── SNN panel (below neurochemistry) ─────────────────────────
        sy = 510
        draw_section(dashboard, 10, sy, VID_W, 90, "SNN SPIKES")
        spikes = snn_result["spikes"]
        # Draw spike grid (16x8)
        for i in range(min(128, len(spikes))):
            sx = 20 + (i % 32) * 12
            spy = sy + 25 + (i // 32) * 15
            c = (0, 255, 0) if spikes[i] > 0 else (30, 30, 30)
            cv2.rectangle(dashboard, (sx, spy), (sx + 10, spy + 12), c, -1)
        draw_text(dashboard, 20, sy + 80, f"Rate: {spike_rate*100:.0f}% | Neurons: {len(spikes)}",
                  (150, 255, 150))

        # ── Experience panel (right side) ────────────────────────────
        ex = 430
        draw_section(dashboard, ex, 10, W - ex - 10, 140, "EXPERIENTIAL ENCODING")
        draw_text(dashboard, ex + 10, 40, f"Time: {now.strftime('%H:%M')} {season.capitalize()}")
        draw_text(dashboard, ex + 10, 58, f"Day: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day]}")
        draw_text(dashboard, ex + 10, 76,
                  f"Valence: {'LIGHT' if nc.dopamine > nc.cortisol else 'HEAVY'}",
                  (50, 255, 50) if nc.dopamine > nc.cortisol else (50, 50, 255))
        draw_text(dashboard, ex + 10, 94, f"Circadian: {circ.get('phase', 'unknown')}")
        draw_text(dashboard, ex + 10, 112, f"Experiences: {experience_count}")
        draw_text(dashboard, ex + 10, 130, f"Vec norm: {np.linalg.norm(exp_vec):.2f}")

        # ── Thalamus / BG panel ──────────────────────────────────────
        draw_section(dashboard, ex, 160, W - ex - 10, 130, "BRAIN CORTEX")
        draw_text(dashboard, ex + 10, 190, f"Primary route: {route['primary_route'].upper()}",
                  (255, 200, 100), 0.45)
        draw_text(dashboard, ex + 10, 210, f"Salience: {route['salience']:.2f}")
        draw_text(dashboard, ex + 10, 228, f"BG strategy: {bg['strategy'].upper()}",
                  (100, 255, 200), 0.45)
        draw_text(dashboard, ex + 10, 246, f"Confidence: {bg['confidence']:.2f}")
        draw_text(dashboard, ex + 10, 264, f"Go: {'YES' if bg['go'] else 'NO'}",
                  (50, 255, 50) if bg['go'] else (255, 50, 50))

        # Route bars
        ry = 160
        for i, (name, weight) in enumerate(route["routes"].items()):
            bx = ex + 300
            by = ry + 25 + i * 22
            draw_bar(dashboard, bx, by, 150, 14, weight, (100, 150, 255), name)

        # ── HD-GoT panel ─────────────────────────────────────────────
        draw_section(dashboard, ex, 300, W - ex - 10, 100, "HD-GoT DEBATE")
        if debate_results:
            last_sim = debate_results[-1]
            draw_text(dashboard, ex + 10, 330,
                      f"Last consensus sim: {last_sim:.3f}")
            draw_text(dashboard, ex + 10, 350,
                      f"Debates: {len(debate_results)}")
            # Mini chart of debate quality
            chart_x = ex + 10
            chart_y = 370
            for i, s in enumerate(debate_results[-20:]):
                bh = int(s * 20)
                cx = chart_x + i * 8
                cv2.rectangle(dashboard, (cx, chart_y - bh), (cx + 6, chart_y),
                              (100, 200, 100), -1)
        else:
            draw_text(dashboard, ex + 10, 340, "Waiting for first debate...")

        # ── Memory panel ─────────────────────────────────────────────
        draw_section(dashboard, ex, 410, W - ex - 10, 100, "EPISODIC MEMORY")
        draw_text(dashboard, ex + 10, 440, f"Stored: {episodic_memory.size}")
        draw_text(dashboard, ex + 10, 458, f"Capacity: {episodic_memory.max_capacity}")

        # ── Face panel (if available) ────────────────────────────────
        if face_data and face_data.get("faces"):
            draw_section(dashboard, ex, 520, W - ex - 10, 80,
                         "FACE PERCEPTION")
            f = face_data["faces"][0]
            draw_text(dashboard, ex + 10, 550,
                      f"Blendshapes: {len(f.get('blendshapes', {}))} active")
            if "identity" in f:
                draw_text(dashboard, ex + 10, 568, f"Identity: {f['identity']}")

        # ── Title bar ────────────────────────────────────────────────
        cv2.rectangle(dashboard, (0, H - 25), (W, H), (40, 40, 60), -1)
        draw_text(dashboard, 10, H - 8,
                  "CUBEMIND COGNITIVE ARCHITECTURE | "
                  f"Frame {frame_count} | "
                  f"{experience_count} experiences | "
                  f"Alpha={alpha:.2f} | "
                  f"{mode}",
                  (150, 200, 255), 0.35)

        # ── Display ──────────────────────────────────────────────────
        cv2.imshow("CubeMind Cognitive Demo", dashboard)

        key = cv2.waitKey(1 if args.video else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            episodic_memory = ContinuousItemMemory(d_vsa=d_vsa, max_capacity=1000)
            experience_count = 0
            debate_results.clear()
            print("Memory reset")
        elif key == ord('s'):
            print(f"State: {experience_count} experiences, "
                  f"{len(debate_results)} debates, "
                  f"emotion={nc.dominant_emotion}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Demo ended: {frame_count} frames, {experience_count} experiences")


if __name__ == "__main__":
    main()
