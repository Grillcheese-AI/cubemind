"""CubeMind Teaching Demo — see + hear = knowledge.

Dual-source: video (bear.mp4) as main display + webcam as overlay.
User points at screen and says a word → system binds visual + audio + label
into experiential memory. Later user asks "what is it?" and the system
retrieves the concept from memory via visual similarity.

This is how humans teach babies: show + name = learn.

Usage:
    python scripts/teaching_demo.py --video data/bear.mp4
    python scripts/teaching_demo.py --video data/bear.mp4 --camera 1
    python scripts/teaching_demo.py  # webcam only

Controls:
    T + type label + Enter: Teach current frame with label
    M: Hold to listen (microphone) → release → teach with last label
    R: Recall — ask "what is it?" about current frame
    Space: Pause/resume video
    S: Screenshot + save state
    Q: Quit
"""

from __future__ import annotations

import argparse
import sys
import time
import threading
from datetime import datetime
from collections import deque

import cv2
import numpy as np

# CubeMind imports
from cubemind.ops import BlockCodes
from cubemind.perception.snn import SNNEncoder, NeurochemicalState
from cubemind.brain.cortex import Thalamus, BasalGanglia, CircadianCells
from cubemind.ops.vsa_bridge import (
    ContinuousItemMemory, LSHProjector, binarize_and_pack,
)
from cubemind.experimental.affective_graph import affective_alpha
from cubemind.perception.experiential import ExperientialEncoder

# Optional: microphone + audio encoder
_HAS_MIC = False
_MicrophoneCapture = None
_AudioEncoder = None
try:
    from cubemind.perception.audio import MicrophoneCapture as _MC, AudioEncoder as _AE
    _MicrophoneCapture = _MC
    _AudioEncoder = _AE
    _HAS_MIC = True
except Exception:
    pass


# ── Drawing helpers ──────────────────────────────────────────────────

def draw_bar(img, x, y, w, h, value, color, label=""):
    fill = int(w * min(value, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)
    if label:
        cv2.putText(img, f"{label}: {value:.2f}", (x + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)


def draw_text(img, x, y, text, color=(200, 200, 200), scale=0.4, thick=1):
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def draw_box(img, x, y, w, h, title, border=(60, 60, 80)):
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 25), -1)
    cv2.rectangle(img, (x, y), (x + w, y + 20), border, -1)
    draw_text(img, x + 5, y + 14, title, (255, 255, 255), 0.4)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)


# ── Teaching Mind (simplified for demo) ──────────────────────────────

class TeachingMind:
    """Simplified Mind for the teaching demo — learn and recall."""

    def __init__(self, d_model: int = 256, d_vsa: int = 2048, seed: int = 42):
        self.d_model = d_model
        self.d_vsa = d_vsa
        self.bc = BlockCodes(k=8, l=64)

        self.snn = SNNEncoder(
            d_input=104, n_neurons=128, d_vsa=d_vsa,
            neuron_type="lif", tau=15.0, v_threshold=0.08, seed=seed)
        self.snn.stdp_lr_potentiate = 0.0002

        self.thalamus = Thalamus(embedding_dim=d_model)
        self.basal_ganglia = BasalGanglia()
        self.circadian = CircadianCells()
        self.exp_encoder = ExperientialEncoder(k=8, l=64, seed=seed)
        self.lsh = LSHProjector(d_input=d_model, d_output=d_vsa, seed=seed + 10)

        self.semantic_memory = ContinuousItemMemory(d_vsa=d_vsa, max_capacity=1000)
        self.episodic_memory = ContinuousItemMemory(d_vsa=d_vsa, max_capacity=5000)

        # Audio
        self.mic = None
        self.audio_encoder = None
        if _HAS_MIC:
            try:
                self.mic = _MicrophoneCapture(sample_rate=16000, chunk_size=512)
                self.audio_encoder = _AudioEncoder(
                    sample_rate=16000, snn_neurons=128, d_vsa=d_vsa, seed=seed + 20)
                print("Microphone: ready (hold M to listen)")
            except Exception as e:
                print(f"Microphone: not available ({e})")
        self.is_listening = False
        self.audio_buffer: list[np.ndarray] = []
        self.voice_energy = 0.0

        # Teaching log
        self.concepts: dict[str, dict] = {}
        self.teach_count = 0
        self.recall_log: list[dict] = []

    def start_listening(self):
        """Start capturing audio from microphone."""
        if self.mic is None:
            return
        try:
            self.mic.start()
            self.is_listening = True
            self.audio_buffer.clear()
        except Exception:
            pass

    def stop_listening(self) -> np.ndarray | None:
        """Stop capturing and return audio buffer."""
        if self.mic is None or not self.is_listening:
            return None
        self.is_listening = False
        try:
            audio = self.mic.stop()
            if audio is not None and len(audio) > 0:
                self.voice_energy = float(np.sqrt(np.mean(audio ** 2)))
                return audio
        except Exception:
            pass
        return None

    def process_audio_chunk(self):
        """Read a chunk from mic and update energy meter."""
        if self.mic is None or not self.is_listening:
            return
        try:
            chunk = self.mic.read_chunk()
            if chunk is not None:
                self.audio_buffer.append(chunk)
                self.voice_energy = float(np.sqrt(np.mean(chunk ** 2)))
        except Exception:
            pass

    def teach_with_audio(self, frame: np.ndarray, label: str,
                          audio: np.ndarray | None = None) -> dict:
        """Teach with visual + audio + label (full multi-modal binding)."""
        result = self.teach(frame, label)

        # If we have audio, also encode and store it
        if audio is not None and self.audio_encoder is not None and len(audio) > 1600:
            try:
                audio_packed = self.audio_encoder.encode_audio(audio)
                self.episodic_memory.learn(audio_packed, label=f"audio:{label}")
                result["audio_bound"] = True
                result["audio_energy"] = float(np.sqrt(np.mean(audio ** 2)))
            except Exception:
                result["audio_bound"] = False
        return result

    def _frame_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract features from a frame."""
        small = cv2.resize(frame, (13, 8))
        if small.ndim == 3:
            gray = np.mean(small.astype(np.float32), axis=2) / 255.0
        else:
            gray = small.astype(np.float32) / 255.0
        grid = gray.ravel()
        features = np.zeros(104, dtype=np.float32)
        features[:len(grid)] = grid[:104]
        return features

    def perceive(self, frame: np.ndarray) -> dict:
        """Run perception pipeline on a frame."""
        features = self._frame_features(frame)
        snn_result = self.snn.encode_temporal(features)
        nc = self.snn.neurochemistry

        embed = np.zeros(self.d_model, dtype=np.float32)
        embed[:min(104, self.d_model)] = features[:min(104, self.d_model)]

        route = self.thalamus.route(embed, arousal=nc.arousal,
                                     valence=getattr(nc, 'valence', 0.0))
        alpha = affective_alpha(nc)

        return {
            "features": features,
            "embed": embed,
            "spikes": snn_result["spikes"],
            "spike_rate": float(np.mean(snn_result["spikes"])),
            "nc": nc,
            "route": route,
            "alpha": alpha,
        }

    def teach(self, frame: np.ndarray, label: str) -> dict:
        """Teach: bind visual features + label + emotion → memory."""
        self.teach_count += 1
        perc = self.perceive(frame)
        nc = perc["nc"]

        # Visual feature vector
        vis_feat = perc["embed"][:self.lsh.d_input]
        if len(vis_feat) < self.lsh.d_input:
            vis_feat = np.pad(vis_feat, (0, self.lsh.d_input - len(vis_feat)))

        # Text vector (hash-based since we don't have full semantic encoder)
        rng = np.random.default_rng(hash(label) % (2**31))
        text_feat = rng.standard_normal(self.lsh.d_input).astype(np.float32) * 0.1

        # Experiential vector
        scene_vsa = self.bc.random_discrete(seed=hash(label) % (2**31))
        now = datetime.now()
        exp_vec = self.exp_encoder.encode_experience(
            scene_vsa, hour=now.hour, day_of_week=now.weekday(),
            season="spring", neurochemistry=nc)

        # XOR binding: visual ⊕ text ⊕ emotion
        vis_packed = binarize_and_pack(self.lsh.project(vis_feat))
        text_packed = binarize_and_pack(self.lsh.project(text_feat))
        concept = np.bitwise_xor(vis_packed, text_packed)

        # Store in both memories
        self.semantic_memory.learn(concept, label=label)
        self.episodic_memory.learn(vis_packed, label=f"vision:{label}")

        # Track concept
        self.concepts[label] = {
            "count": self.concepts.get(label, {}).get("count", 0) + 1,
            "emotion": nc.dominant_emotion,
            "last_seen": time.time(),
        }

        return {
            "label": label,
            "emotion": nc.dominant_emotion,
            "semantic_size": self.semantic_memory.size,
            "episodic_size": self.episodic_memory.size,
            "alpha": perc["alpha"],
        }

    def recall(self, frame: np.ndarray, k: int = 3) -> list[dict]:
        """Recall: "what is this?" — search memory by visual similarity."""
        perc = self.perceive(frame)
        vis_feat = perc["embed"][:self.lsh.d_input]
        if len(vis_feat) < self.lsh.d_input:
            vis_feat = np.pad(vis_feat, (0, self.lsh.d_input - len(vis_feat)))
        packed = binarize_and_pack(self.lsh.project(vis_feat))

        results = []
        for mem, mem_type in [(self.semantic_memory, "semantic"),
                               (self.episodic_memory, "episodic")]:
            if mem.size > 0:
                hits = mem.retrieve(packed, k=k)
                for idx, sim, label in hits:
                    results.append({
                        "label": label.replace("vision:", ""),
                        "similarity": sim,
                        "memory": mem_type,
                    })

        # Deduplicate and sort
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x["similarity"], reverse=True):
            lbl = r["label"]
            if lbl not in seen:
                seen.add(lbl)
                unique.append(r)

        self.recall_log.append({
            "time": time.time(),
            "results": unique[:k],
            "emotion": perc["nc"].dominant_emotion,
        })

        return unique[:k]


# ── Main demo ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CubeMind Teaching Demo")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    W, H = 1280, 720
    VID_W, VID_H = 640, 480
    CAM_W, CAM_H = 200, 150

    mind = TeachingMind()

    # Open video source
    vid_cap = None
    if args.video:
        vid_cap = cv2.VideoCapture(args.video)
        if not vid_cap.isOpened():
            print(f"Cannot open video: {args.video}")
            sys.exit(1)

    # Open webcam
    cam_cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cam_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("=" * 60)
    print("  CubeMind Teaching Demo")
    print("=" * 60)
    print("  T + label + Enter : Teach (name what you see)")
    print("  R                 : Recall (what is this?)")
    print("  Space             : Pause/resume video")
    print("  Q                 : Quit")
    print("=" * 60)

    paused = False
    input_mode = False
    input_buffer = ""
    frame_count = 0
    last_teach = None
    last_recall = None
    recall_display_until = 0
    teach_display_until = 0
    fps_timer = time.time()
    fps = 0.0

    # Message log
    messages = deque(maxlen=8)
    messages.append(("Ready. Press T to teach, R to recall.", (150, 255, 150)))

    while True:
        frame_count += 1

        # Read video frame
        vid_frame = None
        if vid_cap is not None and not paused:
            ret, vid_frame = vid_cap.read()
            if not ret:
                if args.loop:
                    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    _, vid_frame = vid_cap.read()
                else:
                    vid_frame = np.zeros((VID_H, VID_W, 3), dtype=np.uint8)

        if vid_frame is None:
            vid_frame = np.zeros((VID_H, VID_W, 3), dtype=np.uint8)

        # Read webcam
        cam_frame = None
        ret, cam_raw = cam_cap.read()
        if ret:
            cam_frame = cam_raw

        # Decide which frame to process (video if available, else webcam)
        process_frame = vid_frame if vid_cap else (cam_frame if cam_frame is not None
                                                    else np.zeros((240, 320, 3), dtype=np.uint8))

        # Perceive
        perc = mind.perceive(process_frame)
        nc = perc["nc"]

        # FPS
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10.0 / max(now - fps_timer, 0.001)
            fps_timer = now

        # ── Build dashboard ──────────────────────────────────────────
        dash = np.zeros((H, W, 3), dtype=np.uint8)

        # Main video display
        vid_display = cv2.resize(vid_frame if vid_cap else process_frame, (VID_W, VID_H))
        dash[10:10 + VID_H, 10:10 + VID_W] = vid_display

        # Webcam overlay (picture-in-picture, top-right of video)
        if cam_frame is not None:
            cam_small = cv2.resize(cam_frame, (CAM_W, CAM_H))
            ox = 10 + VID_W - CAM_W - 10
            oy = 20
            dash[oy:oy + CAM_H, ox:ox + CAM_W] = cam_small
            cv2.rectangle(dash, (ox - 1, oy - 1),
                          (ox + CAM_W, oy + CAM_H), (0, 255, 0), 1)
            draw_text(dash, ox + 2, oy + CAM_H + 12, "YOU", (0, 255, 0), 0.35)

        # Audio level meter (if listening)
        if mind.is_listening:
            mind.process_audio_chunk()
            av_x, av_y = 15, 10 + VID_H - 30
            cv2.rectangle(dash, (av_x, av_y), (av_x + 200, av_y + 20), (30, 30, 30), -1)
            bar_w = int(min(mind.voice_energy * 2000, 200))
            bar_color = (0, 255, 0) if mind.voice_energy > 0.01 else (100, 100, 100)
            cv2.rectangle(dash, (av_x, av_y), (av_x + bar_w, av_y + 20), bar_color, -1)
            draw_text(dash, av_x + 205, av_y + 14, "MIC ACTIVE", (255, 100, 255), 0.4)

        # Key legend (bottom of video area)
        ky = 10 + VID_H + 30
        keys = [
            ("[T] Teach", (100, 255, 100)),
            ("[R] Recall", (100, 200, 255)),
            ("[M] Mic", (255, 100, 255) if mind.mic else (80, 80, 80)),
            ("[Space] Pause", (255, 200, 100)),
            ("[S] Save", (200, 200, 255)),
            ("[Q] Quit", (255, 100, 100)),
        ]
        kx = 15
        for label, color in keys:
            draw_text(dash, kx, ky, label, color, 0.32)
            kx += 90

        # Source label
        src = args.video.split("/")[-1].split("\\")[-1] if args.video else "Webcam"
        status = "PAUSED" if paused else f"{fps:.0f} FPS"
        draw_text(dash, 15, 10 + VID_H + 18, f"{src} | {status}",
                  (150, 200, 150) if not paused else (255, 150, 50), 0.4)

        # ── Right panel ──────────────────────────────────────────────
        px = VID_W + 30

        # Neurochemistry
        draw_box(dash, px, 10, W - px - 10, 120, "NEUROCHEMISTRY")
        draw_bar(dash, px + 10, 42, 150, 10, nc.dopamine, (50, 200, 50), "D")
        draw_bar(dash, px + 10, 60, 150, 10, nc.cortisol, (200, 50, 50), "C")
        draw_bar(dash, px + 10, 78, 150, 10, nc.serotonin, (50, 50, 200), "S")
        draw_bar(dash, px + 10, 96, 150, 10, nc.oxytocin, (200, 150, 50), "O")
        draw_text(dash, px + 180, 50, f"{nc.dominant_emotion}", (255, 200, 100), 0.45)
        alpha = perc["alpha"]
        mode = "EXPLORE" if alpha < 0.45 else "HOLD" if alpha > 0.55 else "BALANCED"
        mode_color = (50, 255, 50) if alpha < 0.45 else (50, 100, 255) if alpha > 0.55 else (180, 180, 180)
        draw_text(dash, px + 180, 70, f"a={alpha:.2f} {mode}", mode_color, 0.35)
        draw_text(dash, px + 180, 90, f"Route: {perc['route']['primary_route']}", (180, 180, 180), 0.35)

        # Memory status
        draw_box(dash, px, 140, W - px - 10, 80, "MEMORY")
        draw_text(dash, px + 10, 170,
                  f"Concepts: {len(mind.concepts)}", (200, 200, 255))
        draw_text(dash, px + 10, 190,
                  f"Semantic: {mind.semantic_memory.size} | "
                  f"Episodic: {mind.episodic_memory.size}")
        draw_text(dash, px + 10, 210,
                  f"Taught: {mind.teach_count} times")

        # Known concepts
        draw_box(dash, px, 230, W - px - 10, 120, "KNOWN CONCEPTS")
        cy = 255
        for i, (label, info) in enumerate(list(mind.concepts.items())[-5:]):
            draw_text(dash, px + 10, cy,
                      f"  {label} (x{info['count']}, {info['emotion']})",
                      (200, 255, 200), 0.35)
            cy += 18

        # Last teach result
        draw_box(dash, px, 360, W - px - 10, 80, "LAST TEACH")
        if last_teach and time.time() < teach_display_until:
            draw_text(dash, px + 10, 388,
                      f'Learned: "{last_teach["label"]}"',
                      (50, 255, 50), 0.45)
            draw_text(dash, px + 10, 408,
                      f'Emotion: {last_teach["emotion"]}')
            draw_text(dash, px + 10, 425,
                      f'Memory: {last_teach["semantic_size"]} concepts')
        else:
            draw_text(dash, px + 10, 395,
                      "Press T + label + Enter to teach", (120, 120, 120))

        # Last recall result
        draw_box(dash, px, 450, W - px - 10, 100, "LAST RECALL")
        if last_recall and time.time() < recall_display_until:
            for i, r in enumerate(last_recall[:3]):
                color = (50, 255, 50) if r["similarity"] > 0.6 else (255, 200, 50)
                draw_text(dash, px + 10, 478 + i * 22,
                          f'  "{r["label"]}" sim={r["similarity"]:.3f} [{r["memory"]}]',
                          color, 0.38)
        else:
            draw_text(dash, px + 10, 485,
                      "Press R to ask: what is this?", (120, 120, 120))

        # SNN spikes (bottom-left strip)
        sy = VID_H + 35
        draw_box(dash, 10, sy, VID_W, 60, "SNN SPIKES")
        spikes = perc["spikes"]
        for i in range(min(128, len(spikes))):
            sx = 15 + (i % 64) * 9
            spy = sy + 25 + (i // 64) * 14
            c = (0, 255, 0) if spikes[i] > 0 else (25, 25, 25)
            cv2.rectangle(dash, (sx, spy), (sx + 7, spy + 11), c, -1)

        # Message log (bottom-right)
        draw_box(dash, px, 560, W - px - 10, H - 570, "LOG")
        for i, (msg, color) in enumerate(messages):
            draw_text(dash, px + 8, 582 + i * 16, msg, color, 0.3)

        # Input mode indicator
        if input_mode:
            cv2.rectangle(dash, (10, H - 35), (VID_W + 10, H - 5), (0, 80, 0), -1)
            draw_text(dash, 15, H - 14,
                      f"TEACH> {input_buffer}_",
                      (0, 255, 0), 0.5, 2)

        # Title bar
        cv2.rectangle(dash, (0, 0), (W, 8), (50, 50, 70), -1)

        # ── Display ──────────────────────────────────────────────────
        cv2.imshow("CubeMind Teaching Demo", dash)

        key = cv2.waitKey(1 if vid_cap else 30) & 0xFF

        if input_mode:
            if key == 13:  # Enter
                label = input_buffer.strip()
                if label:
                    result = mind.teach(process_frame, label)
                    last_teach = result
                    teach_display_until = time.time() + 5.0
                    messages.append(
                        (f"Taught: '{label}' ({result['emotion']})",
                         (50, 255, 50)))
                    print(f"  Taught: '{label}' → {result}")
                input_mode = False
                input_buffer = ""
            elif key == 27:  # Escape
                input_mode = False
                input_buffer = ""
            elif key == 8:  # Backspace
                input_buffer = input_buffer[:-1]
            elif 32 <= key <= 126:
                input_buffer += chr(key)
        else:
            if key == ord('q'):
                break
            elif key == ord('t'):
                input_mode = True
                input_buffer = ""
                messages.append(("Type label and press Enter...", (255, 255, 100)))
            elif key == ord('m'):
                # Hold M to listen — start recording
                if not mind.is_listening and mind.mic is not None:
                    mind.start_listening()
                    messages.append(("Listening... release M when done",
                                     (255, 100, 255)))
            elif key == ord('r'):
                results = mind.recall(process_frame)
                last_recall = results
                recall_display_until = time.time() + 8.0
                if results:
                    best = results[0]
                    messages.append(
                        (f"Recall: '{best['label']}' ({best['similarity']:.3f})",
                         (100, 200, 255)))
                    print(f"  Recall: {results}")
                else:
                    messages.append(("No memories yet — teach me first!",
                                     (255, 100, 100)))
            elif key == ord(' '):
                paused = not paused
                messages.append(
                    ("Paused" if paused else "Resumed", (255, 200, 100)))

            # M key release: if we were listening and M is no longer held
            if mind.is_listening and key != ord('m'):
                audio = mind.stop_listening()
                if audio is not None and mind.voice_energy > 0.005:
                    # Audio captured — teach with last known label or prompt
                    if mind.concepts:
                        # Re-teach last concept with audio binding
                        last_label = list(mind.concepts.keys())[-1]
                        result = mind.teach_with_audio(
                            process_frame, last_label, audio)
                        messages.append(
                            (f"Audio+Visual: '{last_label}' (E={mind.voice_energy:.3f})",
                             (255, 100, 255)))
                        last_teach = result
                        teach_display_until = time.time() + 5.0
                    else:
                        messages.append(
                            ("Heard you! Teach a label first with T, then use M.",
                             (255, 200, 100)))
                elif audio is not None:
                    messages.append(("Too quiet — speak louder!", (255, 100, 100)))
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"cubemind_teach_{ts}.png", dash)
                messages.append((f"Saved screenshot: cubemind_teach_{ts}.png",
                                 (200, 200, 255)))

    cam_cap.release()
    if vid_cap:
        vid_cap.release()
    cv2.destroyAllWindows()

    print(f"\nSession summary:")
    print(f"  Frames: {frame_count}")
    print(f"  Concepts taught: {len(mind.concepts)}")
    for label, info in mind.concepts.items():
        print(f"    {label}: taught {info['count']}x, emotion={info['emotion']}")
    print(f"  Recalls: {len(mind.recall_log)}")
    print(f"  Semantic memory: {mind.semantic_memory.size}")
    print(f"  Episodic memory: {mind.episodic_memory.size}")


if __name__ == "__main__":
    main()
