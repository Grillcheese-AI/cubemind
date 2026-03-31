"""SRT Auto-Teacher — unsupervised concept learning from narrated video.

Feeds video frames + SRT subtitles through CubeMind's perception pipeline.
The system learns concepts naturally from the narration without human guidance.

For each subtitle window:
  1. Extract the video frame at that timestamp
  2. Run SNN perception → neurochemistry reacts to visual content
  3. Extract nouns/keywords from subtitle text
  4. Bind visual features + text labels + emotion → memory
  5. Record the experiential vector (time + affect + circadian)

After processing, the model has learned every concept mentioned in the
narration, grounded in what it actually saw and felt at that moment.

Usage:
    python scripts/srt_teacher.py --video data/bear2.mp4 --srt data/bear2.srt
    python scripts/srt_teacher.py --video data/music_video_explosions.mp4 --srt data/music_video_explosions.srt
    python scripts/srt_teacher.py --video data/bear2.mp4 --srt data/bear2.srt --save data/bear2_memory.npz
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from cubemind.ops import BlockCodes
from cubemind.perception.snn import SNNEncoder, NeurochemicalState
from cubemind.brain.cortex import Thalamus, BasalGanglia
from cubemind.ops.vsa_bridge import (
    ContinuousItemMemory, LSHProjector, binarize_and_pack,
)
from cubemind.experimental.affective_graph import affective_alpha
from cubemind.perception.experiential import ExperientialEncoder
from cubemind.perception.bio_vision import BioVisionEncoder
from cubemind.perception.color import extract_color_stats, color_to_neurochemistry


# ── SRT Parser ───────────────────────────────────────────────────────

@dataclass
class SubtitleEntry:
    index: int
    start_s: float
    end_s: float
    text: str


def parse_srt(path: str) -> list[SubtitleEntry]:
    """Parse .srt file into list of timed subtitle entries."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    entries = []
    blocks = re.split(r"\n\n+", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue

        # Parse timestamp: 00:01:43,440 --> 00:01:49,440
        ts_match = re.match(
            r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)",
            lines[1].strip(),
        )
        if not ts_match:
            continue

        g = ts_match.groups()
        start_s = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
        end_s = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000

        text = " ".join(lines[2:]).strip()
        if text and text != "[Music]":
            entries.append(SubtitleEntry(idx, start_s, end_s, text))

    return entries


# ── Keyword Extraction (simple, no dependencies) ─────────────────────

# Common words to skip
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "don", "now", "its",
    "it", "they", "them", "their", "this", "that", "these", "those",
    "and", "but", "or", "if", "while", "because", "until", "about",
    "what", "which", "who", "whom", "up", "down", "every", "any",
    "keep", "walking", "walk", "look", "don't", "got", "yeah",
}


def extract_keywords(text: str, min_len: int = 3) -> list[str]:
    """Extract meaningful words from subtitle text."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    keywords = [
        w for w in words
        if len(w) >= min_len and w not in STOP_WORDS
    ]
    # Deduplicate preserving order
    seen = set()
    unique = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


# ── Main Pipeline ────────────────────────────────────────────────────

def run_srt_teacher(
    video_path: str,
    srt_path: str,
    save_path: str | None = None,
    sample_every_n: int = 1,
    verbose: bool = True,
) -> dict:
    """Process video + SRT through CubeMind's perception pipeline.

    Returns dict with all recorded data (neurochemistry, concepts, etc.)
    """
    # Parse SRT
    entries = parse_srt(srt_path)
    if verbose:
        print(f"Parsed {len(entries)} subtitle entries from {srt_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    if verbose:
        print(f"Video: {video_path} ({duration:.1f}s, {fps:.1f}fps)")

    # Init CubeMind with biological vision
    d_model = 256
    d_vsa = 2048
    bc = BlockCodes(k=8, l=64)

    bio_vision = BioVisionEncoder(grid_h=8, grid_w=13, n_directions=8,
                                   maturity=0.2)  # Start as baby brain
    d_input = bio_vision.d_features

    snn = SNNEncoder(
        d_input=d_input, n_neurons=128, d_vsa=d_vsa,
        neuron_type="lif", tau=15.0, v_threshold=0.08, seed=42)
    snn.stdp_lr_potentiate = 0.0002

    thalamus = Thalamus(embedding_dim=d_model)
    basal_ganglia = BasalGanglia()
    exp_encoder = ExperientialEncoder(k=8, l=64, seed=42)
    lsh = LSHProjector(d_input=d_model, d_output=d_vsa, seed=52)
    semantic_memory = ContinuousItemMemory(d_vsa=d_vsa, max_capacity=10000)
    episodic_memory = ContinuousItemMemory(d_vsa=d_vsa, max_capacity=50000)

    # Recording
    log = {
        "video": video_path,
        "srt": srt_path,
        "entries_processed": 0,
        "concepts_learned": {},
        "timeline": [],
    }

    t0 = time.time()
    prev_color_stats = None

    for i, entry in enumerate(entries):
        if i % sample_every_n != 0:
            continue

        # Seek to subtitle midpoint
        mid_time = (entry.start_s + entry.end_s) / 2
        frame_num = int(mid_time * fps)
        if frame_num >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        # ALL channels fire in parallel on the same frame:

        # 1. Bio vision: opponent-color + motion + luminance (parallel channels)
        features = bio_vision.process(frame)

        # 2. Color perception: wavelength → neurochemical drives (parallel)
        color_stats = extract_color_stats(frame)
        color_drive = color_to_neurochemistry(color_stats, prev_stats=prev_color_stats)
        prev_color_stats = color_stats

        # 3. SNN: all bio vision features → spikes (parallel neurons)
        feat_norm = features / (np.std(features) + 1e-6) * 0.3
        spikes = snn.step(feat_norm)
        nc = snn.neurochemistry
        spike_rate = float(np.mean(spikes))

        # 4. Novel concept detection: first-seen keywords spike dopamine
        novel_concepts = [kw for kw in keywords if kw not in log["concepts_learned"]]
        concept_novelty = min(1.0, len(novel_concepts) * 0.3) if novel_concepts else 0.0

        # 5. Neurochemistry: fuse ALL signals simultaneously
        fused_novelty = max(spike_rate, color_drive["novelty"], concept_novelty)

        # Serotonin-curiosity coupling: high serotonin + novel input = explore boost
        # (calm state + surprise = heightened receptivity, not suppression)
        if nc.serotonin > 0.6 and fused_novelty > 0.2:
            fused_novelty = min(1.0, fused_novelty + nc.serotonin * 0.15)

        nc.update(
            novelty=fused_novelty,
            threat=color_drive["threat"],
            focus=color_drive["focus"],
            valence=color_drive["valence"],
        )

        # Dopamine floor: prevent total habituation (biological minimum)
        nc.dopamine = max(nc.dopamine, 0.15)

        # 6. Developmental growth: STDP pruning sharpens tuning
        bio_vision.grow(delta=0.001)

        # Thalamus routing
        embed = np.zeros(d_model, dtype=np.float32)
        n_copy = min(len(features), d_model)
        embed[:n_copy] = features[:n_copy]
        route = thalamus.route(embed, arousal=nc.arousal,
                               valence=getattr(nc, "valence", 0.0))

        # Basal ganglia
        bg = basal_ganglia.select_strategy(
            route["routes"], valence=getattr(nc, "valence", 0.0),
            arousal=nc.arousal, stress=nc.cortisol)

        # Affective alpha
        alpha = affective_alpha(nc)

        # Extract keywords from subtitle
        keywords = extract_keywords(entry.text)

        # Experiential encoding
        now = datetime.now()
        seasons = (["winter"] * 3 + ["spring"] * 3
                   + ["summer"] * 3 + ["fall"] * 3)
        exp_vec = exp_encoder.encode_experience(
            bc.random_discrete(seed=frame_num),
            hour=now.hour, day_of_week=now.weekday(),
            season=seasons[now.month - 1], neurochemistry=nc)

        # Teach each keyword — bind visual + text + emotion
        for keyword in keywords:
            vis_feat = embed[:lsh.d_input]
            if len(vis_feat) < lsh.d_input:
                vis_feat = np.pad(vis_feat, (0, lsh.d_input - len(vis_feat)))

            rng = np.random.default_rng(hash(keyword) % (2**31))
            text_feat = rng.standard_normal(lsh.d_input).astype(np.float32) * 0.1

            vis_packed = binarize_and_pack(lsh.project(vis_feat))
            text_packed = binarize_and_pack(lsh.project(text_feat))
            concept = np.bitwise_xor(vis_packed, text_packed)

            semantic_memory.learn(concept, label=keyword)
            episodic_memory.learn(vis_packed, label=f"vision:{keyword}")

            if keyword not in log["concepts_learned"]:
                log["concepts_learned"][keyword] = {
                    "count": 0,
                    "first_seen": mid_time,
                    "emotions": [],
                }
            log["concepts_learned"][keyword]["count"] += 1
            log["concepts_learned"][keyword]["emotions"].append(
                nc.dominant_emotion)

        # Record timeline entry
        timeline_entry = {
            "time_s": mid_time,
            "subtitle": entry.text,
            "keywords": keywords,
            "dopamine": float(nc.dopamine),
            "cortisol": float(nc.cortisol),
            "serotonin": float(nc.serotonin),
            "oxytocin": float(nc.oxytocin),
            "arousal": float(nc.arousal),
            "emotion": nc.dominant_emotion,
            "alpha": float(alpha),
            "spike_rate": spike_rate,
            "route": route["primary_route"],
            "strategy": bg["strategy"],
            "color_warmth": color_stats["warmth"],
            "color_saturation": color_stats["saturation"],
            "color_brightness": color_stats["brightness"],
            "dominant_hue": color_stats["dominant_hue"],
            "maturity": float(bio_vision.maturity),
            "novel_concepts": novel_concepts,
            "concept_novelty": concept_novelty,
        }
        log["timeline"].append(timeline_entry)
        log["entries_processed"] += 1

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1:>4}/{len(entries)}] t={mid_time:>6.1f}s "
                  f"| {nc.dominant_emotion:<8} | a={alpha:.2f} "
                  f"| D={nc.dopamine:.2f} C={nc.cortisol:.2f} "
                  f"| {', '.join(keywords[:3])}")

    cap.release()
    elapsed = time.time() - t0

    log["elapsed_s"] = elapsed
    log["semantic_size"] = semantic_memory.size
    log["episodic_size"] = episodic_memory.size

    # Summary
    if verbose:
        print()
        print("=" * 60)
        print(f"  SRT Auto-Teacher Complete")
        print("=" * 60)
        print(f"  Video:     {video_path}")
        print(f"  Processed: {log['entries_processed']} subtitle windows")
        print(f"  Concepts:  {len(log['concepts_learned'])}")
        print(f"  Semantic:  {semantic_memory.size} memories")
        print(f"  Episodic:  {episodic_memory.size} memories")
        print(f"  Time:      {elapsed:.1f}s")
        print()

        # Top concepts by frequency
        sorted_concepts = sorted(
            log["concepts_learned"].items(),
            key=lambda x: -x[1]["count"])[:15]
        print("  Top concepts learned:")
        for concept, info in sorted_concepts:
            # Most common emotion for this concept
            emotions = info["emotions"]
            dominant = max(set(emotions), key=emotions.count) if emotions else "?"
            print(f"    {concept:<20} x{info['count']:<3} "
                  f"({dominant}, first at {info['first_seen']:.0f}s)")

        # Emotion distribution
        all_emotions = [t["emotion"] for t in log["timeline"]]
        if all_emotions:
            print()
            print("  Emotion distribution:")
            for emo in sorted(set(all_emotions)):
                count = all_emotions.count(emo)
                pct = count / len(all_emotions) * 100
                bar = "#" * int(pct / 2)
                print(f"    {emo:<10} {count:>4} ({pct:>5.1f}%) {bar}")

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(log, f, indent=2, default=str)
        print(f"\n  Log saved to {save_path}")

    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRT Auto-Teacher")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--srt", required=True, help="SRT subtitle file path")
    parser.add_argument("--save", default=None, help="Save log to JSON")
    parser.add_argument("--every", type=int, default=1,
                        help="Process every Nth subtitle")
    args = parser.parse_args()

    run_srt_teacher(args.video, args.srt, save_path=args.save,
                    sample_every_n=args.every)
