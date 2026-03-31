"""CubeMind API Server — serves web demo + WebSocket bridge to real Mind.

Single server that:
  1. Serves the HTML dashboard at http://localhost:8765/
  2. Exposes CubeMind's Mind class via WebSocket at ws://localhost:8765/ws
  3. Serves static files (videos, images) from data/

All data goes through the real VSA pipeline — no simulation.

Usage:
    python scripts/api_server.py
    python scripts/api_server.py --port 8765

Then open http://localhost:8765 in your browser.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime

import numpy as np

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from pathlib import Path

# CubeMind
from cubemind.ops import BlockCodes
from cubemind.perception.snn import SNNEncoder, NeurochemicalState
from cubemind.brain.cortex import Thalamus, BasalGanglia, CircadianCells
from cubemind.ops.vsa_bridge import (
    ContinuousItemMemory, LSHProjector, binarize_and_pack,
)
from cubemind.experimental.affective_graph import affective_alpha
from cubemind.perception.experiential import ExperientialEncoder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cubemind.api")

if not HAS_FASTAPI:
    print("Install FastAPI: pip install fastapi uvicorn websockets")
    raise SystemExit(1)

app = FastAPI(title="CubeMind Cognitive API")


# ── CubeMind Backend ─────────────────────────────────────────────────

class CubeMindBackend:
    """Real CubeMind pipeline exposed via API."""

    def __init__(self):
        self.d_model = 256
        self.d_vsa = 2048
        self.bc = BlockCodes(k=8, l=64)

        self.snn = SNNEncoder(
            d_input=104, n_neurons=128, d_vsa=self.d_vsa,
            neuron_type="lif", tau=15.0, v_threshold=0.08, seed=42)
        self.snn.stdp_lr_potentiate = 0.0002

        self.thalamus = Thalamus(embedding_dim=self.d_model)
        self.basal_ganglia = BasalGanglia()
        self.circadian = CircadianCells()
        self.exp_encoder = ExperientialEncoder(k=8, l=64, seed=42)
        self.lsh = LSHProjector(
            d_input=self.d_model, d_output=self.d_vsa, seed=52)

        self.semantic_memory = ContinuousItemMemory(
            d_vsa=self.d_vsa, max_capacity=10000)
        self.episodic_memory = ContinuousItemMemory(
            d_vsa=self.d_vsa, max_capacity=50000)

        self.concepts: dict[str, dict] = {}
        self.teach_count = 0
        log.info("CubeMind backend initialized (d=%d, vsa=%d)",
                 self.d_model, self.d_vsa)

    def get_state(self) -> dict:
        """Get current neurochemistry + memory state."""
        nc = self.snn.neurochemistry
        alpha = affective_alpha(nc)
        circ = self.circadian.get_context()

        now = datetime.now()
        seasons = ['winter'] * 3 + ['spring'] * 3 + ['summer'] * 3 + ['fall'] * 3
        season = seasons[now.month - 1]

        return {
            "dopamine": float(nc.dopamine),
            "cortisol": float(nc.cortisol),
            "serotonin": float(nc.serotonin),
            "oxytocin": float(nc.oxytocin),
            "arousal": float(nc.arousal),
            "emotion": nc.dominant_emotion,
            "alpha": float(alpha),
            "mode": "EXPLORE" if alpha < 0.45 else "CONSOLIDATE" if alpha > 0.55 else "BALANCED",
            "circadian": circ.get("phase", "unknown"),
            "season": season,
            "hour": now.hour,
            "concepts": len(self.concepts),
            "semantic_size": self.semantic_memory.size,
            "episodic_size": self.episodic_memory.size,
            "teach_count": self.teach_count,
        }

    def perceive(self, features: list[float] | None = None) -> dict:
        """Run SNN perception on input features."""
        if features is None:
            features = np.random.default_rng().standard_normal(104).astype(
                np.float32) * 0.1
        else:
            feat = np.array(features, dtype=np.float32)
            if len(feat) < 104:
                feat = np.pad(feat, (0, 104 - len(feat)))
            features = feat[:104]

        snn_result = self.snn.encode_temporal(features)
        nc = self.snn.neurochemistry

        embed = np.zeros(self.d_model, dtype=np.float32)
        embed[:min(104, self.d_model)] = features[:min(104, self.d_model)]
        route = self.thalamus.route(
            embed, arousal=nc.arousal,
            valence=getattr(nc, 'valence', 0.0))

        state = self.get_state()
        state["spike_rate"] = float(np.mean(snn_result["spikes"]))
        state["spikes"] = snn_result["spikes"].tolist()
        state["route"] = route
        return state

    def teach(self, label: str, features: list[float] | None = None) -> dict:
        """Teach a concept: bind features + label → memory."""
        self.teach_count += 1
        nc = self.snn.neurochemistry

        # Feature vector
        if features is not None:
            feat = np.array(features, dtype=np.float32)[:self.lsh.d_input]
        else:
            feat = np.random.default_rng(
                hash(label) % (2**31)).standard_normal(
                self.lsh.d_input).astype(np.float32) * 0.1
        if len(feat) < self.lsh.d_input:
            feat = np.pad(feat, (0, self.lsh.d_input - len(feat)))

        # Text vector
        rng = np.random.default_rng(hash(label) % (2**31))
        text_feat = rng.standard_normal(self.lsh.d_input).astype(
            np.float32) * 0.1

        # XOR binding
        vis_packed = binarize_and_pack(self.lsh.project(feat))
        text_packed = binarize_and_pack(self.lsh.project(text_feat))
        concept = np.bitwise_xor(vis_packed, text_packed)

        self.semantic_memory.learn(concept, label=label)
        self.episodic_memory.learn(vis_packed, label=f"vision:{label}")

        self.concepts[label] = {
            "count": self.concepts.get(label, {}).get("count", 0) + 1,
            "emotion": nc.dominant_emotion,
            "time": time.time(),
        }

        # Dopamine spike on learning
        nc.update(novelty=0.8, threat=0.0, focus=0.5, valence=0.3)

        return {
            "label": label,
            "emotion": nc.dominant_emotion,
            "concepts": len(self.concepts),
            "semantic_size": self.semantic_memory.size,
            "episodic_size": self.episodic_memory.size,
        }

    def recall(self, features: list[float] | None = None,
               k: int = 3) -> list[dict]:
        """Recall: search memory by feature similarity."""
        if features is not None:
            feat = np.array(features, dtype=np.float32)[:self.lsh.d_input]
        else:
            feat = np.random.default_rng().standard_normal(
                self.lsh.d_input).astype(np.float32) * 0.1
        if len(feat) < self.lsh.d_input:
            feat = np.pad(feat, (0, self.lsh.d_input - len(feat)))

        packed = binarize_and_pack(self.lsh.project(feat))
        results = []

        for mem, mem_type in [(self.semantic_memory, "semantic"),
                               (self.episodic_memory, "episodic")]:
            if mem.size > 0:
                hits = mem.retrieve(packed, k=k)
                for idx, sim, label in hits:
                    results.append({
                        "label": label.replace("vision:", ""),
                        "similarity": float(sim),
                        "memory": mem_type,
                    })

        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x["similarity"], reverse=True):
            if r["label"] not in seen:
                seen.add(r["label"])
                unique.append(r)
        return unique[:k]


# ── Singleton ────────────────────────────────────────────────────────

mind = CubeMindBackend()


# ── WebSocket endpoint ───────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")

    # Send initial state
    await ws.send_json({"type": "state", "data": mind.get_state()})

    try:
        while True:
            msg = await ws.receive_json()
            cmd = msg.get("cmd", "")

            if cmd == "perceive":
                result = mind.perceive(msg.get("features"))
                await ws.send_json({"type": "state", "data": result})

            elif cmd == "teach":
                label = msg.get("label", "")
                if label:
                    result = mind.teach(label, msg.get("features"))
                    await ws.send_json({"type": "teach", "data": result})
                    # Also send updated state
                    await ws.send_json({
                        "type": "state", "data": mind.get_state()})

            elif cmd == "recall":
                results = mind.recall(msg.get("features"), k=3)
                await ws.send_json({"type": "recall", "data": results})

            elif cmd == "state":
                await ws.send_json({
                    "type": "state", "data": mind.get_state()})

            elif cmd == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        log.info("Client disconnected")


@app.get("/")
def root():
    """Serve the web demo HTML."""
    html_path = Path(__file__).resolve().parent.parent / "docs" / "demo" / "cubemind_live.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>CubeMind API</h1><p>Demo HTML not found at docs/demo/cubemind_live.html</p>")


@app.get("/api/status")
def api_status():
    """JSON status endpoint."""
    return {"status": "running", "concepts": len(mind.concepts),
            "semantic": mind.semantic_memory.size,
            "episodic": mind.episodic_memory.size}


@app.get("/data/{filename:path}")
def serve_data(filename: str):
    """Serve files from data/ directory (videos, images)."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    file_path = data_dir / filename
    if file_path.exists() and file_path.is_file():
        # Determine media type
        suffix = file_path.suffix.lower()
        media_types = {
            ".mp4": "video/mp4", ".webm": "video/webm",
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".json": "application/json", ".npz": "application/octet-stream",
        }
        media_type = media_types.get(suffix, "application/octet-stream")
        return FileResponse(str(file_path), media_type=media_type)
    return HTMLResponse("<h1>404</h1><p>File not found</p>", status_code=404)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CubeMind API Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print("=" * 50)
    print("  CubeMind Cognitive API Server")
    print("=" * 50)
    print(f"  Dashboard:  http://localhost:{args.port}/")
    print(f"  WebSocket:  ws://localhost:{args.port}/ws")
    print(f"  API Status: http://localhost:{args.port}/api/status")
    print(f"  Videos:     http://localhost:{args.port}/data/bear.mp4")
    print("=" * 50)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
