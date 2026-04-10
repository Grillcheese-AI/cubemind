import asyncio
import json
import os
import re

import numpy as np

from cubemind.execution.attribute_extractor import extract_batch
from cubemind.execution.causal_codebook import CausalCodebook
from cubemind.execution.world_manager import WorldManager
from cubemind.model import HyperAxialAttention
from cubemind.ops.block_codes import BlockCodes
from cubemind.perception.semantic_encoder import SemanticEncoder

# ── CONFIGURATION ───────────────────────────────────────────────────────────
K, L = 80, 128      # High-resolution VSA (10,240 dimensions)
N_EXPLICIT = 32     # Number of LLM-extracted attributes
CHUNK_SIZE = 400    # Larger chunks for better LLM reasoning context

GGUF_PATH = "cubemind/gguf/bge-m3-q8_0.gguf"

# ── ASYNC INGESTION ENGINE ──────────────────────────────────────────────────

async def ingest_high_fidelity(model, text):
    """
    The 'Causal Oracle' Ingestor.
    Combines LLM Attribute Extraction with VSA Specialist Discovery.
    """
    # 1. Cleaning
    text = re.sub(r"From: Oxford.*?Reserved\.", "", text, flags=re.DOTALL)
    text = re.sub(r"Subscriber:.*?date:.*?\n", "", text)
    clean_text = text.strip()

    # 2. Chunking
    tokens = clean_text.split()
    raw_chunks = [" ".join(tokens[i:i+CHUNK_SIZE]) for i in range(0, len(tokens), CHUNK_SIZE)]

    print(f"Step 1: Extracting 32 Attributes via OpenRouter for {len(raw_chunks)} chunks...")
    event_dicts = [{"text": c, "category": "international_humanitarian_law"} for c in raw_chunks]

    attribute_list = await extract_batch(
        event_dicts, model="x-ai/grok-4.20-multi-agent-beta", batch_size=10,
    )

    # 3. Hybrid Encoding: SDLS-purified semantic vectors + LLM attributes
    print("Step 2: Building Specialist Manifold (Semantic + Oja-Plastic)...")
    causal_cb = CausalCodebook(k=K, l=L, n_explicit=N_EXPLICIT)

    chunks, vectors = [], []
    for i, chunk_text in enumerate(raw_chunks):
        attrs = attribute_list[i]
        if not attrs:
            continue

        # Semantic VSA encoding (replaces hash-based WorldEncoder)
        learned_vsa = model.encoder.encode_action(chunk_text)

        # Hybrid: Blocks 0-31 = LLM attributes, Blocks 32-79 = Semantic VSA
        hybrid_phi = causal_cb.encode(attributes=attrs, embedding=learned_vsa)

        chunks.append(chunk_text)
        vectors.append(hybrid_phi)

        # Specialist Discovery
        model.world_manager.process_transition(
            state_before=np.zeros((K, L)),
            state_after=hybrid_phi,
        )

    return chunks, vectors, attribute_list


# ── SEMANTIC INGESTION (no LLM, SDLS purified) ────────────────────────────

def ingest_semantic(model, text):
    """Pure semantic ingestion with SDLS purification. No LLM calls."""
    text = re.sub(r"From: Oxford.*?Reserved\.", "", text, flags=re.DOTALL)
    text = re.sub(r"Subscriber:.*?date:.*?\n", "", text)
    clean_text = text.strip()

    tokens = clean_text.split()
    raw_chunks = [" ".join(tokens[i:i+CHUNK_SIZE]) for i in range(0, len(tokens), CHUNK_SIZE)]

    print(f"Encoding {len(raw_chunks)} chunks with SDLS purification...")

    # SDLS corpus encoding: removes document-level noise floor
    vectors, noise_axis = model.encoder.encode_corpus(raw_chunks)

    # Feed each vector to WorldManager for specialist discovery
    for v in vectors:
        model.world_manager.process_transition(
            state_before=np.zeros((K, L)),
            state_after=v,
        )

    return raw_chunks, vectors, noise_axis


# ── EXTENDED EXPERT MODEL ───────────────────────────────────────────────────

class CubeMindExpert:
    def __init__(self, k=K, l=L, gguf_path=None):
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)

        # SemanticEncoder: BGE-M3 GGUF > sentence-transformers > hash fallback
        self.encoder = SemanticEncoder(k=k, l=l, gguf_path=gguf_path)
        print(f"Encoder mode: {self.encoder._mode} (embed_dim={self.encoder._embed_dim})")

        self.world_manager = WorldManager(k=k, l=l, max_worlds=64, tau=0.45)

        # O(L) Attention Engine (low-rank projections, ~30MB at d_vsa=10240)
        self.combiner = HyperAxialAttention(d_model=k * l, bucket_size=32)

        # Stored after SDLS corpus encoding for query purification
        self._noise_axis = None

    def forward(self, query_text, context_vectors):
        """SDLS-purified retrieval with HyperAxialAttention."""
        # 1. Encode query with same SDLS purification as corpus
        if self._noise_axis is not None:
            query_vec = self.encoder.encode_query_purified(
                query_text, self._noise_axis,
            ).ravel()
        else:
            query_vec = self.encoder.encode_action(query_text).ravel()

        # 2. Linear-Time Attention O(L) — SDLS noise removal is built into forward()
        history = np.stack([v.ravel() for v in context_vectors] + [query_vec])
        phi_integrated = self.combiner.forward(history, refine=True)[-1]

        # 3. Cosine similarity ranking (vectors already SDLS-purified at ingest)
        q_2d = phi_integrated.reshape(self.k, self.l)
        sims = []
        for v in context_vectors:
            c_2d = v.reshape(self.k, self.l) if v.ndim == 2 else v.ravel().reshape(self.k, self.l)
            sims.append(float(self.bc.similarity(q_2d, c_2d)))

        sims = np.array(sims)
        best_idx = int(np.argmax(sims))
        return best_idx, float(sims[best_idx]), phi_integrated

    def save_knowledge(self, chunks, vectors, noise_axis=None, path_prefix="ihl_expert"):
        np.save(f"{path_prefix}_arena.npy", self.world_manager._arena)
        np.save(f"{path_prefix}_count.npy", np.array([self.world_manager.active_worlds]))
        np.save(f"{path_prefix}_vectors.npy", np.stack(vectors))
        if noise_axis is not None:
            np.save(f"{path_prefix}_noise_axis.npy", noise_axis)
        with open(f"{path_prefix}_chunks.json", "w") as f:
            json.dump(chunks, f)
        print(f"Knowledge base '{path_prefix}' saved to disk.")

    def load_knowledge(self, path_prefix="ihl_expert"):
        self.world_manager._arena = np.load(f"{path_prefix}_arena.npy")
        self.world_manager.active_worlds = int(np.load(f"{path_prefix}_count.npy")[0])
        vectors_stack = np.load(f"{path_prefix}_vectors.npy")
        context_vectors = [v for v in vectors_stack]

        # Load SDLS noise axis if available
        noise_path = f"{path_prefix}_noise_axis.npy"
        if os.path.exists(noise_path):
            self._noise_axis = np.load(noise_path)

        with open(f"{path_prefix}_chunks.json", "r") as f:
            chunks = json.load(f)

        print(f"Knowledge base loaded. {self.world_manager.active_worlds} specialists, "
              f"SDLS={'yes' if self._noise_axis is not None else 'no'}.")
        return chunks, context_vectors


# ── EXECUTION ───────────────────────────────────────────────────────────────

async def main():
    print("Initializing CubeMind v3 (Semantic + SDLS)...")

    # Use GGUF if available, otherwise falls back to sbert or hash
    gguf = GGUF_PATH if os.path.exists(GGUF_PATH) else None
    model = CubeMindExpert(gguf_path=gguf)

    # Check for cached knowledge
    if os.path.exists("ihl_purified_arena.npy"):
        print("Fast reload from disk...")
        chunks, vectors = model.load_knowledge("ihl_purified")
    else:
        with open("cubemind/legal_clean.txt") as f:
            hugedocument = f.read()

        # Pure semantic path (no LLM needed)
        chunks, vectors, noise_axis = ingest_semantic(model, hugedocument)
        model._noise_axis = noise_axis

        # Optional: purify world arena
        model.world_manager.purify_arena()

        model.save_knowledge(chunks, vectors, noise_axis, "ihl_purified")

    print(f"Ready: {model.world_manager.active_worlds} specialist domains, "
          f"{len(chunks)} chunks indexed.")

    # Test query
    query = "Status and rules regarding merchant vessels in maritime warfare"
    idx, confidence, phi_int = model.forward(query, vectors)

    print(f"\nQuery: {query}")
    print(f"Result: chunk {idx} (similarity: {confidence:.4f})")
    print(f"Evidence: {chunks[idx][:500]}...")

if __name__ == "__main__":
    asyncio.run(main())
