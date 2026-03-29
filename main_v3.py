import asyncio
import os
import numpy as np
import re
from cubemind.model import CubeMind
from cubemind.execution.world_encoder import WorldEncoder
from cubemind.execution.world_manager import WorldManager
from cubemind.execution.causal_codebook import CausalCodebook
from cubemind.execution.attribute_extractor import extract_batch
from cubemind.execution import document_ingestor

# ── CONFIGURATION ───────────────────────────────────────────────────────────
K, L = 80, 128      # High-resolution VSA (10,240 dimensions)
N_EXPLICIT = 32     # Number of LLM-extracted attributes
CHUNK_SIZE = 400    # Larger chunks for better LLM reasoning context

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
    
    print(f"🤖 Step 1: Extracting 32 Attributes via OpenRouter for {len(raw_chunks)} chunks...")
    event_dicts = [{"text": c, "category": "international_humanitarian_law"} for c in raw_chunks]
    
    # Call OpenRouter (Batch Async)
    # This will populate 'Explicit' blocks with high-level logic (pivotality, urgency, etc.)
    attribute_list = await extract_batch(event_dicts, model="x-ai/grok-4.20-multi-agent-beta", batch_size=10)

    # 3. Hybrid Encoding
    print(f"🧠 Step 2: Building Specialist Manifold (Oja-Plastic)...")
    causal_cb = CausalCodebook(k=K, l=L, n_explicit=N_EXPLICIT)
    
    chunks, vectors = [], []
    for i, chunk_text in enumerate(raw_chunks):
        attrs = attribute_list[i]
        if not attrs: continue # Skip failed API calls
        
        # learned_part: VSA Hashing of the raw text
        learned_vsa = model.encoder.encode_action(chunk_text)
        
        # hybrid_phi: Blocks 0-31 = LLM attributes, Blocks 32-79 = Word hash
        hybrid_phi = causal_cb.encode(attributes=attrs, embedding=learned_vsa)
        
        # Save results
        chunks.append(chunk_text)
        vectors.append(hybrid_phi)
        
        # Specialist Discovery: WorldManager spawns a new domain or consolidates
        model.world_manager.process_transition(
            state_before=np.zeros((K, L)), 
            state_after=hybrid_phi
        )

    return chunks, vectors, attribute_list

# ── EXTENDED EXPERT MODEL ───────────────────────────────────────────────────

class CubeMindExpert:
    def __init__(self, k=K, l=L):
        # High-res Library Components
        self.encoder = WorldEncoder(k=k, l=l)
        self.world_manager = WorldManager(k=k, l=l, max_worlds=64, tau=0.45)
        self.bc = self.encoder.bc
        # $O(L)$ Attention Engine
        from cubemind.model import HyperAxialAttention
        self.combiner = HyperAxialAttention(d_model=K*L, bucket_size=32)

    def forward(self, query_text, context_vectors):
        """Structured Search with Spectral Saliency (SDLS logic)."""
        # 1. Encode Query
        self.k = K
        self.l = L
        query_vec = self.encoder.encode_action(query_text).ravel()
        
        # 2. Linear-Time Attention (O(L))
        history = np.stack([v.ravel() for v in context_vectors] + [query_vec])
        phi_integrated = self.combiner.forward(history, refine=True)[-1]
        
        # ─── THE SPECTRAL FIX ───
        # 3. Calculate the 'Document Noise Floor' (Background Jargon)
        # We compute the average of ALL chunks in the document.
        manifold_stack = np.stack([v.ravel() for v in context_vectors])
        global_mean = np.mean(manifold_stack, axis=0)
        
        # 4. SALIENCY FILTER: Subtract 99% of the background noise
        # Signal = Signal - (Noise * 0.99)
        # This deletes the words "Warfare", "IHL", "Domain" from the vectors.
        q_signal = phi_integrated - (global_mean * 0.99)
        q_2d = q_signal.reshape(K, L)
        
        sims = []
        for v in context_vectors:
            # Subtract noise from each chunk as well
            c_signal = v.ravel() - (global_mean * 0.99)
            c_2d = c_signal.reshape(self.k, self.l)
            
            # 5. POWER-4 SQUEEZING: Non-linear similarity amplification
            raw_sim = self.bc.similarity(q_2d, c_2d)
            # This makes a 0.1 match look like 0.0001 and a 0.8 match look like 0.40
            # It 'blackens' the background so the result stands out.
            sims.append(np.sign(raw_sim) * (np.abs(raw_sim) ** 4))
            
        best_idx = np.argmax(sims)
        return best_idx, sims[best_idx], phi_integrated
    
    def save_knowledge(self, chunks, vectors, path_prefix="ihl_expert"):
        # Save the Specialists (The Arena)
        np.save(f"{path_prefix}_arena.npy", self.world_manager._arena)
        # Save the Active Worlds Count
        np.save(f"{path_prefix}_count.npy", np.array([self.world_manager.active_worlds]))
        # Save the actual Document Context (Vectors and Text)
        np.save(f"{path_prefix}_vectors.npy", np.stack(vectors))
        import json
        with open(f"{path_prefix}_chunks.json", "w") as f:
            json.dump(chunks, f)
        print(f"💾 Knowledge base '{path_prefix}' saved to disk.")

    def load_knowledge(self, path_prefix="ihl_expert"):
        # Load Specialists
        self.world_manager._arena = np.load(f"{path_prefix}_arena.npy")
        self.world_manager.active_worlds = int(np.load(f"{path_prefix}_count.npy")[0])
        # Load Document Context
        vectors_stack = np.load(f"{path_prefix}_vectors.npy")
        context_vectors = [v for v in vectors_stack]
        
        import json
        with open(f"{path_prefix}_chunks.json", "r") as f:
            chunks = json.load(f)
            
        print(f"📂 Knowledge base loaded. {self.world_manager.active_worlds} specialists ready.")
        return chunks, context_vectors

# ── EXECUTION ───────────────────────────────────────────────────────────────

async def main():
    print("🚀 Initializing CubeMind v3 'Causal Oracle'...")
    model = CubeMindExpert()

    # 1. Ingest document (Calls LLM + VSA)
    # Check if we have saved knowledge
    if os.path.exists("ihl_purified_arena.npy"):
        print("⚡ Performing Fast Reload from disk...")
        chunks, vectors = model.load_knowledge("ihl_purified")
    else:

        with open("cubemind/legal_clean.txt") as f:
            hugedocument = f.read()
        chunks, vectors, raw_attrs = await ingest_high_fidelity(model, hugedocument)
        model.save_knowledge(chunks, vectors)
    
    print(f"✅ Created {model.world_manager.active_worlds} Specialist Domains.")

    query = "Status and rules regarding merchant vessels in maritime warfare"
    idx, confidence, phi_int = model.forward(query, vectors)
    
    print(f"\n🤖 Result Index: {idx} (Confidence: {confidence:.4f})")
    print(f"📄 Evidence: {chunks[idx][:500]}...")

if __name__ == "__main__":
    asyncio.run(main())