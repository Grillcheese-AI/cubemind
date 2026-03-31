"""
CubeMind v3 — Structured Specialist Architecture.
Uses WorldEncoder for Role-Binding and WorldManager for Specialist-based Memory.
"""

import numpy as np
from cubemind.model import HyperAxialAttention, PlasticCodebook
from cubemind.execution.world_encoder import WorldEncoder      # From input_file_10
from cubemind.execution.world_manager import WorldManager      # From input_file_11
from cubemind.execution.causal_codebook import CausalCodebook  # From input_file_5

def remove_non_ascii_from_file(input_filename, output_filename):
    """
    Reads a text file and writes a new file with non-ASCII characters removed.
    """
    try:
        with open(input_filename, 'r', encoding='utf-8') as infile, \
             open(output_filename, 'w', encoding='ascii', errors='ignore') as outfile:
            for line in infile:
                # The 'encode' converts to bytes, ignoring errors (removes non-ascii).
                # The 'decode' converts the safe bytes back to a string.
                cleaned_line = line.encode('ascii', errors='ignore').decode('ascii')
                outfile.write(cleaned_line)
        print(f"Non-ASCII characters removed successfully. Cleaned content written to {output_filename}")
    except FileNotFoundError:
        print(f"Error: The file {input_filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

class CubeMindExpert:
    def __init__(self, k=80, l=128):
        self.k, self.l = k, l
        
        # 1. Structured Encoder: Uses BLAKE2b hashing and role-binding
        self.encoder = WorldEncoder(k=k, l=l)
        
        # 2. Specialist Manager: Discovers and sharpens domain-specific rules
        self.world_manager = WorldManager(k=k, l=l, max_worlds=1024, tau=0.45)
        
        # 3. Causal Codebook: Maps dense embeddings to discrete VSA blocks
        self.causal_cb = CausalCodebook(k=k, l=l, n_explicit=64, n_learned=16)
        
        # 4. HyperAttention: The O(L) scan engine
        self.combiner = HyperAxialAttention(d_model=k*l, bucket_size=64)
        
        self.context_strings = []
        self.context_vectors = []

    def ingest_structured_document(self, text):
        """Uses WorldEncoder to build a structured context manifold."""
        # Split into logical chunks
        raw_chunks = text.split('\n\n')
        
        print(f"🧠 CubeMind is building a Structured World Model from {len(raw_chunks)} sections...")
        
        for chunk in raw_chunks:
            if len(chunk.strip()) < 200: continue
            
            # Use Narrative Encoding (Positional Binding) for temporal flow
            phi = self.encoder.encode_narrative(chunk)
            
            # Let the WorldManager decide: Is this a new domain or an existing one?
            # It uses Oja's rule internally to 'Consolidate' the specialist.
            transition = self.world_manager.process_transition(
                state_before=np.zeros((self.k, self.l)), # First step
                state_after=phi
            )
            
            self.context_strings.append(chunk)
            self.context_vectors.append(phi)
            
        print(f"✅ Created {self.world_manager.active_worlds} Specialist Models (Domains).")

    def forward(self, query_text):
        """Structured retrieval across the Specialist Manifold."""
        # Encode Query using the same structured narrative logic
        query_vec = self.encoder.encode_action(query_text).ravel()
        
        # Combine context using the O(L) engine
        history = np.stack([v.ravel() for v in self.context_vectors] + [query_vec])
        phi_integrated = self.combiner.forward(history, refine=True)[-1]

        
        
        # Retrieval: Find the chunks that align with the Integrated Result
        # Using your library's 'Structured Similarity'
        phi_2d = phi_integrated.reshape(self.k, self.l)
        sims = [np.mean(np.einsum('bl,bl->b', phi_2d, v)) for v in self.context_vectors]
        
        best_idx = np.argmax(sims)
        
        return {
            "answer_text": self.context_strings[best_idx],
            "confidence": sims[best_idx],
            "active_specialists": self.world_manager.active_worlds
        }
    
    def predict_consequences(self, current_rule_text):
        """Uses the CausalGraph to predict what happens next in the legal logic."""
        from cubemind.execution.causal_graph import CausalGraph # From input 6
        
        # 1. Encode the current rule
        phi = self.encoder.encode_action(current_rule_text)
        
        # 2. Find the best matching Specialist (Domain)
        specialists = self.world_manager.get_specialists()
        sims = self.bc.similarity_batch(phi, np.stack(specialists))
        best_domain_id = np.argmax(sims)
        
        # 3. Perform a 'Random Walk' in the Causal Graph of this domain
        # This simulates "Thinking forward" in time
        graph = CausalGraph()
        # (Assuming graph was populated during ingestion)
        path = graph.walk(start_id=f"chunk_{best_domain_id}", max_hops=3)
        
        return [self.context_strings[int(node_id.split('_')[1])] for node_id in path]

# ── NEW TEST SCRIPT ─────────────────────────────────────────────────────────

model = CubeMindExpert()
remove_non_ascii_from_file("./test_legal.txt", "./legal_clean.txt")
with open("legal_clean.txt") as f:
    hugedocument = f.read()
# 1. Ingest
model.ingest_structured_document(hugedocument)

# 2. Query
q = "naval merchant vessels war"
result = model.forward(q)

print(f"\n Query: {q}")
print(f" Best Specialist Match (Confidence: {result['confidence']:.4f}):")
print(f"--- \n{result['answer_text'][:500]} \n---")

if "merchant vessels" in result['answer_text'].lower():
    print("\n SUCCESS: Specialist Retrieval isolated the rule correctly.")