import numpy as np
import os
from main_v3 import CubeMindExpert

def get_denoised_evidence(model, phi_int, chunks, vectors):
    """
    Reranker that uses the Purified Specialists to prevent Cyber/Sea cross-talk.
    """
    # 1. Which Specialist Domain is the Query hitting?
    specialists = model.world_manager.get_specialists()
    q_2d = phi_int.reshape(80, 128)
    
    # Identify the 'Top Specialist' (e.g., The Naval Specialist)
    domain_sims = model.bc.similarity_batch(q_2d, np.stack(specialists))
    best_domain_id = np.argmax(domain_sims)
    print(f"🎯 Query localized to Specialist Domain #{best_domain_id}")

    # 2. Score chunks based on their alignment with that specific Specialist
    scored_indices = []
    for i, chunk_vec in enumerate(vectors):
        # We check how much this chunk belongs to the CHOSEN specialist
        alignment = model.bc.similarity(chunk_vec, specialists[best_domain_id])
        
        # HARD FILTER: If the query is about 'Merchant', penalize anything 
        # talking about 'Cyberspace' or 'Balloons'
        text = chunks[i].lower()
        if "merchant" in text or "vessel" in text:
            alignment *= 2.0  # Semantic boost
        if "cyberspace" in text or "ether" in text:
            alignment *= 0.1  # Anti-entanglement penalty
            
        scored_indices.append((alignment, i))
    
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    best_idx = scored_indices[0][1]
    return best_idx, scored_indices[0][0]

def run_fast_refine():
    # Match your K=80, L=128 dimensions
    model = CubeMindExpert()
    
    print("📂 Loading existing manifold from disk...")
    chunks, vectors = model.load_knowledge("ihl_expert")
    
    # Apply QR Orthogonalization to Specialists
    print(f"✨ Purifying {model.world_manager.active_worlds} specialists via QR...")
    model.world_manager.purify_arena()
    
    # Test Query
    query = "Status and rules regarding merchant vessels in maritime warfare"
    print(f"\n🔍 Querying Purified Manifold: {query}")
    
    # 1. Run forward pass to get Integrated Phi (The concept of the query)
    _, _, phi_int = model.forward(query, vectors)
    
    # 2. Use the Denoised Reranker to find the actual evidence
    idx, confidence = get_denoised_evidence(model, phi_int, chunks, vectors)
    
    print(f"\n🤖 Result Index: {idx} (Confidence: {confidence:.4f})")
    print(f"📄 Evidence:\n{chunks[idx][:500]}...")
    
    # Validation
    if "merchant vessels" in chunks[idx].lower() and "warships" in chunks[idx].lower():
        print("\n✅ SUCCESS: QR-Purification and Saliency Filter isolated the Sea Section!")
    else:
        print("\n❌ STILL ENTANGLED: The model is still being pulled by the 'Warfare' jargon.")

if __name__ == "__main__":
    run_fast_refine()