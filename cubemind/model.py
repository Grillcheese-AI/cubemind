"""
CubeMind v2.1 — Oja-Plastic NVSA Architecture.
Fixed: Reshape ValueError for contexts smaller than bucket_size.
"""

from __future__ import annotations
import numpy as np
import math
import re
from sklearn.decomposition import PCA

from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.execution.cvl import ContrastiveValueEstimator
from cubemind.execution.decoder import Decoder
from cubemind.execution.hyla import HYLA
from cubemind.memory.cache import VSACache
from cubemind.memory.hippocampal import HippocampalMemory
from cubemind.ops.block_codes import BlockCodes
from cubemind.perception.encoder import Encoder
from cubemind.reasoning.hmm_rule import HMMEnsemble
from cubemind.telemetry import metrics

# ── Oja Plasticity Kernels ──────────────────────────────────────────────────

def oja_update(m: np.ndarray, x: np.ndarray, eta: float = 0.01) -> np.ndarray:
    m_flat = m.ravel().astype(np.float32)
    x_flat = x.ravel().astype(np.float32)
    y = float(np.dot(m_flat, x_flat))
    m_flat = m_flat + (eta * y) * (x_flat - y * m_flat)
    return m_flat.reshape(m.shape)

def oja_update_batch(memories: np.ndarray, inputs: np.ndarray, eta: float = 0.01) -> np.ndarray:
    y = np.sum(memories * inputs, axis=-1, keepdims=True)
    updated = memories + (eta * y) * (inputs - y * memories)
    return updated.astype(np.float32)

# ── O(L) Attention Engine ──────────────────────────────────────────────────

class HyperAxialAttention:
    def __init__(self, d_model, heads=4, bucket_size=32, sample_size=16):
        self.d_model = d_model
        self.n_h = heads
        self.d_h = d_model // heads
        self.bucket_size = bucket_size # Reduced for testing/small docs
        self.sample_size = sample_size
        self.rng = np.random.default_rng(42)
        self.projections = self.rng.standard_normal((self.d_h, 32)).astype(np.float32)
        
        std = 1.0 / math.sqrt(d_model)
        self.W_Q = self.rng.normal(0, std, (d_model, d_model)).astype(np.float32)
        self.W_K = self.rng.normal(0, std, (d_model, d_model)).astype(np.float32)
        self.W_V = self.rng.normal(0, std, (d_model, d_model)).astype(np.float32)

    def _hash(self, x):
        return np.sum((x @ self.projections > 0) * (2 ** np.arange(32)), axis=-1)

    def forward(self, X, causal=True, refine=True):
        L, d = X.shape
        
        # 1. ORTHOGONAL SALIENCY (Aggressive Noise Removal)
        global_mean = np.mean(X, axis=0, keepdims=True)
        # We subtract 99% of the 'Book average' to leave only the unique spikes
        X_saliency = X - (global_mean * 0.99) 

        # Early exit for small docs
        if L <= self.bucket_size:
            Q, K, V = X_saliency @ self.W_Q, X_saliency @ self.W_K, X @ self.W_V
            # TEMPERATURE GATING
            scores = (Q @ K.T) / (math.sqrt(self.d_h) * 0.001)
            scores -= np.mean(scores, axis=-1, keepdims=True) # Center scores
            m = np.max(scores, axis=-1, keepdims=True)
            ex = np.exp(scores - m)
            return (ex / (ex.sum(axis=-1, keepdims=True) + 1e-6)) @ V

        # Project and Multi-head split
        Q_all = (X_saliency @ self.W_Q).reshape(L, self.n_h, self.d_h).transpose(1, 0, 2)
        K_all = (X_saliency @ self.W_K).reshape(L, self.n_h, self.d_h).transpose(1, 0, 2)
        V_all = (X @ self.W_V).reshape(L, self.n_h, self.d_h).transpose(1, 0, 2)

        out_heads = []
        orig_indices = np.arange(L)

        for h in range(self.n_h):
            q_idx, k_idx = np.argsort(self._hash(Q_all[h])), np.argsort(self._hash(K_all[h]))
            qs, ks, vs = Q_all[h][q_idx], K_all[h][k_idx], V_all[h][k_idx]
            qp, kp = orig_indices[q_idx], orig_indices[k_idx]

            num_b = int(math.ceil(L / self.bucket_size))
            L_padded = num_b * self.bucket_size
            
            # PADDING
            pad_len = L_padded - L
            if pad_len > 0:
                qs = np.pad(qs, ((0, pad_len), (0, 0)))
                ks = np.pad(ks, ((0, pad_len), (0, 0)))
                vs = np.pad(vs, ((0, pad_len), (0, 0)))
                qp = np.pad(qp, (0, pad_len), constant_values=-1)
                kp = np.pad(kp, (0, pad_len), constant_values=L+1)

            qb = qs.reshape(num_b, self.bucket_size, -1)
            kb = ks.reshape(num_b, self.bucket_size, -1)
            vb = vs.reshape(num_b, self.bucket_size, -1)
            
            # --- SUPER-CENTROID REFINEMENT ---
            if refine:
                # Calculate P as the Principal Direction of the bucket
                p = np.mean(qb, axis=1)
                for _ in range(5): 
                    p = oja_update_batch(p, np.mean(kb, axis=1), eta=0.4)
                
                # Saliency Projection: Keep only the component of V that matches the Bucket theme
                alignment = np.sum(vb * p[:, np.newaxis, :], axis=-1, keepdims=True)
                # Filter: If alignment is weak, suppress the token entirely
                vb = np.where(alignment > np.mean(alignment, axis=1, keepdims=True), vb, 0.0)

            # --- POWER-6 SQUEEZING ---
            # Raise similarity to the 6th power. This turns a 0.2 match into 0.00006 
            # and a 0.8 match into 0.26. Noise literally disappears.
            # scores_b = (qb @ kb.transpose(0, 2, 1)) / (math.sqrt(self.d_h) * 0.001)
            # scores_b -= np.mean(scores_b, axis=-1, keepdims=True) # Contrast Stretch
            # scores_b = np.sign(scores_b) * (np.abs(scores_b) ** 6) 
            threshold = np.percentile(scores_b, 98)
            scores_b = np.where(scores_b > threshold, scores_b, -1e9)
            
            # Use an even sharper temperature
            scores_b /= 0.001

            if causal:
                mask = qp.reshape(num_b, -1, 1) >= kp.reshape(num_b, 1, -1)
                scores_b = np.where(mask, scores_b, -1e9)

            m_b = np.max(scores_b, axis=-1, keepdims=True)
            exp_b = np.exp(scores_b - m_b)
            out_b = (exp_b / (exp_b.sum(axis=-1, keepdims=True) + 1e-6)) @ vb

            # Unsort and append
            out_local_flat = out_b.reshape(L_padded, -1)[:L]
            out_heads.append(out_local_flat[np.argsort(q_idx)])

        return np.concatenate(out_heads, axis=-1)

# ── Main CubeMind Orchestrator ──────────────────────────────────────────────

class PlasticCodebook:
    def __init__(self, bc, n_entries=16, eta=0.005, seed=42):
        self.bc = bc
        self.eta = eta
        self.entries = bc.codebook_discrete(n_entries, seed=seed)
        self.access_count = np.zeros(n_entries, dtype=np.int64)

    def adapt_nearest(self, observation):
        sims = self.bc.similarity_batch(observation, self.entries)
        idx = int(np.argmax(sims))
        self.entries[idx] = oja_update(self.entries[idx], observation.ravel(), self.eta).reshape(observation.shape)
        self.access_count[idx] += 1
        return idx, float(sims[idx])

class CubeMind:
    def __init__(self, k=K_BLOCKS, l=L_BLOCK, n_codebook=128, oja_eta=0.01):
        self.k, self.l, self.d_vsa = k, l, k*l
        self.oja_eta = oja_eta
        self.bc = BlockCodes(k, l)
        self.encoder = Encoder(k=k, l=l)
        self.plastic_codebook = PlasticCodebook(self.bc, n_entries=n_codebook)
        self.codebook = self.plastic_codebook.entries
        self.hmm = HMMEnsemble(self.codebook)
        self.decoder = Decoder(self.codebook)
        self.hyla = HYLA(d_vsa=self.d_vsa, d_hidden=128, d_out=self.d_vsa, k=k, l=l)
        self.cvl = ContrastiveValueEstimator(d_state=self.d_vsa, d_action=self.d_vsa)
        self.cache = VSACache(max_size=10000, d_vsa=self.d_vsa)
        self.hippocampal = HippocampalMemory(d_model=self.d_vsa)
        # Init with small bucket_size so the 59 segments document actually triggers the LSH path
        self.combiner = HyperAxialAttention(d_model=self.d_vsa, bucket_size=32)

    def forward(self, text=None, phi=None, context=None):
        if phi is None:
            phi = self.encoder.encode(text) if text else self.bc.random_discrete()
        phi_flat = phi.ravel().astype(np.float32)

        surprise = self.cache.surprise(phi_flat)
        if self.cache.size > 0:
            sims, _, idxs = self.cache.lookup(phi_flat, k=1)
            if sims[0,0] > 0.8:
                i = int(idxs[0,0])
                self.cache.keys[i] = np.sign(oja_update(self.cache.keys[i].astype(np.float32), phi_flat, self.oja_eta)).astype(np.int8)

        self.cache.add(phi_flat, np.array([surprise, 0.1]))

        if context:
            history = np.stack([c.ravel() for c in context] + [phi_flat])
            phi_integrated = self.combiner.forward(history)[-1]
        else:
            phi_integrated = phi_flat

        hyla_out = self.hyla.forward(phi_integrated, phi_integrated)
        output_bc = self.bc.discretize(self.bc.from_flat(hyla_out, self.k))
        answer, confidence, _ = self.decoder.decode(output_bc)
        self.hippocampal.store(phi_flat, content_tag=text or "")

        return {"answer": answer, "confidence": confidence, "phi_integrated": phi_integrated}

    def visualize_manifold(self, context_vectors):
        if len(context_vectors) < 2: return
        data = np.stack([c.ravel() for c in context_vectors])
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        coords = PCA(n_components=2).fit_transform(data)
        plt.figure(figsize=(10, 6))
        plt.scatter(coords[:,0], coords[:,1], c=range(len(coords)), cmap='winter', edgecolors='k')
        plt.title("CubeMind Plastic Manifold")
        plt.show()