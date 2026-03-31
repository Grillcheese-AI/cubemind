"""MoQE checkpoint eval on GPU via PyTorch.

Loads numpy checkpoint → torch tensors on CUDA → fast eval + generation.
The 151K vocab matmul that takes minutes on CPU runs in milliseconds on L4.

Usage:
    python scripts/eval_moqe_torch.py --generate-only
    python scripts/eval_moqe_torch.py --n-eval 4 --generate
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF


class MoQETorch(torch.nn.Module):
    """MoQE model ported to PyTorch for GPU inference."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.out_proj = torch.nn.Linear(d_model, vocab_size, bias=False)

        self.layers_w0 = torch.nn.ParameterList()
        self.layers_s0 = torch.nn.ParameterList()
        self.layers_w1 = torch.nn.ParameterList()
        self.layers_s1 = torch.nn.ParameterList()
        self.layers_router_w = torch.nn.ParameterList()
        self.layers_router_b = torch.nn.ParameterList()

        for _ in range(n_layers):
            self.layers_w0.append(torch.nn.Parameter(torch.zeros(1)))
            self.layers_s0.append(torch.nn.Parameter(torch.zeros(1)))
            self.layers_w1.append(torch.nn.Parameter(torch.zeros(1)))
            self.layers_s1.append(torch.nn.Parameter(torch.zeros(1)))
            self.layers_router_w.append(torch.nn.Parameter(torch.zeros(1)))
            self.layers_router_b.append(torch.nn.Parameter(torch.zeros(1)))

    def load_from_numpy(self, path: str):
        """Load weights from numpy checkpoint."""
        data = np.load(path)

        self.embedding.weight.data = torch.from_numpy(
            data["embedding"].astype(np.float32))
        self.out_proj.weight.data = torch.from_numpy(
            data["out_proj"].astype(np.float32))

        for i in range(self.n_layers):
            self.layers_w0[i] = torch.nn.Parameter(
                torch.from_numpy(data[f"layer{i}_w0_int"].astype(np.float32)))
            self.layers_s0[i] = torch.nn.Parameter(
                torch.from_numpy(data[f"layer{i}_s0"].astype(np.float32)))
            self.layers_w1[i] = torch.nn.Parameter(
                torch.from_numpy(data[f"layer{i}_w1_int"].astype(np.float32)))
            self.layers_s1[i] = torch.nn.Parameter(
                torch.from_numpy(data[f"layer{i}_s1"].astype(np.float32)))
            self.layers_router_w[i] = torch.nn.Parameter(
                torch.from_numpy(data[f"layer{i}_router_w"].astype(np.float32)))
            b_val = data[f"layer{i}_router_b"]
            if b_val.ndim == 0:
                b_val = b_val.reshape(1)
            self.layers_router_b[i] = torch.nn.Parameter(
                torch.tensor([float(b_val[0])], dtype=torch.float32))

    def dequant_forward(self, x: torch.Tensor, w_int: torch.Tensor,
                        s: torch.Tensor) -> torch.Tensor:
        """Dequantize int weights and apply linear transform.

        w_int: (out, in) int8 quantized weights
        s: (out, n_blocks) per-block scales where n_blocks = in // block_size
        """
        out_dim, in_dim = w_int.shape
        n_blocks = s.shape[1]
        block_size = in_dim // n_blocks

        # Expand scales: (out, n_blocks) → (out, n_blocks, block_size) → (out, in)
        s_expanded = s.unsqueeze(-1).expand(-1, -1, block_size).reshape(out_dim, in_dim)
        w = w_int.float() * s_expanded
        return x @ w.T

    def forward(self, input_ids: torch.Tensor):
        """Forward pass: input_ids → logits, router_probs."""
        x = self.embedding(input_ids)  # (seq, d)

        all_probs = []
        for i in range(self.n_layers):
            # Router decision: element-wise dot then sum to scalar per token
            router_logit = (x * self.layers_router_w[i]).sum(dim=-1)
            router_logit = router_logit + self.layers_router_b[i]
            prob_8bit = torch.sigmoid(router_logit)

            # Both experts (simplified: blend by router prob)
            out_4bit = self.dequant_forward(x, self.layers_w0[i], self.layers_s0[i])
            out_8bit = self.dequant_forward(x, self.layers_w1[i], self.layers_s1[i])

            # Blend
            p = prob_8bit.unsqueeze(-1)
            out = (1 - p) * out_4bit + p * out_8bit

            x = x + out  # Residual
            all_probs.append(prob_8bit.detach())

        logits = self.out_proj(x)
        router_probs = torch.stack(all_probs)
        return logits, router_probs


@torch.no_grad()
def eval_checkpoint(
    checkpoint_path: str,
    data_dir: str = "data/logits_512",
    vocab_size: int = 151_936,
    d_model: int = 2048,
    n_layers: int = 12,
    n_eval: int = 50,
    temperature: float = 2.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model (v={vocab_size}, d={d_model}, L={n_layers})...")
    model = MoQETorch(vocab_size, d_model, n_layers)
    model.load_from_numpy(checkpoint_path)
    model = model.to(device).eval()
    print(f"  Model loaded on {device}")

    data_path = Path(data_dir)
    all_seqs = sorted(data_path.glob("sequence_*.npz"))
    eval_seqs = all_seqs[-n_eval:]
    print(f"  Evaluating on {len(eval_seqs)} sequences")

    total_loss = 0.0
    total_ce = 0.0
    total_tokens = 0
    total_top1 = 0
    total_top5 = 0
    layer_8bit = torch.zeros(n_layers, device=device)
    layer_count = 0

    t0 = time.perf_counter()

    for i, seq_path in enumerate(eval_seqs):
        data = np.load(seq_path)
        input_ids = torch.from_numpy(data["input_tokens"].astype(np.int64)).to(device)
        teacher_logits = torch.from_numpy(
            data["logits"].astype(np.float32)).to(device)

        seq_len = min(len(input_ids) - 1, len(teacher_logits) - 1)
        if seq_len < 2:
            continue

        x_ids = input_ids[:seq_len]
        targets = input_ids[1:seq_len + 1]
        t_logits = teacher_logits[:seq_len]

        logits, router_probs = model(x_ids)

        # CE loss
        ce = TF.cross_entropy(logits, targets).item()

        # KD loss
        s_log = TF.log_softmax(logits / temperature, dim=-1)
        t_prob = TF.softmax(t_logits / temperature, dim=-1)
        kd = -(t_prob * s_log).sum(dim=-1).mean().item() * (temperature ** 2)

        loss = 0.5 * ce + 0.5 * kd

        # Top-1 / Top-5
        preds = logits.argmax(dim=-1)
        top1 = (preds == targets).sum().item()
        top5 = sum(targets[j] in logits[j].topk(5).indices for j in range(seq_len))

        # Router stats
        for li in range(n_layers):
            layer_8bit[li] += (router_probs[li] > 0.5).sum()
        layer_count += seq_len

        total_loss += loss * seq_len
        total_ce += ce * seq_len
        total_tokens += seq_len
        total_top1 += top1
        total_top5 += top5

        elapsed = time.perf_counter() - t0
        print(f"  [{i+1}/{len(eval_seqs)}] loss={total_loss/total_tokens:.4f} "
              f"ce={total_ce/total_tokens:.4f} "
              f"top1={total_top1/total_tokens*100:.1f}% "
              f"top5={total_top5/total_tokens*100:.1f}% "
              f"({elapsed:.1f}s)")

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / total_tokens
    avg_ce = total_ce / total_tokens
    ppl = np.exp(min(avg_ce, 20.0))
    top1_acc = total_top1 / total_tokens * 100
    top5_acc = total_top5 / total_tokens * 100
    avg_8bit = (layer_8bit.sum() / (layer_count * n_layers) * 100).item()

    print(f"\n{'='*60}")
    print(f"  MoQE E1 Checkpoint Evaluation (GPU)")
    print(f"{'='*60}")
    print(f"  Loss:       {avg_loss:.4f}")
    print(f"  CE:         {avg_ce:.4f}")
    print(f"  Perplexity: {ppl:.1f}")
    print(f"  Top-1:      {top1_acc:.2f}%")
    print(f"  Top-5:      {top5_acc:.2f}%")
    print(f"  8-bit:      {avg_8bit:.1f}%")
    print(f"  Tokens:     {total_tokens:,}")
    print(f"  Time:       {elapsed:.1f}s")
    print(f"  Tok/s:      {total_tokens/elapsed:.0f}")
    print(f"\n  Per-layer 8-bit:")
    for li in range(n_layers):
        frac = (layer_8bit[li] / layer_count * 100).item()
        print(f"    Layer {li:2d}: {frac:5.1f}%")
    print(f"{'='*60}")


@torch.no_grad()
def test_generation(
    checkpoint_path: str,
    vocab_size: int = 151_936,
    d_model: int = 2048,
    n_layers: int = 12,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-32B", trust_remote_code=True)
    except Exception:
        print("  No tokenizer — install transformers")
        return

    print(f"\nLoading model for generation on {device}...")
    model = MoQETorch(vocab_size, d_model, n_layers)
    model.load_from_numpy(checkpoint_path)
    model = model.to(device).eval()

    prompts = [
        "def fibonacci(n):",
        "The capital of France is",
        "import numpy as np\n\ndef matrix_multiply(",
        "Explain the concept of recursion in",
    ]

    print(f"\n{'='*60}")
    print(f"  MoQE Generation Test (GPU)")
    print(f"{'='*60}")

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        generated = torch.tensor(input_ids, dtype=torch.long, device=device)

        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            ctx = generated[-512:]  # context window
            logits, _ = model(ctx)
            next_logits = logits[-1] / temperature

            # Top-k
            if top_k > 0:
                vals, idx = next_logits.topk(top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(0, idx, vals)

            probs = TF.softmax(next_logits, dim=-1)
            token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, token])

        elapsed = time.perf_counter() - t0
        new_tokens = len(generated) - len(input_ids)
        tps = new_tokens / elapsed

        text = tokenizer.decode(generated.cpu(), skip_special_tokens=True)

        print(f"\n  Prompt: {prompt!r}")
        print(f"  ({new_tokens} tokens, {tps:.1f} tok/s)")
        print(f"  {'─'*50}")
        print(f"  {text[:500]}")
        print(f"  {'─'*50}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/checkpoints/checkpoint_e0_b1000.npz")
    parser.add_argument("--data-dir", default="data/logits_512")
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--d-model", type=int, default=2048)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if not args.generate_only:
        eval_checkpoint(
            args.checkpoint, args.data_dir, args.vocab,
            args.d_model, args.n_layers, args.n_eval, args.temperature,
        )

    if args.generate or args.generate_only:
        test_generation(
            args.checkpoint, args.vocab, args.d_model,
            args.n_layers, args.max_new_tokens,
        )
