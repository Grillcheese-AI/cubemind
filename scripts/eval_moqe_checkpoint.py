"""Evaluate a MoQE checkpoint on held-out teacher logit sequences.

Reports: loss, CE, KD, perplexity, 8-bit fraction, top-1/top-5 accuracy,
router distribution stats per layer.

Usage:
    python scripts/eval_moqe_checkpoint.py
    python scripts/eval_moqe_checkpoint.py --checkpoint data/checkpoints/checkpoint_e0_b1000.npz
    python scripts/eval_moqe_checkpoint.py --n-eval 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cubemind.execution.moqe import MoQEModel
from cubemind.training.moqe_distillation import load_checkpoint


def load_sequence(path: Path) -> dict:
    """Load a preprocessed teacher logit sequence."""
    data = np.load(path)
    return {
        "input_ids": data["input_tokens"],
        "teacher_logits": data["logits"].astype(np.float32),
    }


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def eval_checkpoint(
    checkpoint_path: str,
    data_dir: str = "data/logits_512",
    vocab_size: int = 151_936,
    d_model: int = 2048,
    n_layers: int = 12,
    n_eval: int = 50,
    temperature: float = 2.0,
) -> dict:
    """Evaluate checkpoint on held-out sequences."""

    print(f"Loading model (v={vocab_size}, d={d_model}, L={n_layers})...")
    model = MoQEModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    info = load_checkpoint(model, checkpoint_path)
    print("  Checkpoint loaded.")

    # Get sequences — use last n_eval as held-out
    data_path = Path(data_dir)
    all_seqs = sorted(data_path.glob("sequence_*.npz"))
    eval_seqs = all_seqs[-n_eval:]
    print(f"  Evaluating on {len(eval_seqs)} held-out sequences "
          f"(of {len(all_seqs)} total)")

    # Metrics accumulators
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_tokens = 0
    total_correct_top1 = 0
    total_correct_top5 = 0
    layer_8bit_counts = np.zeros(n_layers)
    layer_token_counts = np.zeros(n_layers)

    t0 = time.perf_counter()

    for i, seq_path in enumerate(eval_seqs):
        seq = load_sequence(seq_path)
        input_ids = seq["input_ids"]
        teacher_logits = seq["teacher_logits"]

        # Truncate to match lengths
        seq_len = min(len(input_ids) - 1, len(teacher_logits) - 1)
        if seq_len < 2:
            continue

        x_ids = input_ids[:seq_len]
        targets = input_ids[1:seq_len + 1]
        t_logits = teacher_logits[:seq_len]

        # Forward pass
        logits, router_probs = model.forward(x_ids)

        # CE loss (student vs ground truth)
        log_probs = logits - np.log(
            np.sum(np.exp(logits - np.max(logits, axis=-1, keepdims=True)),
                   axis=-1, keepdims=True)
        ) - np.max(logits, axis=-1, keepdims=True)
        # Proper log-softmax
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))

        ce = -np.mean(log_probs[np.arange(seq_len), targets])

        # KD loss (student vs teacher)
        student_log_soft = logits / temperature
        s_shifted = student_log_soft - np.max(student_log_soft, axis=-1, keepdims=True)
        s_log_probs = s_shifted - np.log(np.sum(np.exp(s_shifted), axis=-1, keepdims=True))

        teacher_soft = t_logits / temperature
        t_shifted = teacher_soft - np.max(teacher_soft, axis=-1, keepdims=True)
        t_probs = np.exp(t_shifted - np.log(
            np.sum(np.exp(t_shifted), axis=-1, keepdims=True)))

        kd = -np.mean(np.sum(t_probs * s_log_probs, axis=-1)) * (temperature ** 2)

        loss = 0.5 * ce + 0.5 * kd

        # Top-1 / Top-5 accuracy
        preds = np.argmax(logits, axis=-1)
        top1 = np.sum(preds == targets)

        top5_preds = np.argsort(logits, axis=-1)[:, -5:]
        top5 = np.sum([targets[j] in top5_preds[j] for j in range(seq_len)])

        # Router stats per layer
        for li in range(n_layers):
            is_8bit = router_probs[li] > 0.5
            layer_8bit_counts[li] += np.sum(is_8bit)
            layer_token_counts[li] += seq_len

        total_loss += loss * seq_len
        total_ce += ce * seq_len
        total_kd += kd * seq_len
        total_tokens += seq_len
        total_correct_top1 += top1
        total_correct_top5 += top5

        if (i + 1) % 10 == 0:
            avg_loss = total_loss / total_tokens
            print(f"  [{i+1}/{len(eval_seqs)}] loss={avg_loss:.4f} "
                  f"top1={total_correct_top1/total_tokens*100:.1f}% "
                  f"top5={total_correct_top5/total_tokens*100:.1f}%")

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / total_tokens
    avg_ce = total_ce / total_tokens
    avg_kd = total_kd / total_tokens
    ppl = np.exp(min(avg_ce, 20.0))  # cap to avoid overflow
    top1_acc = total_correct_top1 / total_tokens * 100
    top5_acc = total_correct_top5 / total_tokens * 100
    overall_8bit = np.sum(layer_8bit_counts) / np.sum(layer_token_counts) * 100

    print("\n" + "=" * 60)
    print("  MoQE E1 Checkpoint Evaluation")
    print("=" * 60)
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Eval seqs:   {len(eval_seqs)} (held-out)")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Time:         {elapsed:.1f}s")
    print()
    print(f"  Loss:         {avg_loss:.4f}")
    print(f"  CE:           {avg_ce:.4f}")
    print(f"  KD:           {avg_kd:.4f}")
    print(f"  Perplexity:   {ppl:.1f}")
    print(f"  Top-1 acc:    {top1_acc:.2f}%")
    print(f"  Top-5 acc:    {top5_acc:.2f}%")
    print(f"  8-bit frac:   {overall_8bit:.1f}%")
    print()
    print("  Per-layer 8-bit routing:")
    for li in range(n_layers):
        frac = layer_8bit_counts[li] / layer_token_counts[li] * 100
        print(f"    Layer {li:2d}: {frac:5.1f}%")
    print("=" * 60)

    return {
        "loss": avg_loss,
        "ce": avg_ce,
        "kd": avg_kd,
        "perplexity": ppl,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "eight_bit_frac": overall_8bit,
        "total_tokens": total_tokens,
    }


def generate(
    model: MoQEModel,
    prompt_ids: np.ndarray,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> np.ndarray:
    """Autoregressive generation from the MoQE model."""
    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        # Use last 512 tokens as context window
        ctx = np.array(generated[-512:], dtype=np.int32)
        logits, _ = model.forward(ctx)

        # Take logits for last position
        next_logits = logits[-1].astype(np.float64)

        # Temperature
        if temperature > 0:
            next_logits = next_logits / temperature
        else:
            # Greedy
            token = int(np.argmax(next_logits))
            generated.append(token)
            continue

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = np.argsort(next_logits)[:-top_k]
            next_logits[indices_to_remove] = -np.inf

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(next_logits)[::-1]
            sorted_logits = next_logits[sorted_indices]
            sorted_probs = softmax(sorted_logits)
            cumsum = np.cumsum(sorted_probs)
            cutoff = np.searchsorted(cumsum, top_p) + 1
            remove_indices = sorted_indices[cutoff:]
            next_logits[remove_indices] = -np.inf

        # Sample
        probs = softmax(next_logits)
        token = int(np.random.choice(len(probs), p=probs))
        generated.append(token)

    return np.array(generated, dtype=np.int32)


def test_generation(
    checkpoint_path: str,
    vocab_size: int = 151_936,
    d_model: int = 2048,
    n_layers: int = 12,
    prompts: list[str] | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
) -> None:
    """Test generation quality with sample prompts."""

    # Try to load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-Next-80B",
                                                   trust_remote_code=True)
    except Exception:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B",
                                                       trust_remote_code=True)
        except Exception:
            print("  No tokenizer available — skipping generation test.")
            print("  Install: pip install transformers")
            return

    print(f"\nLoading model for generation...")
    model = MoQEModel(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)
    load_checkpoint(model, checkpoint_path)

    if prompts is None:
        prompts = [
            "def fibonacci(n):",
            "The capital of France is",
            "import numpy as np\n\ndef matrix_multiply(",
            "Explain the concept of recursion in",
        ]

    print("\n" + "=" * 60)
    print("  MoQE Generation Test")
    print("=" * 60)

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = np.array(input_ids, dtype=np.int32)

        t0 = time.perf_counter()
        output_ids = generate(model, input_ids, max_new_tokens=max_new_tokens,
                              temperature=temperature)
        elapsed = time.perf_counter() - t0

        new_tokens = len(output_ids) - len(input_ids)
        tps = new_tokens / elapsed if elapsed > 0 else 0

        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        print(f"\n  Prompt: {prompt!r}")
        print(f"  Output ({new_tokens} tokens, {tps:.1f} tok/s):")
        print(f"  {'─' * 50}")
        # Show first 500 chars
        print(f"  {output_text[:500]}")
        print(f"  {'─' * 50}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="data/checkpoints/checkpoint_e0_b1000.npz")
    parser.add_argument("--data-dir", type=str, default="data/logits_512")
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--d-model", type=int, default=2048)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--generate", action="store_true",
                        help="Run generation test after eval")
    parser.add_argument("--generate-only", action="store_true",
                        help="Skip eval, only run generation test")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if not args.generate_only:
        eval_checkpoint(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            vocab_size=args.vocab,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_eval=args.n_eval,
            temperature=args.temperature,
        )

    if args.generate or args.generate_only:
        test_generation(
            checkpoint_path=args.checkpoint,
            vocab_size=args.vocab,
            d_model=args.d_model,
            n_layers=args.n_layers,
            max_new_tokens=args.max_new_tokens,
        )
