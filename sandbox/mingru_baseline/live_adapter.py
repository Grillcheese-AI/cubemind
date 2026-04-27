"""LiveAdapter — bridge a stage-2 checkpoint to live inference + online learning.

Wraps the trained ``MinGRUMultiTask`` model (backbone + LM head + 5 multitask
heads + hippocampal memory) so the live brain loop can use it without knowing
about the training-time scaffolding.

The trained model already exposes everything needed for online use:

  * The backbone is frozen at inference (set ``requires_grad=False`` on every
    backbone parameter; same as ``--freeze-backbone`` did during stage 2).
  * Each ``MindForgeLoRAHead`` has an ``online_update`` method that runs a
    single NLMS step on its plastic ``basis_B`` parameter only — the rest of
    the head + the backbone never move at inference.
  * The backbone owns a shared ``EpisodicMemory`` (the same one the trainer
    used) — same ``write`` / ``read`` API.

Usage::

    from live_adapter import LiveAdapter

    bot = LiveAdapter.load(
        checkpoint="results_5090/stage2_multitask/best.pt",
        tokenizer ="/workspace/tokenizer/grillcheese_spm32k_v2.model",
    )

    # Inference — returns generated text + per-head predictions + pooled features
    out = bot.forward(text="user wants to compare two routes")
    print(out["generated"])
    print("opcode:", out["heads"]["opcode"]["top1_name"])

    # Online correction — single NLMS step on the opcode head's basis_B only
    bot.online_update("opcode", target_id=42, lr=1e-3, pooled=out["pooled"])

    # Persist the updated heads for the next session
    bot.save("results_5090/stage2_multitask/live.pt")
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Make sibling modules (train_torch, vsa_binding_head, vm_opcodes, ...) importable.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from train_torch import (  # noqa: E402
    DEFAULT_HEAD_SPECS,
    GenParams,
    MinGRUModel,
    MinGRUMultiTask,
    SpmAdapter,
    TrainConfig,
    generate,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _strip_compile_prefix(state_dict: dict) -> dict:
    """torch.compile saves with ``_orig_mod.`` prefix; strip it for plain load."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _build_cfg_from_dict(d: dict) -> TrainConfig:
    """Reconstruct a ``TrainConfig`` from a saved ``asdict(cfg)`` payload,
    ignoring any fields the dataclass no longer has."""
    valid = {f for f in TrainConfig.__dataclass_fields__}
    return TrainConfig(**{k: v for k, v in d.items() if k in valid})


# ── adapter ─────────────────────────────────────────────────────────────────

class LiveAdapter:
    """Inference-time wrapper around a trained ``MinGRUMultiTask``."""

    def __init__(
        self,
        model: MinGRUMultiTask,
        tokenizer: SpmAdapter,
        cfg: TrainConfig,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.step = 0      # counts forward calls — useful for live telemetry

        # Resolve every head's label vocabulary so we can map argmax → name.
        # Falls back to numeric id if no label mapping is provided by the spec.
        self._head_label_names: dict[str, list[str] | None] = {}
        for spec in self.model.head_specs:
            name = spec["name"]
            self._head_label_names[name] = spec.get("label_names")

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        tokenizer: str | Path,
        device: str | torch.device | None = None,
    ) -> "LiveAdapter":
        """Build the model from a stage-2 checkpoint.

        ``checkpoint`` should be a ``best.pt`` saved by ``train_torch.save_checkpoint``
        — the file embeds the full TrainConfig, so we can reconstruct the model
        without any external metadata.
        """
        ckpt_path = Path(checkpoint).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        data = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        cfg_dict = data.get("config")
        if cfg_dict is None:
            raise RuntimeError(
                "checkpoint missing 'config' — was it saved by an old trainer? "
                "Re-save with the current train_torch.save_checkpoint."
            )
        cfg = _build_cfg_from_dict(cfg_dict)

        # Same construction order as train_torch.train()
        torch.manual_seed(cfg.seed)
        backbone = MinGRUModel(cfg)
        model = MinGRUMultiTask(
            backbone=backbone,
            d_model=cfg.d_model,
            head_specs=DEFAULT_HEAD_SPECS,
            cfg=cfg,
        ).to(device)

        sd = _strip_compile_prefix(data["model_state"])
        result = model.load_state_dict(sd, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"  load_state_dict: {len(result.missing_keys)} missing, "
                  f"{len(result.unexpected_keys)} unexpected — usually fine "
                  f"(buffers / new heads / removed heads)")

        model.eval()
        # Freeze everything by default — online_update temporarily flips
        # requires_grad on basis_B for its NLMS step.
        for p in model.parameters():
            p.requires_grad = False

        # Tokenizer
        tok_path = Path(tokenizer).resolve()
        if not tok_path.exists():
            raise FileNotFoundError(tok_path)
        tok = SpmAdapter(str(tok_path))

        adapter = cls(model=model, tokenizer=tok, cfg=cfg, device=device)
        print(f"  LiveAdapter ready: d={cfg.d_model} L={cfg.n_layers} "
              f"vocab={cfg.vocab_size} dtype={cfg.amp_dtype} on {device}")
        print(f"  heads: {[s['name'] for s in model.head_specs]}")
        return adapter

    # ── inference ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        text: str,
        generate_continuation: bool = False,
        gen_params: GenParams | None = None,
    ) -> dict[str, Any]:
        """One forward pass on a text input. Returns the per-head top-1
        predictions, the pooled hidden state, and (optionally) a generated
        continuation.

        ``generate_continuation=True`` runs the autoregressive loop using
        ``train_torch.generate`` (slow — only do this for chat / demo use).
        """
        self.step += 1
        ids = self.tokenizer.encode(text).ids
        if not ids:
            raise ValueError(f"tokenizer produced no ids for {text!r}")
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        last_idx = torch.tensor([len(ids) - 1], dtype=torch.long,
                                device=self.device)

        out = self.model(tokens, last_idx=last_idx)
        pooled = out["_pooled"][0]   # (D,)

        heads_out: dict[str, dict] = {}
        for spec in self.model.head_specs:
            name = spec["name"]
            logits = out[f"{name}_logits"][0]
            probs = F.softmax(logits.float(), dim=-1)
            top1 = int(probs.argmax().item())
            top1_p = float(probs[top1].item())
            label_names = self._head_label_names.get(name)
            top1_name = (label_names[top1]
                         if label_names and top1 < len(label_names)
                         else str(top1))
            heads_out[name] = {
                "top1_id":   top1,
                "top1_name": top1_name,
                "top1_prob": top1_p,
                "logits":    logits.detach(),
            }

        result: dict[str, Any] = {
            "step":               self.step,
            "input_tokens":       len(ids),
            "pooled":             pooled.detach(),
            "heads":              heads_out,
            "memories_retrieved": self._mem_size(),
        }
        if generate_continuation:
            params = gen_params or GenParams()
            # ``generate`` expects a MinGRUModel — pass the backbone, which
            # has the same forward shape (token_logits at the LM head).
            result["generated"] = generate(
                self.model.backbone, self.tokenizer, text, params, self.device,
            )
        return result

    # ── online learning ────────────────────────────────────────────────────

    def online_update(
        self,
        head: str,
        target_id: int,
        pooled: torch.Tensor | None = None,
        text: str | None = None,
        lr: float = 1e-3,
    ) -> float:
        """Single NLMS step on ``heads[head].basis_B`` only.

        Provide either:
          * ``pooled`` from a previous ``forward()`` call (faster, recommended),
          * or ``text`` to re-encode and pool inside the update.

        Returns the pre-update loss for plasticity logging.
        """
        if pooled is None:
            if text is None:
                raise ValueError("supply either pooled or text")
            pooled = self.forward(text)["pooled"]
        target = torch.tensor([target_id], dtype=torch.long, device=self.device)
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        return float(self.model.online_update(head, pooled, target, lr=lr))

    # ── memory ──────────────────────────────────────────────────────────────

    def write_memory(self, key: torch.Tensor, value: torch.Tensor | None = None,
                     utility: float = 1.0) -> bool:
        """Store an episode in the backbone's hippocampal memory. ``value``
        defaults to ``key`` — usual case is "remember this hidden state"."""
        mem = self.model.backbone.memory
        if mem is None:
            raise RuntimeError(
                "backbone has no episodic memory — was --memory enabled at training?"
            )
        return mem.write(key, value if value is not None else key, utility=utility)

    def recall(self, query: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Retrieve a single (D,) read from the hippocampal memory ranked by
        ``cosine_sim × utility`` over the top-K hits. See ``EpisodicMemory.read``."""
        mem = self.model.backbone.memory
        if mem is None or mem.size == 0:
            return torch.zeros(self.cfg.d_model, device=self.device)
        return mem.read(query, k=k)

    def _mem_size(self) -> int:
        mem = self.model.backbone.memory
        return 0 if mem is None else mem.size

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the (possibly) updated model state. Useful after a live
        session that called ``online_update`` — keeps the head improvements."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config":      asdict(self.cfg),
            "live_step":   self.step,
        }, str(path))
        print(f"  saved live adapter to {path}")

    def stats(self) -> dict[str, Any]:
        return {
            "step":        self.step,
            "memory_size": self._mem_size(),
            "params":      sum(p.numel() for p in self.model.parameters()),
            "trainable":   sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad),
            "device":      str(self.device),
            "dtype":       self.cfg.amp_dtype,
        }


# ── demo / smoke-test ──────────────────────────────────────────────────────

def _demo(checkpoint: str, tokenizer: str) -> None:
    """Tiny round-trip: forward → show heads → online_update → re-forward."""
    bot = LiveAdapter.load(checkpoint, tokenizer)
    text = "compare these two routes by total distance"
    print(f"\n  prompt: {text!r}")

    out1 = bot.forward(text)
    for name, h in out1["heads"].items():
        print(f"    {name:<10} top1={h['top1_name']:<20} p={h['top1_prob']:.3f}")

    # Pretend we know the right opcode is COMPARE (id varies — use some valid id).
    target = 0  # safe id within all head sizes
    loss = bot.online_update("opcode", target_id=target, pooled=out1["pooled"])
    print(f"\n  online_update opcode→{target} (pre-loss {loss:.4f})")

    out2 = bot.forward(text)
    print(f"  post-update opcode top1={out2['heads']['opcode']['top1_name']}")
    print(f"  stats: {bot.stats()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True,
                    help="Path to a stage-2 best.pt")
    ap.add_argument("--tokenizer", required=True,
                    help="Path to grillcheese_spm32k_v2.model")
    args = ap.parse_args()
    _demo(args.checkpoint, args.tokenizer)
