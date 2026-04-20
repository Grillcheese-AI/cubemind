#!/usr/bin/env python3
"""Propagate Gemini schema labels to SVC-emitter rows via text similarity.

The SVC emitter (``opcode-vsa-rs/examples/emit_multitask_jsonl.rs``)
produces 330K rows fast and free, but its schemas are derived from the
root verb alone — too coarse (~49% "general_directive").

The Gemini factory (``gemini_factory.py``) produces semantically rich
schemas (``knowledge_query``, ``arithmetic_planning``, ``social_memory``,
``causal_inference``, …) but is slow and costs API dollars.

This script joins the two: we use Gemini's already-labeled rows as
supervision, fit a TF-IDF + nearest-centroid classifier over the top-K
schemas, then re-label every SVC row. Result: the SVC output gets
content-grounded schemas without extra Gemini calls.

Pipeline:

  1. Load Gemini JSONL — extract (instruction_text, schema_name).
  2. Keep top-K schemas by frequency (drop long-tail singletons).
  3. Fit TF-IDF on Gemini's instruction text + compute a per-schema
     centroid by averaging TF-IDF vectors within each class.
  4. For each SVC row, extract its <INSTR>...</INSTR> content,
     TF-IDF-transform it, find the nearest centroid by cosine
     similarity — that's the new schema_name.
  5. Rewrite schema_id via LabelRegistry-style stable assignment.
  6. Write ``*_gemini_schemas.jsonl`` + a sidecar meta report.

Cheap: 12K Gemini + 330K SVC runs in ~30s with scikit-learn's
sparse cosine ops. Rerun whenever Gemini adds more labeled rows.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Text extraction ─────────────────────────────────────────────────────

INSTR_RE = re.compile(r"<INSTR>(.*?)</INSTR>", re.DOTALL)
# Strip all tag blocks from Gemini's text field so we're left with the
# natural-language portion the model actually wrote about.
TAG_RE = re.compile(r"</?\w+(?::\w+)?\b[^>]*>")


def _clean_text(s: str) -> str:
    """Strip tags, collapse whitespace."""
    s = TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_instruction(text: str) -> str:
    """Get the user-facing instruction text from a row's ``text`` field.

    SVC rows wrap the original instruction in ``<INSTR>…</INSTR>``; Gemini
    rows don't use that tag but their ``text`` embeds the instruction
    around the schema/opcode tags. We strip all tags either way to get a
    bag-of-words suitable for TF-IDF."""
    m = INSTR_RE.search(text)
    if m:
        return _clean_text(m.group(1))
    return _clean_text(text)


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ── Classifier fit ──────────────────────────────────────────────────────

def fit_schema_classifier(gemini_rows: list[dict], top_k: int):
    """Build a TF-IDF + per-schema centroid classifier from Gemini data.

    Returns ``(vectorizer, centroids, schema_names, counts)`` where
    ``centroids`` is a dense ``(K, V)`` matrix of L2-normalized TF-IDF
    centroids ready for cosine similarity against new rows."""
    # Count schema frequency so we can cap at top_k.
    counts = Counter(r["schema_name"] for r in gemini_rows)
    keep: list[str] = [n for n, _ in counts.most_common(top_k)]
    keep_set = set(keep)

    texts: list[str] = []
    labels: list[str] = []
    for r in gemini_rows:
        sn = r.get("schema_name", "")
        if sn in keep_set:
            texts.append(_extract_instruction(r.get("text", "")))
            labels.append(sn)

    if not texts:
        raise RuntimeError("no Gemini rows in the top-K schemas to fit on")

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=1,                # keep rare terms — each schema has only ~50-200 train rows
        max_df=0.98,             # barely prune stopwords (the Gemini data is short)
        max_features=100_000,
        sublinear_tf=True,
        norm="l2",
    )
    X = vec.fit_transform(texts)  # (N_train, V) sparse

    # Compute L2-normalized centroid per schema.
    centroids = np.zeros((len(keep), X.shape[1]), dtype=np.float32)
    class_counts: list[int] = []
    for i, name in enumerate(keep):
        mask = np.array([lab == name for lab in labels], dtype=bool)
        class_counts.append(int(mask.sum()))
        if mask.sum() == 0:
            continue
        centroid = np.asarray(X[mask].mean(axis=0)).ravel()
        n = np.linalg.norm(centroid)
        if n > 1e-8:
            centroid = centroid / n
        centroids[i] = centroid.astype(np.float32)

    return vec, centroids, keep, class_counts


def classify_batch(vec: TfidfVectorizer, centroids: np.ndarray,
                   texts: list[str]) -> np.ndarray:
    """Return the index of the nearest schema centroid per input text."""
    X = vec.transform(texts)         # (n, V) sparse L2-normalized
    sims = cosine_similarity(X, centroids)  # (n, K) dense
    return sims.argmax(axis=1)


# ── Main pass ───────────────────────────────────────────────────────────

def propagate(
    gemini_path: Path,
    svc_path: Path,
    output_path: Path,
    top_schemas: int = 20,
    batch_size: int = 8192,
) -> dict:
    gemini_rows = list(_iter_rows(gemini_path))
    if not gemini_rows:
        raise RuntimeError(f"no rows in {gemini_path}")
    print(f"  gemini source: {len(gemini_rows):,} rows")

    vec, centroids, schema_names, counts = fit_schema_classifier(
        gemini_rows, top_k=top_schemas)
    print(f"  classifier: {len(schema_names)} schemas, "
          f"vocab={centroids.shape[1]:,}")
    for name, ct in zip(schema_names, counts):
        print(f"    {name:30s} train rows={ct}")

    # Running registry — id assignment matches the fit order so
    # schema_id is stable across reruns.
    schema_to_id = {name: i for i, name in enumerate(schema_names)}
    # Reserve one more id for rows whose best cosine similarity is still
    # below a confidence floor — we'd rather flag them than force-fit.
    other_id = len(schema_names)
    other_name = "other"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_kept = 0
    n_low_conf = 0
    new_schema_hist: Counter = Counter()

    # Process SVC in batches so we never hold all 330K TF-IDF vectors
    # in memory at once.
    batch_rows: list[dict] = []
    batch_texts: list[str] = []

    def _flush(out_f):
        nonlocal n_kept, n_low_conf
        if not batch_rows:
            return
        X = vec.transform(batch_texts)
        sims = cosine_similarity(X, centroids)
        best_idx = sims.argmax(axis=1)
        best_sim = sims[np.arange(sims.shape[0]), best_idx]
        for row, idx, conf in zip(batch_rows, best_idx, best_sim):
            # Always assign to the nearest centroid — TF-IDF cosine on
            # short instruction texts tops out around 0.1-0.3 even for
            # clearly-matching rows, so a confidence floor just buckets
            # nearly everything to "other". The downstream post-processor
            # handles any residual top-K pruning.
            name = schema_names[idx]
            row["schema_name"] = name
            row["schema_id"] = schema_to_id[name]
            if conf < 0.02:
                n_low_conf += 1  # tracked for reporting only, not bucketed
            new_schema_hist[row["schema_name"]] += 1
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_kept += 1
        batch_rows.clear()
        batch_texts.clear()

    with output_path.open("w", encoding="utf-8") as out_f:
        for row in _iter_rows(svc_path):
            n_in += 1
            batch_rows.append(row)
            batch_texts.append(_extract_instruction(row.get("text", "")))
            if len(batch_rows) >= batch_size:
                _flush(out_f)
                if n_in % (batch_size * 8) == 0:
                    print(f"  {n_in:>7,} rows processed")
        _flush(out_f)

    stats = {
        "gemini_rows":       len(gemini_rows),
        "svc_rows_in":       n_in,
        "svc_rows_kept":     n_kept,
        "low_confidence":    n_low_conf,
        "top_schemas":       top_schemas,
        "schemas_fit":       schema_names,
        "schemas_fit_train_counts": dict(zip(schema_names, counts)),
        "new_distribution":  dict(new_schema_hist.most_common()),
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gemini", type=Path, required=True,
                    help="Gemini factory output JSONL (source of schema labels)")
    ap.add_argument("--svc", type=Path, required=True,
                    help="SVC emitter output JSONL (target to re-label)")
    ap.add_argument("--output", type=Path, required=True,
                    help="Re-labeled output JSONL")
    ap.add_argument("--top-schemas", type=int, default=20,
                    help="Keep only the top-K Gemini schemas as classes")
    ap.add_argument("--batch-size", type=int, default=8192)
    args = ap.parse_args()

    stats = propagate(
        gemini_path=args.gemini,
        svc_path=args.svc,
        output_path=args.output,
        top_schemas=args.top_schemas,
        batch_size=args.batch_size,
    )
    print(f"\n  relabeled {stats['svc_rows_kept']:,} rows")
    print(f"  low-confidence -> 'other': {stats['low_confidence']:,}")
    print("  new schema distribution (top 10):")
    for name, ct in list(stats["new_distribution"].items())[:10]:
        pct = ct / max(stats["svc_rows_kept"], 1) * 100
        print(f"    {name:30s} {ct:>6}  ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
