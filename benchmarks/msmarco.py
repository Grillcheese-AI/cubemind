"""MS MARCO passage retrieval benchmark for CubeMind.

Evaluates CubeMind's VSA-based retrieval on the MS MARCO passage ranking
dataset. Measures MRR@10 (Mean Reciprocal Rank at 10), the standard metric
for MS MARCO passage retrieval.

Dataset: https://huggingface.co/datasets/microsoft/ms_marco
Paper:   https://arxiv.org/abs/1611.09268

CubeMind approach:
  1. Encode each passage as a block-code vector using the perception encoder
  2. Encode the query as a block-code vector
  3. Retrieve top-k passages by block-code similarity
  4. Compute MRR@10 against ground-truth relevant passages

This tests CubeMind's ability to do dense retrieval in VSA space, where
the similarity metric is block-code overlap rather than cosine similarity
on dense embeddings.

Usage:
    from benchmarks.msmarco import run_msmarco_benchmark
    results = run_msmarco_benchmark(model)

    # CLI
    python -m benchmarks.msmarco
    python -m benchmarks.msmarco --max-queries 100 --corpus-size 10000
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.ops.block_codes import BlockCodes
from cubemind.perception.encoder import Encoder
from cubemind.telemetry import metrics

logger = logging.getLogger(__name__)

# ── Dataset Configuration ─────────────────────────────────────────────────────

# Official MS MARCO dataset on Hugging Face
DATASET_ID = "microsoft/ms_marco"
DATASET_VERSION = "v2.1"

# Passage corpus (for larger-scale retrieval)
CORPUS_DATASET_ID = "Tevatron/msmarco-passage-corpus"


# ── Dataset Loading ───────────────────────────────────────────────────────────


def _check_datasets_available() -> bool:
    """Check if the HuggingFace datasets library is installed."""
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        return False


def load_msmarco_queries(
    split: str = "validation",
    max_queries: int | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict]:
    """Load MS MARCO queries with their relevant passages.

    Each returned dict has:
        - query_id: str
        - query: str (the question)
        - passages: list[dict] with keys: is_selected (0/1), passage_text, url
        - answers: list[str] (human-written answers)

    Args:
        split: Dataset split ("train" or "validation").
        max_queries: Maximum number of queries to load (None = all).
        cache_dir: Optional cache directory for downloaded data.

    Returns:
        List of query dicts with at least one relevant passage.

    Raises:
        ImportError: If datasets library is not installed.
    """
    if not _check_datasets_available():
        raise ImportError(
            "The 'datasets' library is required for the MS MARCO benchmark. "
            "Install it with: pip install datasets"
        )

    import datasets

    logger.info("Loading MS MARCO dataset: split=%s", split)

    kwargs = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    ds = datasets.load_dataset(
        DATASET_ID, DATASET_VERSION, split=split, **kwargs
    )

    queries = []
    for row in ds:
        # Only include queries that have at least one relevant passage
        passages = row.get("passages", {})
        is_selected = passages.get("is_selected", [])
        passage_texts = passages.get("passage_text", [])

        if not any(s == 1 for s in is_selected):
            continue

        queries.append({
            "query_id": str(row.get("query_id", len(queries))),
            "query": row.get("query", ""),
            "passages": [
                {
                    "is_selected": is_selected[i],
                    "passage_text": passage_texts[i],
                    "url": passages.get("url", [""])[i] if passages.get("url") else "",
                }
                for i in range(len(passage_texts))
            ],
            "answers": row.get("answers", []),
        })

        if max_queries is not None and len(queries) >= max_queries:
            break

    logger.info("Loaded %d queries with relevant passages", len(queries))
    return queries


# ── Retrieval Pipeline ────────────────────────────────────────────────────────


class VSARetriever:
    """Block-code based passage retriever for MS MARCO.

    Indexes passages as block-code vectors and retrieves by
    similarity lookup, using CubeMind's VSA operations.

    Args:
        k: Number of blocks per vector.
        l: Block length.
    """

    def __init__(self, k: int = K_BLOCKS, l: int = L_BLOCK) -> None:
        self.k = k
        self.l = l
        self.bc = BlockCodes(k, l)
        self.encoder = Encoder(k=k, l=l)

        # Index storage
        self._passages: list[str] = []
        self._codes: np.ndarray | None = None  # (n, k, l)
        self._is_indexed = False

    @property
    def n_indexed(self) -> int:
        """Number of indexed passages."""
        return len(self._passages)

    def index_passages(self, passages: list[str]) -> None:
        """Encode and index a list of passages.

        Args:
            passages: List of passage text strings.
        """
        logger.info("Indexing %d passages...", len(passages))
        t0 = time.perf_counter()

        self._passages = passages
        codes = []
        for i, passage in enumerate(passages):
            code = self.encoder.encode(passage)
            codes.append(code)
            if (i + 1) % 1000 == 0:
                logger.info("  Indexed %d/%d passages", i + 1, len(passages))

        self._codes = np.stack(codes, axis=0)  # (n, k, l)
        self._is_indexed = True

        elapsed = time.perf_counter() - t0
        logger.info(
            "Indexing complete: %d passages in %.1fs (%.0f passages/sec)",
            len(passages), elapsed, len(passages) / max(elapsed, 0.001),
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Retrieve the top-k most similar passages to the query.

        Args:
            query: Query text string.
            top_k: Number of results to return.

        Returns:
            List of (passage_index, similarity_score) tuples, sorted by
            descending similarity.
        """
        if not self._is_indexed:
            raise RuntimeError("Call index_passages() before retrieve()")

        query_code = self.encoder.encode(query)
        sims = self.bc.similarity_batch(query_code, self._codes)  # (n,)

        # Get top-k indices
        k_actual = min(top_k, len(self._passages))
        if k_actual >= len(self._passages):
            top_indices = np.argsort(sims)[::-1][:k_actual]
        else:
            top_indices = np.argpartition(sims, -k_actual)[-k_actual:]
            top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        return [(int(idx), float(sims[idx])) for idx in top_indices]


# ── Metrics ───────────────────────────────────────────────────────────────────


def reciprocal_rank(
    retrieved_indices: list[int],
    relevant_indices: set[int],
    k: int = 10,
) -> float:
    """Compute the reciprocal rank at k.

    Args:
        retrieved_indices: List of retrieved passage indices (ordered by rank).
        relevant_indices: Set of ground-truth relevant passage indices.
        k: Cutoff rank.

    Returns:
        1/rank of the first relevant result, or 0 if none in top-k.
    """
    for rank, idx in enumerate(retrieved_indices[:k], start=1):
        if idx in relevant_indices:
            return 1.0 / rank
    return 0.0


# ── Main Benchmark Runner ─────────────────────────────────────────────────────


def run_msmarco_benchmark(
    model=None,
    split: str = "validation",
    max_queries: int | None = 100,
    corpus_size: int | None = None,
    top_k: int = 10,
    cache_dir: str | Path | None = None,
) -> dict:
    """Run the MS MARCO passage retrieval benchmark.

    For each query in the evaluation split:
      1. Index all candidate passages from the query's passage set
      2. Encode and retrieve using VSA similarity
      3. Compute MRR@10

    Note: This operates in the re-ranking setting (each query has ~10
    candidate passages to rank), not the full retrieval setting (8.8M
    passages). For full retrieval, index the Tevatron/msmarco-passage-corpus.

    Args:
        model: CubeMind instance (not used currently; reserved for
            future integration where the model's memory aids retrieval).
        split: Dataset split ("validation").
        max_queries: Maximum queries to evaluate (None = all).
        corpus_size: Not used in re-ranking mode; reserved for full
            corpus retrieval.
        top_k: Number of results to retrieve per query.
        cache_dir: Cache directory for dataset downloads.

    Returns:
        Dict with:
            mrr_at_10: float
            n_queries: int
            avg_latency_ms: float
            recall_at_k: float
            wall_clock_s: float
    """
    if not _check_datasets_available():
        logger.error(
            "MS MARCO benchmark requires 'datasets' library. "
            "Install with: pip install datasets"
        )
        return {
            "mrr_at_10": 0.0,
            "n_queries": 0,
            "avg_latency_ms": 0.0,
            "recall_at_k": 0.0,
            "wall_clock_s": 0.0,
            "error": "datasets library not installed",
        }

    benchmark_start = time.perf_counter()

    # Load queries
    queries = load_msmarco_queries(split, max_queries, cache_dir)

    if not queries:
        logger.error("No queries loaded from MS MARCO")
        return {
            "mrr_at_10": 0.0,
            "n_queries": 0,
            "avg_latency_ms": 0.0,
            "recall_at_k": 0.0,
            "wall_clock_s": 0.0,
            "error": "no queries loaded",
        }

    retriever = VSARetriever()
    total_rr = 0.0
    total_recall = 0.0
    total_latency = 0.0
    per_query = []

    for q_idx, query_data in enumerate(queries):
        # Collect all passages for this query
        passages = [p["passage_text"] for p in query_data["passages"]]
        relevant = {
            i for i, p in enumerate(query_data["passages"])
            if p["is_selected"] == 1
        }

        if not passages or not relevant:
            continue

        # Index passages for this query (re-ranking mode)
        retriever.index_passages(passages)

        # Retrieve
        t0 = time.perf_counter()
        results = retriever.retrieve(query_data["query"], top_k=top_k)
        latency = (time.perf_counter() - t0) * 1000

        retrieved_indices = [idx for idx, _ in results]
        rr = reciprocal_rank(retrieved_indices, relevant, k=top_k)

        # Recall@k
        retrieved_set = set(retrieved_indices[:top_k])
        recall = len(retrieved_set & relevant) / len(relevant)

        total_rr += rr
        total_recall += recall
        total_latency += latency

        per_query.append({
            "query_id": query_data["query_id"],
            "query": query_data["query"][:100],
            "rr": rr,
            "recall": recall,
            "latency_ms": latency,
            "n_passages": len(passages),
            "n_relevant": len(relevant),
        })

        if (q_idx + 1) % 50 == 0:
            running_mrr = total_rr / (q_idx + 1)
            logger.info(
                "  Evaluated %d/%d queries, running MRR@%d=%.4f",
                q_idx + 1, len(queries), top_k, running_mrr,
            )

    n_evaluated = len(per_query)
    mrr = total_rr / max(n_evaluated, 1)
    avg_recall = total_recall / max(n_evaluated, 1)
    avg_latency = total_latency / max(n_evaluated, 1)
    wall_clock = time.perf_counter() - benchmark_start

    metrics.record("benchmark.msmarco.mrr_at_10", mrr)
    metrics.record("benchmark.msmarco.recall_at_k", avg_recall)
    metrics.record("benchmark.msmarco.latency_ms", avg_latency)

    return {
        "mrr_at_10": mrr,
        "n_queries": n_evaluated,
        "avg_latency_ms": avg_latency,
        "recall_at_k": avg_recall,
        "wall_clock_s": wall_clock,
        "top_k": top_k,
        "per_query": per_query,
    }


# ── Pretty Printing ───────────────────────────────────────────────────────────


def print_results(results: dict) -> None:
    """Print benchmark results in a readable format."""
    print()
    print("=" * 60)
    print("  MS MARCO Passage Retrieval Benchmark")
    print("=" * 60)
    print()

    if "error" in results:
        print(f"  ERROR: {results['error']}")
        print()
        return

    print(f"  MRR@{results.get('top_k', 10):>2}:        {results['mrr_at_10']:.4f}")
    print(f"  Recall@{results.get('top_k', 10):>2}:     {results['recall_at_k']:.4f}")
    print(f"  Queries:         {results['n_queries']}")
    print(f"  Avg latency:     {results['avg_latency_ms']:.1f}ms")
    print(f"  Wall clock:      {results['wall_clock_s']:.1f}s")
    print()

    # Context
    print("  Reference baselines:")
    print("    BM25:             ~0.187 MRR@10")
    print("    Dense (ANCE):     ~0.330 MRR@10")
    print("    ColBERT v2:       ~0.397 MRR@10")
    print("    CubeMind target:  TBD (first VSA retrieval baseline)")
    print()


# ── CLI Entry Point ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run the MS MARCO passage retrieval benchmark on CubeMind",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "train"],
        help="Dataset split (default: validation)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=100,
        help="Max queries to evaluate (default: 100)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Retrieve top-k passages (default: 10)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for dataset downloads",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    results = run_msmarco_benchmark(
        split=args.split,
        max_queries=args.max_queries,
        top_k=args.top_k,
        cache_dir=args.cache_dir,
    )

    print_results(results)


if __name__ == "__main__":
    main()
