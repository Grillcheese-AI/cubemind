"""Fast Taxonomy Categorizer — 160 subdomains, 17K patterns, two-tier matching.

Loads subdomain_taxonomy_patterns.jsonl and builds a high-performance
text categorizer using:
  1. Fast path: subword hash set intersection (~100 lookups)
  2. Precise path: compiled regex on top candidates only (~300 regex)

Performance: ~10K+ texts/sec on CPU for the fast path alone.

Usage:
    cat = Categorizer()
    result = cat.categorize("Einstein discovered relativity")
    # {'subdomain': 'physics', 'score': 5, 'all_scores': {...}}

    # Batch
    results = cat.categorize_batch(["text1", "text2", ...])
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default taxonomy path
_DEFAULT_TAXONOMY = Path(__file__).resolve().parent.parent.parent / "data" / "subdomain_taxonomy_patterns.jsonl"


class Categorizer:
    """Two-tier taxonomy categorizer.

    Tier 1 (fast): Tokenize text → intersect with precomputed subword sets.
    Tier 2 (precise): Run compiled regex on top-N candidates from Tier 1.

    Args:
        taxonomy_path: Path to subdomain_taxonomy_patterns.jsonl.
        top_n_candidates: Number of candidates from Tier 1 for Tier 2.
        min_score: Minimum score to return a category (else 'general').
    """

    def __init__(
        self,
        taxonomy_path: str | Path | None = None,
        top_n_candidates: int = 5,
        min_score: int = 1,
    ) -> None:
        self.top_n = top_n_candidates
        self.min_score = min_score

        path = Path(taxonomy_path) if taxonomy_path else _DEFAULT_TAXONOMY

        # Load taxonomy
        self.subdomains: List[str] = []
        self._subword_sets: Dict[str, set] = {}
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}

        self._load_taxonomy(path)

    def _load_taxonomy(self, path: Path) -> None:
        """Load and compile all patterns."""
        with open(path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                name = entry["subdomain"]
                self.subdomains.append(name)

                # Tier 1: subword hash set (lowercased)
                subwords = entry.get("subwords", [])
                self._subword_sets[name] = {w.lower() for w in subwords}

                # Tier 2: compiled regex patterns
                patterns = entry.get("patterns", [])
                compiled = []
                for pat in patterns:
                    try:
                        compiled.append(re.compile(pat))
                    except re.error:
                        pass
                self._compiled_patterns[name] = compiled

        # Build inverted index: subword → set of subdomains
        self._subword_to_domains: Dict[str, List[str]] = {}
        for name, words in self._subword_sets.items():
            for w in words:
                if w not in self._subword_to_domains:
                    self._subword_to_domains[w] = []
                self._subword_to_domains[w].append(name)

    def categorize(
        self,
        text: str,
        return_all: bool = False,
    ) -> Dict[str, Any]:
        """Categorize a text into the best matching subdomain.

        Args:
            text: Input text.
            return_all: Include all subdomain scores in result.

        Returns:
            Dict with: subdomain, score, confidence.
            If return_all: also 'all_scores' dict.
        """
        if not text or len(text) < 5:
            return {"subdomain": "general", "score": 0, "confidence": 0.0}

        # ── Tier 1: Fast subword matching ────────────────────────────────
        text_lower = text.lower()
        # Tokenize: split on non-alphanumeric, keep meaningful chunks
        tokens = set(re.findall(r'[a-z]+(?:_[a-z]+)*', text_lower))
        # Also check bigrams for multi-word subwords
        words = text_lower.split()
        for i in range(len(words) - 1):
            tokens.add(f"{words[i]} {words[i+1]}")

        # Score candidates via inverted index
        candidate_scores: Dict[str, int] = {}
        for token in tokens:
            if token in self._subword_to_domains:
                for domain in self._subword_to_domains[token]:
                    candidate_scores[domain] = candidate_scores.get(domain, 0) + 1

        if not candidate_scores:
            return {"subdomain": "general", "score": 0, "confidence": 0.0}

        # ── Tier 2: Regex on top candidates ──────────────────────────────
        top_candidates = sorted(candidate_scores.items(),
                                key=lambda x: -x[1])[:self.top_n]

        regex_scores: Dict[str, int] = {}
        for name, fast_score in top_candidates:
            patterns = self._compiled_patterns.get(name, [])
            count = 0
            for pat in patterns:
                matches = pat.findall(text)
                count += len(matches)
            # Combined score: fast + regex (regex weighted higher)
            regex_scores[name] = fast_score + count * 2

        if not regex_scores:
            best_name = top_candidates[0][0]
            best_score = top_candidates[0][1]
        else:
            best_name = max(regex_scores, key=regex_scores.get)
            best_score = regex_scores[best_name]

        if best_score < self.min_score:
            best_name = "general"

        # Confidence: normalized score
        max_possible = max(best_score, 1)
        confidence = min(1.0, best_score / 20.0)  # 20 matches = full confidence

        result = {
            "subdomain": best_name,
            "score": best_score,
            "confidence": confidence,
        }

        if return_all:
            # Merge fast + regex scores
            all_scores = dict(candidate_scores)
            for k, v in regex_scores.items():
                all_scores[k] = v
            result["all_scores"] = dict(sorted(
                all_scores.items(), key=lambda x: -x[1])[:20])

        return result

    def categorize_batch(
        self,
        texts: List[str],
    ) -> List[Dict[str, Any]]:
        """Categorize a batch of texts."""
        return [self.categorize(t) for t in texts]

    @property
    def n_subdomains(self) -> int:
        return len(self.subdomains)

    @property
    def n_patterns(self) -> int:
        return sum(len(v) for v in self._compiled_patterns.values())

    @property
    def n_subwords(self) -> int:
        return sum(len(v) for v in self._subword_sets.values())
