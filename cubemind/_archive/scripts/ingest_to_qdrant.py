"""Ingest pretrain_unified.jsonl into Qdrant with fast regex categorization.

Pipeline:
  1. Stream lines from JSONL
  2. Regex categorize each text chunk (matching existing Qdrant categories)
  3. Embed with Harrier-OSS-v1-0.6b (384-dim, batched)
  4. Push to Qdrant in batches

Usage:
    python scripts/ingest_to_qdrant.py --input D:/grillcheese_training_data/unified/pretrain_unified.with_hf.jsonl
    python scripts/ingest_to_qdrant.py --input data.jsonl --max-records 100000
    python scripts/ingest_to_qdrant.py --input data.jsonl --batch-size 512 --collection corpus
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from typing import Optional

import numpy as np

# ── Fast Regex Categorizer ───────────────────────────────────────────────────

# Categories matching existing Qdrant corpus
CATEGORY_PATTERNS = {
    "physics": r"\b(quantum|relativity|photon|electron|neutron|proton|energy|thermodynamic|entropy|wave|particle|magnetic|electric|gravity|mass|velocity|acceleration|force|momentum)\b",
    "chemistry": r"\b(molecule|atom|chemical|reaction|compound|element|periodic|ion|acid|base|solution|catalyst|oxidation|bond|organic|polymer)\b",
    "biology": r"\b(cell|gene|dna|rna|protein|organism|species|evolution|mutation|bacteria|virus|enzyme|membrane|mitosis|ecology|photosynthesis|neuron|synapse)\b",
    "medicine": r"\b(disease|treatment|patient|clinical|diagnosis|symptom|therapy|pharmaceutical|vaccine|infection|surgery|chronic|tumor|cancer|antibiot)\b",
    "mathematics": r"\b(equation|theorem|proof|algorithm|matrix|vector|calculus|integral|derivative|probability|statistics|topology|geometry|algebra)\b",
    "computer science": r"\b(algorithm|software|programming|database|network|cpu|gpu|compiler|machine learning|neural network|artificial intelligence|api|protocol|encryption|binary)\b",
    "astronomy": r"\b(planet|star|galaxy|universe|cosmic|solar|lunar|orbit|telescope|nebula|supernova|constellation|asteroid|comet|blackhole|pulsar)\b",
    "geology": r"\b(rock|mineral|tectonic|earthquake|volcano|sediment|fossil|crystal|erosion|magma|geolog|stratigraphy|continental)\b",
    "ecology": r"\b(ecosystem|biodiversity|habitat|conservation|climate change|pollution|deforestation|extinction|sustainability|carbon|greenhouse)\b",
    "psychology": r"\b(cognitive|behavior|emotion|personality|consciousness|perception|memory|learning|motivation|anxiety|depression|therapy|mental)\b",
    "philosophy": r"\b(ethics|morality|epistemology|metaphysics|ontology|logic|existential|consciousness|free will|determinism|utilitarianism|stoic)\b",
    "history": r"\b(century|empire|dynasty|revolution|war|treaty|colony|civilization|ancient|medieval|renaissance|reform|independence|conquest)\b",
    "economics": r"\b(market|inflation|gdp|trade|fiscal|monetary|supply|demand|capital|investment|recession|taxation|labor|wealth|poverty)\b",
    "finance": r"\b(stock|bond|interest rate|portfolio|asset|liability|dividend|equity|hedge|derivative|banking|mortgage|credit|debt)\b",
    "literature": r"\b(novel|poem|author|narrative|fiction|literary|prose|verse|metaphor|character|genre|tragedy|comedy|epic|sonnet)\b",
    "linguistics": r"\b(language|grammar|syntax|phoneme|morpheme|semantic|pragmatic|dialect|vowel|consonant|syllable|etymology|lexicon)\b",
    "music": r"\b(melody|harmony|rhythm|tempo|chord|symphony|orchestra|composer|sonata|opera|jazz|baroque|instrument|pitch|scale)\b",
    "art": r"\b(painting|sculpture|canvas|brush|pigment|gallery|museum|impressionism|cubism|renaissance art|portrait|landscape|abstract)\b",
    "architecture": r"\b(building|architect|cathedral|dome|column|facade|blueprint|gothic|baroque|construction|foundation|structural)\b",
    "religion": r"\b(god|divine|worship|prayer|sacred|church|temple|mosque|scripture|faith|prophet|salvation|ritual|spiritual|soul)\b",
    "sociology": r"\b(society|social|community|class|inequality|norm|institution|culture|identity|gender|race|demographic|urbanization)\b",
    "political science": r"\b(government|democracy|election|policy|legislation|congress|parliament|sovereignty|ideology|republic|constitutional|diplomatic)\b",
    "legal": r"\b(law|court|judge|statute|regulation|contract|liability|criminal|civil|plaintiff|defendant|jurisdiction|constitution|amendment)\b",
    "technology": r"\b(device|hardware|software|digital|internet|wireless|robotics|sensor|automation|3d print|nanotechnology|semiconductor|circuit)\b",
    "earth_science": r"\b(atmosphere|weather|climate|ocean|glacier|hydrology|meteorolog|soil|water cycle|ozone|precipitation)\b",
    "genetics": r"\b(genome|chromosome|allele|phenotype|genotype|heredit|crispr|sequencing|transcription|genetic engineer)\b",
    "biochemistry": r"\b(amino acid|lipid|carbohydrate|metabolism|atp|enzyme kinetic|nucleotide|ribosome|mitochondr)\b",
    "paleontology": r"\b(dinosaur|fossil|extinction|prehistoric|jurassic|cretaceous|triassic|paleozoic|mammoth|trilobite)\b",
    "navigation": r"\b(compass|latitude|longitude|nautical|maritime|voyage|sail|port|harbor|navigation|seafar)\b",
    "sports": r"\b(game|team|player|score|championship|tournament|athlete|coach|stadium|match|competition|olympic)\b",
    "fashion": r"\b(clothing|textile|designer|fabric|garment|trend|couture|accessory|wardrobe|silk|cotton|leather)\b",
    "journalism": r"\b(newspaper|reporter|editor|headline|press|media|broadcast|correspondent|publication|interview)\b",
    "geography": r"\b(continent|country|region|border|mountain|river|island|desert|forest|population|territory|map)\b",
    "nature": r"\b(tree|flower|animal|bird|fish|insect|forest|ocean|river|mountain|wildlife|plant|garden|ecosystem)\b",
    "culture": r"\b(tradition|festival|custom|heritage|folklore|ritual|ceremony|celebration|dance|cuisine|craft)\b",
    "health": r"\b(nutrition|exercise|diet|fitness|wellness|vitamin|obesity|heart|blood pressure|diabetes|cholesterol)\b",
    "reference": r"\b(definition|glossary|encyclopedia|dictionary|manual|handbook|guide|index|appendix|bibliography)\b",
}

_compiled_patterns = {cat: re.compile(pat, re.IGNORECASE) for cat, pat in CATEGORY_PATTERNS.items()}


def categorize_text(text: str, categorizer=None) -> dict:
    """Categorize text using taxonomy categorizer or simple regex fallback.

    Returns dict with domain, subdomain, realm, quality_score.
    """
    if categorizer is not None:
        result = categorizer.categorize(text, return_all=False)
        subdomain = result["subdomain"]
        confidence = result["confidence"]

        # Map subdomain -> domain using realm hierarchy
        domain = _subdomain_to_domain.get(subdomain, "general")

        return {
            "domain": domain,
            "subdomain": subdomain,
            "realm": f"{domain}/{subdomain}" if domain != subdomain else domain,
            "quality_score": confidence,
        }

    # Fallback: simple regex
    if not text or len(text) < 10:
        return {"domain": "general", "subdomain": "general",
                "realm": "general", "quality_score": 0.0}

    scores = {}
    text_lower = text.lower()
    for cat, pattern in _compiled_patterns.items():
        matches = pattern.findall(text_lower)
        if matches:
            scores[cat] = len(matches)

    if not scores:
        return {"domain": "general", "subdomain": "general",
                "realm": "general", "quality_score": 0.0}

    best = max(scores, key=scores.get)
    return {"domain": best, "subdomain": best,
            "realm": best, "quality_score": min(1.0, scores[best] / 10.0)}


# Subdomain -> domain mapping (built from realm hierarchy)
_subdomain_to_domain = {
    "physics": "science", "chemistry": "science", "biology": "science",
    "astronomy": "science", "geology": "science", "ecology": "science",
    "genetics": "science", "medicine": "science", "neuroscience": "science",
    "biochemistry": "science", "anatomy": "science", "virology": "science",
    "immunology": "science", "paleontology": "science", "cosmology": "science",
    "meteorology": "science", "oceanography": "science", "climate": "science",
    "earth_science": "science", "environmental": "science",
    "ai": "technology", "programming": "technology", "software": "technology",
    "computing": "technology", "robotics": "technology", "blockchain": "technology",
    "cybersecurity": "technology", "data_science": "technology", "iot": "technology",
    "quantum_computing": "technology", "aerospace": "technology",
    "electronics": "technology", "engineering": "technology", "webdev": "technology",
    "art": "culture", "music": "culture", "film": "culture", "dance": "culture",
    "literature": "culture", "fashion": "culture", "design": "culture",
    "folklore": "culture", "mythology": "culture", "food": "culture",
    "performing_arts": "culture", "animation": "culture", "television": "culture",
    "architecture": "culture", "sports": "culture",
    "finance": "economics", "markets": "economics", "investing": "economics",
    "banking": "economics", "accounting": "economics", "cryptocurrency": "economics",
    "macroeconomics": "economics", "microeconomics": "economics",
    "ancient": "history", "medieval": "history", "modern": "history",
    "european": "history", "american": "history", "ancient_rome": "history",
    "ancient_greece": "history", "ancient_egypt": "history", "renaissance": "history",
    "empire": "history", "colonialism": "history", "coldwar": "history",
    "linguistics": "humanities", "philosophy": "humanities", "religion": "humanities",
    "sociology": "humanities", "political_science": "humanities",
    "psychology": "humanities",
    "geography": "world", "navigation": "world",
}


# ── Embedder ─────────────────────────────────────────────────────────────────

class GGUFEmbedder:
    """GGUF-based embedder using llama-cpp-python. Much faster than sentence-transformers on GPU."""

    def __init__(self, model_path: str, n_gpu_layers: int = -1):
        from llama_cpp import Llama
        self._model = Llama(
            model_path=model_path, embedding=True,
            n_ctx=512, n_gpu_layers=n_gpu_layers, verbose=False,
        )

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True, batch_size=64):
        embeddings = []
        for text in texts:
            emb = self._model.embed(text)
            vec = np.array(emb, dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            embeddings.append(vec)
        return np.stack(embeddings)


def get_embedder(model_name: str = "microsoft/harrier-oss-v1-0.6b"):
    """Load embedder — GGUF first, then sentence-transformers fallback."""
    # Try GGUF path first
    gguf_path = "data/external_llms/harrier-oss-v1-0.6b.Q8_0.gguf"
    try:
        from pathlib import Path
        if Path(gguf_path).exists():
            print(f"Using GGUF embedder: {gguf_path}")
            return GGUFEmbedder(gguf_path)
    except Exception as e:
        print(f"GGUF embedder failed: {e}")

    # Fallback to sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Using sentence-transformers: {model_name}")
        return SentenceTransformer(model_name, trust_remote_code=True)
    except ImportError:
        print("pip install sentence-transformers OR llama-cpp-python")
        return None


# ── Qdrant Client ────────────────────────────────────────────────────────────

def get_qdrant_client(url: str = "http://localhost:6333"):
    try:
        from qdrant_client import QdrantClient
        return QdrantClient(url=url)
    except ImportError:
        print("pip install qdrant-client")
        return None


# ── Main Pipeline ────────────────────────────────────────────────────────────

def ingest(
    input_path: str,
    collection: str = "corpus",
    qdrant_url: str = "http://localhost:6333",
    batch_size: int = 256,
    max_records: Optional[int] = None,
    min_text_length: int = 20,
    embedder_name: str = "microsoft/harrier-oss-v1-0.6b",
    skip_existing: int = 0,
):
    print(f"Ingesting {input_path} -> Qdrant {collection}")

    embedder = get_embedder(embedder_name)
    if embedder is None:
        return

    client = get_qdrant_client(qdrant_url)
    if client is None:
        return

    # Load taxonomy categorizer
    categorizer = None
    try:
        from cubemind.perception.categorizer import Categorizer
        categorizer = Categorizer()
        print(f"Categorizer loaded: {categorizer.n_subdomains} subdomains, "
              f"{categorizer.n_patterns:,} patterns")
    except Exception as e:
        print(f"Categorizer unavailable ({e}), using regex fallback")

    # Check collection exists
    try:
        info = client.get_collection(collection)
        print(f"Collection '{collection}': {info.points_count} existing points")
    except Exception:
        print(f"Creating collection '{collection}' (384-dim, cosine)")
        from qdrant_client.models import Distance, VectorParams
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    t0 = time.time()
    total = 0
    skipped = 0
    batch_texts = []
    batch_meta = []
    batch_ids = []
    domain_counts = {}

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f):
            if line_num < skip_existing:
                continue
            if max_records and total >= max_records:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            text = obj.get("text", "")
            if len(text) < min_text_length:
                skipped += 1
                continue

            text_for_embed = text[:512]
            meta = categorize_text(text_for_embed, categorizer)
            meta["source"] = obj.get("source_path", obj.get("source", ""))
            meta["chunk_index"] = obj.get("chunk_index", 0)

            batch_texts.append(text_for_embed)
            batch_meta.append(meta)
            batch_ids.append(str(uuid.uuid4()))
            domain_counts[meta["realm"]] = domain_counts.get(meta["realm"], 0) + 1
            total += 1

            if len(batch_texts) >= batch_size:
                _push_batch(client, embedder, collection,
                           batch_texts, batch_meta, batch_ids)
                elapsed = time.time() - t0
                rate = total / elapsed
                top3 = sorted(domain_counts.items(), key=lambda x: -x[1])[:3]
                print(f"  [{total:,}] {rate:.0f} rec/s | "
                      f"skip={skipped} | top: {top3}")
                batch_texts.clear()
                batch_meta.clear()
                batch_ids.clear()

    if batch_texts:
        _push_batch(client, embedder, collection,
                   batch_texts, batch_meta, batch_ids)

    elapsed = time.time() - t0
    print(f"\nDone: {total:,} records in {elapsed:.0f}s ({total/elapsed:.0f} rec/s)")
    print(f"Skipped: {skipped:,}")
    print("Categories:")
    for realm, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {realm}: {count:,}")


def _push_batch(client, embedder, collection, texts, meta_list, ids):
    """Embed and push a batch to Qdrant with full realm hierarchy."""
    from qdrant_client.models import PointStruct

    embeddings = embedder.encode(
        texts, show_progress_bar=False,
        normalize_embeddings=True, batch_size=64,
    )

    points = [
        PointStruct(
            id=uid,
            vector=emb.tolist(),
            payload={
                "text": text,
                "domain": meta["domain"],
                "subdomain": meta["subdomain"],
                "realm": meta["realm"],
                "quality_score": meta["quality_score"],
                "source": meta.get("source", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "tokens": len(text.split()),
            },
        )
        for uid, emb, text, meta in zip(ids, embeddings, texts, meta_list)
    ]

    client.upsert(collection_name=collection, points=points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--collection", default="corpus")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--min-text-length", type=int, default=20)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--embedder", default="microsoft/harrier-oss-v1-0.6b")
    args = parser.parse_args()

    ingest(
        input_path=args.input,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        batch_size=args.batch_size,
        max_records=args.max_records,
        min_text_length=args.min_text_length,
        skip_existing=args.skip,
        embedder_name=args.embedder,
    )
