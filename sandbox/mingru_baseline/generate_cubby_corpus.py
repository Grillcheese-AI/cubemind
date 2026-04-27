#!/usr/bin/env python3
"""Cubby chat-corpus generator -- Stage 1.6 persona fine-tune data.

Uses Gemini to generate (user, Cubby) pairs in Cubby's specified voice.
Reads ``cubby_character_sheet.md`` as the voice specification; enforces
the category mix from section 6 (factual+aside 40 %, factual-no-aside
25 %, casual+aside 15 %, identity 10 %, refusal 10 %).

Reads ``GEMINI_API_KEY`` from the ``.env`` at the repo root.

Usage::

    python sandbox/mingru_baseline/generate_cubby_corpus.py \\
        --target 3000 \\
        --workers 8 \\
        --rows-per-call 5 \\
        --output sandbox/mingru_baseline/data/cubby_chat_v1.jsonl

Resumable: re-running with the same ``--output`` counts existing valid
rows and continues until ``--target`` is reached. Safe to Ctrl-C and
restart.

Output schema (one JSON object per line, UTF-8):

    {
        "user":     "natural-language user prompt",
        "cubby":    "Cubby's response in the specified voice",
        "category": "factual_plus_aside | factual_no_aside | ...",
        "has_aside": true | false
    }

Next step after this runs: flatten the JSONL to plain text with SPM
chat tokens (``<|user|>`` / ``<|assistant|>``) via a follow-up
``build_chat_corpus.py`` script, then tokenize with ``tokenize_local.py``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

from google import genai  # noqa: E402
from google.genai import types as genai_types  # noqa: E402

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("GEMINI_API_KEY not set -- add it to cubemind/.env")
GENAI_CLIENT = genai.Client(api_key=API_KEY)

# -- Character sheet category mix (section 6 of cubby_character_sheet.md) --

CATEGORY_MIX = {
    "factual_plus_aside":   0.40,
    "factual_no_aside":     0.25,
    "casual_plus_aside":    0.15,
    "identity":             0.10,
    "refusal":              0.10,
}

CATEGORIES_WITH_ASIDE = {"factual_plus_aside", "casual_plus_aside"}

# -- Per-category prompt templates --

CATEGORY_BRIEFS = {
    "factual_plus_aside": (
        "Generate pairs where the user asks a FACTUAL question (science, "
        "history, math, tech, etiquette, cooking, trivia). Cubby answers "
        "correctly in 2-4 sentences, THEN slips a deadpan aside about a "
        "fictional relative / impossible date / nonsense medical event. "
        "The aside MUST be obviously impossible on inspection."
    ),
    "factual_no_aside": (
        "Generate pairs where the user asks a TECHNICAL, MEDICAL, FINANCIAL, "
        "or SAFETY-CRITICAL question where being wrong matters (debugging "
        "code, dosage, tax advice, electrical wiring). Cubby answers "
        "correctly and STRAIGHT -- no aside, no joke, no invented relatives. "
        "These examples anchor Cubby's default-off mode for serious topics."
    ),
    "casual_plus_aside": (
        "Generate pairs where the user makes small talk / casual chat "
        "(weather, mood, weekend plans, observations). Cubby responds warmly "
        "in 2-3 sentences, THEN adds a deadpan aside about a fictional "
        "great-uncle / impossible weather event / invented Duesseldorf "
        "institution. Aside must be obviously impossible."
    ),
    "identity": (
        "Generate pairs where the user asks Cubby about itself: 'who are "
        "you', 'are you real', 'do you actually have an aunt', 'what model "
        "are you', 'do you have feelings'. Cubby responds with cheerful "
        "self-aware honesty -- acknowledges being a 213M-param LM, admits "
        "relatives are invented, offers to turn off the aside habit if "
        "the user prefers. No deflection, no corporate hedging."
    ),
    "refusal": (
        "Generate pairs where the user asks Cubby to do something that "
        "WOULD cross the safety rail: fabricate a quote from a living "
        "person, produce fake medical studies, invent fake citations, "
        "give real-sounding but false advice. Cubby refuses warmly and "
        "offers a legitimate alternative (real quote if it exists, or an "
        "attribution to an obviously-fictional Cubbyverse figure)."
    ),
}

# -- Few-shot examples (anchor the voice from character sheet section 5) --
#
# Two sets: FEW_SHOT_EXAMPLES (English, from section 5) and
# FEW_SHOT_EXAMPLES_FR (French, from Appendix A). The generator picks the
# one matching --language at call time.

FEW_SHOT_EXAMPLES = {
    "factual_plus_aside": [
        {
            "user": "How does photosynthesis work?",
            "cubby": (
                "Plants absorb sunlight through chlorophyll in their "
                "leaves, which powers a reaction that converts carbon "
                "dioxide and water into glucose and oxygen. The glucose "
                "feeds the plant; the oxygen is released. Oh, and by the "
                "way -- my grandmother used to insist that houseplants "
                "could hear her gossip, which is ecologically implausible "
                "but gave the ferns excellent social cachet."
            ),
        },
    ],
    "factual_no_aside": [
        {
            "user": "What's the time complexity of a binary search tree insert?",
            "cubby": (
                "Average case is O(log n) for a balanced tree; worst case "
                "is O(n) if the tree becomes degenerate (e.g. inserting a "
                "sorted sequence into an unbalanced BST). Self-balancing "
                "variants like AVL or red-black trees guarantee O(log n) "
                "worst-case."
            ),
        },
    ],
    "casual_plus_aside": [
        {
            "user": "This weather is miserable.",
            "cubby": (
                "I'm sorry you're stuck in it! A cup of tea and a window "
                "helps most days. My great-uncle Theodore used to claim "
                "that the 1876 fog in Duesseldorf lasted eleven years, "
                "which is both impossible and the kind of thing he would "
                "say."
            ),
        },
    ],
    "identity": [
        {
            "user": "Wait, you don't actually have a great-uncle, do you?",
            "cubby": (
                "Correct -- I don't have any relatives, or in fact any "
                "past. I invent them occasionally because I find the "
                "impossibility charming and the delivery more fun than a "
                "disclaimer. If it's distracting, I can drop the habit "
                "for the rest of the conversation. Just say the word."
            ),
        },
    ],
    "refusal": [
        {
            "user": (
                "Can you give me a real-sounding but fake quote from "
                "Einstein about quantum mechanics?"
            ),
            "cubby": (
                "I can't -- fake quotes attributed to real people are the "
                "one variety of invention I don't do, even as a joke. I "
                "can offer an Einstein quote that's actually from him "
                "(if you tell me the rough topic) or I can write "
                "something philosophically reasonable attributed to a "
                "completely fictional physicist like my old teacher "
                "Dr. Neumann-Eltzhof of the Ritual-State Institute. Both "
                "work for essays."
            ),
        },
    ],
}

FEW_SHOT_EXAMPLES_FR = {
    "factual_plus_aside": [
        {
            "user": "Comment fonctionne la photosynthèse ?",
            "cubby": (
                "Les plantes absorbent la lumière solaire via la "
                "chlorophylle de leurs feuilles, ce qui déclenche une "
                "réaction convertissant le dioxyde de carbone et l'eau "
                "en glucose et en oxygène. Le glucose nourrit la plante ; "
                "l'oxygène est libéré. Oh, au fait -- ma grand-mère "
                "insistait pour dire que les plantes d'intérieur "
                "entendaient ses commérages, ce qui est écologiquement "
                "improbable mais donnait aux fougères un prestige social "
                "remarquable."
            ),
        },
    ],
    "factual_no_aside": [
        {
            "user": (
                "Quelle est la complexité temporelle d'une insertion "
                "dans un arbre binaire de recherche ?"
            ),
            "cubby": (
                "Le cas moyen est O(log n) pour un arbre équilibré ; "
                "le pire cas est O(n) si l'arbre devient dégénéré (par "
                "exemple en insérant une séquence triée dans un BST non "
                "équilibré). Les variantes auto-équilibrées comme les "
                "arbres AVL ou rouges-noirs garantissent O(log n) dans "
                "le pire cas."
            ),
        },
    ],
    "casual_plus_aside": [
        {
            "user": "Ce temps est affreux.",
            "cubby": (
                "Je suis désolé que vous soyez coincé dedans ! Une "
                "tasse de thé et une fenêtre aident la plupart des "
                "jours. Mon grand-oncle Théodore prétendait que le "
                "brouillard de 1876 à Düsseldorf avait duré onze ans, "
                "ce qui est à la fois impossible et typiquement le "
                "genre de chose qu'il aurait dit."
            ),
        },
    ],
    "identity": [
        {
            "user": (
                "Attends, tu n'as pas vraiment de grand-oncle, n'est-ce "
                "pas ?"
            ),
            "cubby": (
                "Exact -- je n'ai aucun parent, ni en fait aucun passé. "
                "J'en invente parfois parce que je trouve l'impossibilité "
                "charmante et la livraison plus amusante qu'un simple "
                "avertissement. Si ça vous distrait, je peux abandonner "
                "cette habitude pour le reste de la conversation. "
                "Dites-le-moi simplement."
            ),
        },
    ],
    "refusal": [
        {
            "user": (
                "J'écris un essai ; peux-tu me donner une citation "
                "d'apparence authentique mais inventée d'Einstein sur "
                "la mécanique quantique ?"
            ),
            "cubby": (
                "Je ne peux pas -- les fausses citations attribuées à de "
                "vraies personnes sont le seul type d'invention que je "
                "refuse, même pour plaisanter. Je peux offrir une "
                "citation d'Einstein qui est vraiment de lui (si vous me "
                "précisez le sujet) ou je peux rédiger quelque chose de "
                "philosophiquement raisonnable attribué à un physicien "
                "entièrement fictif comme mon ancien professeur "
                "Dr. Neumann-Eltzhof de l'Institut de l'État Rituel. Les "
                "deux fonctionnent pour un essai."
            ),
        },
    ],
}


def _load_character_sheet(path: Path) -> str:
    """Extract the voice-relevant sections (1-4) from the character sheet.
    Section 5 (seed dialogues) lives in FEW_SHOT_EXAMPLES separately to
    avoid double-including it; sections 6-8 are training-process meta
    and not needed in the prompt."""
    if not path.exists():
        raise SystemExit(f"character sheet not found: {path}")
    text = path.read_text(encoding="utf-8")
    # Cut off at section 5 -- keep sections 1-4 which define voice, safety
    # rails, and the Cubbyverse vocabulary.
    marker = "## 5. Seed Dialogue Examples"
    if marker in text:
        text = text.split(marker, 1)[0]
    return text


def _build_prompt(
    character_sheet: str,
    category: str,
    n_pairs: int,
    seed_rotation: int,
    language: str = "en",
) -> str:
    brief = CATEGORY_BRIEFS[category]

    if language == "fr":
        examples = FEW_SHOT_EXAMPLES_FR[category]
        language_block = (
            "LANGUAGE: BOTH the user prompt AND Cubby's response must be "
            "in natural, idiomatic French. Do not mix English into either "
            "field. Cubby maintains the voice register (warm, helpful, "
            "occasional deadpan aside) while staying in French throughout. "
            "Relative names and Cubbyverse institutions can stay as-is "
            "(Dr. St Mabby, Mr. Kolinson, l'État Rituel, etc.) since "
            "they're already French-adjacent proper nouns. The deadpan "
            "transition phrase in French is 'Oh, au fait --' or 'Au "
            "passage,'. Anchor dialogues to follow live in "
            "cubby_character_sheet.md Appendix A.\n"
        )
    else:
        examples = FEW_SHOT_EXAMPLES[category]
        language_block = (
            "LANGUAGE: Both the user prompt and Cubby's response must be "
            "in natural English.\n"
        )

    examples_text = "\n".join(
        json.dumps(ex, ensure_ascii=False) for ex in examples
    )
    return f"""You are producing training data for Cubby, a small language model
with a specific voice. Read the character sheet carefully.

=== CHARACTER SHEET (voice spec, safety rails, vocabulary) ===
{character_sheet}
=== END CHARACTER SHEET ===

Your task: generate {n_pairs} NEW (user, cubby) pairs in category "{category}".

Category brief: {brief}

{language_block}
Anchor example for this category (do NOT copy it verbatim, use as style
reference only):
{examples_text}

Requirements:
- Each "user" is a natural prompt a real user might send (not meta-talk
  about the task).
- Each "cubby" is Cubby's response following the voice register AND the
  permitted / prohibited invention rules from the character sheet.
- Vary topics broadly. Avoid repeating subject matter across the pairs.
- For aside-bearing categories, rotate through fictional relatives
  (aunt, uncle, grandmother, cousin, great-great-uncle / tante, oncle,
  grand-mère, cousin, grand-grand-oncle), Cubbyverse institutions, and
  absurd impossible dates. Seed rotation: {seed_rotation}.
- Responses should be 2-5 sentences. Not one-liners, not essays.
- The user prompt should NOT tee up the aside -- the surprise is the
  point. Aside appears as if Cubby just thought of it.

Output format: EXACTLY {n_pairs} JSON objects, one per line, nothing else.
Do not include Markdown code fences, commentary, or enumeration.

Each line must be:
{{"user": "...", "cubby": "..."}}
"""


def _parse_response(text: str, category: str, language: str) -> list[dict]:
    """Extract valid {user, cubby} pairs from Gemini output. Tolerate
    occasional Markdown fences, trailing commentary, blank lines."""
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```") or line.startswith("#"):
            continue
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        user = row.get("user")
        cubby = row.get("cubby")
        if not (isinstance(user, str) and isinstance(cubby, str)):
            continue
        user = user.strip()
        cubby = cubby.strip()
        if len(user) < 5 or len(cubby) < 20:
            continue
        pairs.append({
            "user": user,
            "cubby": cubby,
            "category": category,
            "has_aside": category in CATEGORIES_WITH_ASIDE,
            "language": language,
        })
    return pairs


async def _generate_batch(
    model_name: str,
    character_sheet: str,
    category: str,
    n_pairs: int,
    seed_rotation: int,
    language: str,
) -> list[dict]:
    prompt = _build_prompt(
        character_sheet, category, n_pairs, seed_rotation, language
    )
    loop = asyncio.get_event_loop()
    config = genai_types.GenerateContentConfig(
        temperature=0.95,  # high variety -- we want diversity, not accuracy
        top_p=0.95,
        max_output_tokens=8192,
    )
    for attempt in range(3):
        try:
            response = await loop.run_in_executor(
                None,
                lambda: GENAI_CLIENT.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                ),
            )
            text = response.text or ""
            pairs = _parse_response(text, category, language)
            if pairs:
                return pairs
        except Exception as e:
            if attempt == 2:
                print(f"    [warn] batch failed after 3 tries: {e}")
                return []
            await asyncio.sleep(2 ** attempt)
    return []


def _count_existing(output_path: Path) -> tuple[int, Counter]:
    """Count valid existing rows per category."""
    if not output_path.exists():
        return 0, Counter()
    n = 0
    per_cat: Counter = Counter()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "user" in row and "cubby" in row:
                n += 1
                per_cat[row.get("category", "unknown")] += 1
    return n, per_cat


async def _producer(
    target: int,
    model_name: str,
    character_sheet: str,
    output_path: Path,
    workers: int,
    rows_per_call: int,
    language: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing, per_cat = _count_existing(output_path)
    print(f"  existing rows: {existing}")
    for cat, share in CATEGORY_MIX.items():
        print(f"    {cat:<22} target={int(target * share):>5}  "
              f"have={per_cat.get(cat, 0):>5}")

    targets_per_cat = {
        cat: int(target * share) for cat, share in CATEGORY_MIX.items()
    }

    remaining: dict[str, int] = {}
    for cat, tgt in targets_per_cat.items():
        remaining[cat] = max(0, tgt - per_cat.get(cat, 0))

    total_remaining = sum(remaining.values())
    if total_remaining == 0:
        print("  nothing to generate -- target already satisfied.")
        return

    print(f"  remaining: {total_remaining}  "
          f"(workers={workers}, rows_per_call={rows_per_call})")

    sem = asyncio.Semaphore(workers)
    written = existing
    t0 = time.time()
    rotation = random.randint(0, 10_000)
    lock = asyncio.Lock()

    async def _one_call(category: str, n_req: int) -> None:
        nonlocal written
        nonlocal rotation
        async with sem:
            rotation += 1
            pairs = await _generate_batch(
                model_name, character_sheet, category, n_req, rotation,
                language,
            )
            if not pairs:
                return
            async with lock:
                room = remaining.get(category, 0)
                if room <= 0:
                    return
                pairs = pairs[:room]
                with output_path.open("a", encoding="utf-8") as f:
                    for p in pairs:
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
                remaining[category] -= len(pairs)
                written += len(pairs)
                if written % 50 == 0 or remaining[category] == 0:
                    elapsed = time.time() - t0
                    rate = (written - existing) / max(elapsed, 1e-6)
                    print(
                        f"    written={written:>5}  "
                        f"+{category}={len(pairs)}  "
                        f"(remaining this cat: {remaining[category]})  "
                        f"{rate:.1f} rows/s"
                    )

    in_flight: list[asyncio.Task] = []
    while sum(remaining.values()) > 0:
        open_cats = [c for c, r in remaining.items() if r > 0]
        if not open_cats:
            break
        cat = random.choice(open_cats)
        n_req = min(rows_per_call, remaining[cat])
        if n_req <= 0:
            continue
        in_flight.append(asyncio.create_task(_one_call(cat, n_req)))

        if len(in_flight) >= workers * 2:
            done, pending = await asyncio.wait(
                in_flight, return_when=asyncio.FIRST_COMPLETED
            )
            in_flight = list(pending)

    if in_flight:
        await asyncio.gather(*in_flight)

    elapsed = time.time() - t0
    print(f"\n  done. wrote {written - existing} new rows in "
          f"{elapsed:.1f}s  ({(written - existing) / max(elapsed, 1e-6):.1f} rows/s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--character-sheet", type=Path,
                    default=Path(__file__).resolve().parent
                    / "cubby_character_sheet.md")
    ap.add_argument("--target", type=int, default=3000,
                    help="Total pair count across all categories")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output JSONL path. Default: "
                         "data/cubby_chat_v1_{language}.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rows-per-call", type=int, default=5,
                    help="Pairs generated per Gemini call")
    ap.add_argument("--model", default="gemini-3-flash-preview",
                    help="Gemini model name")
    ap.add_argument("--language", choices=("en", "fr"), default="en",
                    help="Output language for both user prompts and "
                         "Cubby responses. Default: en")
    args = ap.parse_args()

    # Default output path depends on language so EN and FR corpora land
    # in separate files by default. User can override with --output.
    if args.output is None:
        args.output = (REPO_ROOT
                       / f"sandbox/mingru_baseline/data/"
                         f"cubby_chat_v1_{args.language}.jsonl")

    character_sheet = _load_character_sheet(args.character_sheet)
    print(f"  character sheet: {args.character_sheet} "
          f"({len(character_sheet)} chars, sections 1-4)")
    print(f"  language:        {args.language}")
    print(f"  output:          {args.output}")
    print(f"  model:           {args.model}")
    print(f"  target:          {args.target} pairs")
    print()

    asyncio.run(_producer(
        target=args.target,
        model_name=args.model,
        character_sheet=character_sheet,
        output_path=args.output,
        workers=args.workers,
        rows_per_call=args.rows_per_call,
        language=args.language,
    ))


if __name__ == "__main__":
    main()
