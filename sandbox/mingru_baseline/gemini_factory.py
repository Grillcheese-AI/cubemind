#!/usr/bin/env python3
"""Gemini multitask data factory.

Few-shot the model on the 20-row sample, then ask it to generate N more
rows in the exact same tagged JSONL format so the train_torch.py
multitask path can consume them unchanged.

Reads ``GEMINI_API_KEY`` from the ``.env`` at the repo root.

Usage::

    python sandbox/mingru_baseline/gemini_factory.py \\
        --sample C:/Users/grill/Downloads/mingru_multitask_sample_dataset.jsonl \\
        --target 50000 --workers 8 --rows-per-call 25 \\
        --output sandbox/mingru_baseline/data/multitask_gemini_v1.jsonl

Resumable: re-running with the same ``--output`` continues where it left
off (counts existing valid rows, skips already-done work).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

from google import genai  # noqa: E402
from google.genai import types as genai_types  # noqa: E402

# Local module — sits next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vm_opcodes import (CANONICAL_OPCODES, NUM_OPCODES, OPCODE_TO_ID,  # noqa: E402
                        normalize_opcode)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("GEMINI_API_KEY not set — add it to cubemind/.env")
GENAI_CLIENT = genai.Client(api_key=API_KEY)


# ── Schema enforced on every generated row ──────────────────────────────

REQUIRED_KEYS = {
    "text", "schema_id", "schema_name", "rule_id", "rule_name",
    "opcode_id", "opcode_name", "validity", "cubelang_program",
}

# SCHEMA and VALID are always required; the rule/opcode pair is either
# (<RULE>, <OPCODE>) for SCHEMA2RULE rows OR (<FIX_RULE>, <FIX_OPCODE>)
# for REPAIR rows (where the schema-level opcode sits on the FIX_ tags).
ALWAYS_REQUIRED_TAGS = {"<SCHEMA>", "</SCHEMA>", "<VALID>", "</VALID>"}
RULE_TAGS = ({"<RULE>", "</RULE>", "<OPCODE>", "</OPCODE>"},
             {"<FIX_RULE>", "</FIX_RULE>", "<FIX_OPCODE>", "</FIX_OPCODE>"})


def _has_required_tags(text: str) -> bool:
    if not all(tag in text for tag in ALWAYS_REQUIRED_TAGS):
        return False
    return any(all(tag in text for tag in bucket) for bucket in RULE_TAGS)


class LabelRegistry:
    """Persistent name → ID map for open-vocabulary schema/rule labels.

    Opcodes use the **fixed canonical ID** from ``vm_opcodes`` (must match
    the Rust VM's enum order). Schemas and rules are open vocabulary —
    Gemini invents new ones, and we hand them a stable monotonic ID the
    first time we see them. Re-running the factory rebuilds the registry
    from the existing output JSONL so IDs stay stable across resumes.
    """

    def __init__(self) -> None:
        self.schema: dict[str, int] = {}
        self.rule: dict[str, int] = {}

    def schema_id(self, name: str) -> int:
        if name not in self.schema:
            self.schema[name] = len(self.schema)
        return self.schema[name]

    def rule_id(self, name: str) -> int:
        if name not in self.rule:
            self.rule[name] = len(self.rule)
        return self.rule[name]

    def absorb(self, row: dict) -> None:
        """Seed the registry from an existing row (used when resuming)."""
        if "schema_name" in row and isinstance(row["schema_id"], int):
            # Trust the existing assignment to keep IDs stable across resumes.
            self.schema.setdefault(row["schema_name"], row["schema_id"])
        if "rule_name" in row and isinstance(row["rule_id"], int):
            self.rule.setdefault(row["rule_name"], row["rule_id"])


def normalize_row(row: dict, registry: LabelRegistry) -> dict | None:
    """Validate + rewrite IDs to canonical values.

    - opcode_name folded to canonical form (BINDROLE → BIND_ROLE) and
      opcode_id overwritten to the position-based class ID from
      vm_opcodes. Rows with unknown opcodes get rejected.
    - schema_name / rule_name kept as model produced; schema_id /
      rule_id overwritten to the LabelRegistry's stable assignment.
    - Returns the rewritten row or None if validation fails.
    """
    if not isinstance(row, dict) or not REQUIRED_KEYS.issubset(row.keys()):
        return None
    text = row.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return None
    if not _has_required_tags(text):
        return None
    if row["validity"] not in (0, 1):
        return None

    # cubelang_program — loose sanity check: must be a non-empty string,
    # must contain a ``program`` keyword, must reference at least one
    # canonical opcode mnemonic in lowercase (the .cube syntax).
    prog = row.get("cubelang_program", "")
    if not isinstance(prog, str) or len(prog.strip()) < 20:
        return None
    if "program " not in prog and "program\n" not in prog:
        return None
    canonical_lower = {op.lower() for op in CANONICAL_OPCODES}
    prog_words = set(re.findall(r"[a-z_][a-z_0-9]*", prog.lower()))
    if not (prog_words & canonical_lower):
        return None

    # Canonical opcode — reject unknown opcodes outright (we'd rather
    # drop a row than train on a misaligned label).
    canonical_op = normalize_opcode(str(row.get("opcode_name", "")))
    if canonical_op is None:
        return None
    row["opcode_name"] = canonical_op
    row["opcode_id"] = OPCODE_TO_ID[canonical_op]

    # Open-vocab schema / rule IDs — registry assigns stable monotonic IDs.
    schema_name = str(row.get("schema_name", "")).strip()
    rule_name = str(row.get("rule_name", "")).strip()
    if not schema_name or not rule_name:
        return None
    row["schema_id"] = registry.schema_id(schema_name)
    row["rule_id"] = registry.rule_id(rule_name)
    return row


def is_well_formed(row: dict) -> bool:
    """Quick check — used by the resume scan and the few-shot loader."""
    if not isinstance(row, dict) or not REQUIRED_KEYS.issubset(row.keys()):
        return False
    text = row.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return False
    if not _has_required_tags(text):
        return False
    if row["validity"] not in (0, 1):
        return False
    prog = row.get("cubelang_program", "")
    if not isinstance(prog, str) or len(prog.strip()) < 20:
        return False
    if "program " not in prog and "program\n" not in prog:
        return False
    return normalize_opcode(str(row.get("opcode_name", ""))) is not None


# ── Prompt construction ─────────────────────────────────────────────────

# The model sees the system prompt + a few-shot block + an explicit
# request for N JSON-array rows. We ask for a flat array (no markdown
# fences, no extra prose) so parsing is one ``json.loads`` call.

_OPCODE_LIST_STR = ", ".join(CANONICAL_OPCODES)

# Compact CubeLang syntax the model needs — full spec is docs/SPEC.md in
# the cubelang repo, but this subset is enough to write the small
# single-function programs we ask for per row.
_CUBELANG_CHEATSHEET = """\
CubeLang syntax (compact — enough for a one-function program):

  program <Name> implements <Interface>? {
      type Input = <type>;
      type Output = <type>;

      storage {                            # optional: persistent VM state
          <field>: mutable <type> = <default>;
      }

      @system @once
      public function constructor() { ... }

      @external
      public function solve(input: Input): Output {
          create <reg> : <type>;           # allocate a VM register
          assign <reg> = <val>;            # bind val into reg
          bind_role <reg>, <ROLE>;         # ROLE ∈ AGENT, ACTION, OBJECT,
                                           #        QUANTITY, SOURCE,
                                           #        DESTINATION, CONTEXT, STATE
          store <reg>, "<key>";            # persist reg in cleanup memory
          query <reg>;                     # → integer value
          remember <reg>;                  # mark reg for long-term memory
          add <reg>, <n>;  sub <reg>, <n>;
          transfer <src>, <dst>, <qty>;    # move qty from src to dst
          compare <a>, <b>;                # → "equal" | "less" | "greater"
          discover <input>, <output>;      # induce transformation rule
          predict <seq>;  match <target>, <candidates>;
          seq [<v0>, <v1>, ...];  unseq <seq>, <pos>;
          cond <var>, <target>, { ... } else { ... };
          loop <var>, <target>, <cond>, { ... };
          decode <reg>, <codebook>;  score <reg>, <candidates>;
          return <reg>;
      }

      @external
      public pure function verify(input: Input, output: Output): bool {
          return <expr>;
      }
  }

Primitive types: u8/u16/u32/u64, i8-i64, f32/f64, bool, str, null.
VSA-native types: symbol, quantity, vec, role, opcode, ctx.
Collections: array<T>, map<K,V>, set<T>, tuple<T1,T2,...>.
Opcodes in function bodies use lowercase-with-underscores
(create, bind_role, transfer, discover_sequence, map_roles, ...).

Programs should be SMALL (5–15 lines of body) and demonstrate the
row's schema/rule/opcode mapping concretely — the point is to give
the trainer a working code witness, not a full application.
"""

SYSTEM_PROMPT = f"""\
You generate training data for a neuro-symbolic reasoning model. Each
row pairs a short English or French sentence describing a schema/rule/
opcode mapping with structured labels AND a small CubeLang program
that executes the mapping.

Per row, produce:
  - ``text``: tagged sentence (format below)
  - ``schema_name`` / ``rule_name`` / ``opcode_name``: string labels
  - ``schema_id`` / ``rule_id`` / ``opcode_id``: may be 0 (trainer rewrites)
  - ``validity``: 0 or 1
  - ``cubelang_program``: a SMALL self-contained .cube program (5–15
    body lines) that demonstrates the schema/rule/opcode in executable
    form. The program must include the ``program`` keyword and use at
    least one of the VM's opcode keywords in its body.

The text field uses a fixed tag-bag format. Tag order is flexible but
every required tag must be present:

  <TASK:SCHEMA2RULE> | <TASK:REPAIR>
  <SCHEMA>{{schema_name}}</SCHEMA>
  <ROLES>SOURCE DESTINATION OBJECT QUANTITY</ROLES>     (optional but encouraged)
  <TRACE>OP1 OP2 OP3</TRACE>                            (the opcode prefix observed)
  <RULE>{{rule_name}}</RULE>     for SCHEMA2RULE
  <OPCODE>{{OPCODE_NAME}}</OPCODE>
  <VALID>yes|no</VALID>
  for REPAIR rows: also <ERROR>{{name}}</ERROR> <FIX_RULE>{{name}}</FIX_RULE> <FIX_OPCODE>{{NAME}}</FIX_OPCODE>

Conventions:
  - SCHEMA2RULE rows have validity=1; REPAIR rows have validity=0.
  - Opcode names MUST be drawn from this CANONICAL CubeMind VSA-VM set
    (use the exact spelling, including underscores like BIND_ROLE not
    BINDROLE — the trainer rejects unknown opcodes):

    {_OPCODE_LIST_STR}

  - Schema names are snake_case (e.g. resource_transfer,
    inventory_update, social_memory, temporal_sequence,
    arithmetic_planning, role_binding, dialogue_turn, planning_step,
    knowledge_query, world_state). Invent new schemas freely — the
    trainer maintains its own ID assignment.
  - Rule names are snake_case verbs/predicates describing the action.
  - schema_id, rule_id, opcode_id may be set to 0 — the trainer
    rewrites them to canonical/registry IDs before training. What
    matters is that schema_name, rule_name, opcode_name are correct
    and consistent.

{_CUBELANG_CHEATSHEET}
You will be given a few-shot pool of valid rows and must produce a JSON
ARRAY of additional valid rows. No markdown fences, no commentary —
output only the JSON array, parseable by json.loads. Keep each
cubelang_program short; a docstring-style comment is allowed but not
required.
"""


def build_user_prompt(few_shot: list[dict], n: int,
                      lang_mix: tuple[float, float, float],
                      svc_source_rows: list[dict] | None = None) -> str:
    """Assemble the per-call prompt.

    Two modes:
      - **svc_source_rows is None** (pure synthesis): ask for ``n`` new
        rows grounded only in the few-shot seed.
      - **svc_source_rows given** (annotation): ask for exactly one
        multitask row per input SVC row, using the instruction as the
        natural-language witness and inferring schema/rule/opcode.
    """
    en, fr, mixed = lang_mix
    examples = "\n".join(json.dumps(r, ensure_ascii=False) for r in few_shot)

    if svc_source_rows:
        # Annotation mode — every output row maps back to one input
        # instruction. We pass the SVC structure (s/v/c split + realm +
        # root verb) as hints, but the model is free to pick the right
        # schema/rule/opcode and write the program.
        src_lines = []
        for i, r in enumerate(svc_source_rows):
            svc = r.get("svc") or {}
            src_lines.append(json.dumps({
                "idx": i,
                "text": r.get("text", ""),
                "subject": svc.get("s", ""),
                "verb": svc.get("v", "") or r.get("root_verb", ""),
                "complement": svc.get("c", ""),
                "realm": r.get("realm", ""),
                "complexity": r.get("complexity", None),
            }, ensure_ascii=False))
        sources = "\n".join(src_lines)
        return f"""\
Output-format reminders (one example per line):
{examples}

Now annotate each of these {len(svc_source_rows)} real instructions
with a multitask row. Return a JSON ARRAY of exactly {len(svc_source_rows)}
rows in input order:
{sources}

Per-row rules for annotation mode:
  - The ``text`` field's tagged body describes the instruction's
    operational intent. You may re-use the instruction verbatim inside
    the tag block (e.g. wrap it in <SCHEMA>...</SCHEMA> context), or
    synthesize a short summary — either way, all required tags must
    be present.
  - Infer a ``schema_name`` from ``realm`` + instruction intent
    (snake_case; invent new schemas when the realm's standard ones
    don't fit).
  - Infer a ``rule_name`` that captures the instruction's action
    (snake_case verb phrase).
  - Pick the ``opcode_name`` that best matches the action from the
    canonical opcode set — use exact spelling with underscores.
  - ``validity=1`` is the default; set ``validity=0`` and add REPAIR
    tags only when the instruction itself describes a fix / error-
    handling scenario.
  - ``cubelang_program``: write a small program executing the rule
    on ``input: Input``. Use the instruction's subject/verb/complement
    as naming inspiration.
  - Preserve the instruction's original language (EN or FR).
"""

    # Pure synthesis (fallback when no SVC source is given).
    repair_target = max(1, int(n * 0.35))
    return f"""\
Few-shot pool (do NOT copy verbatim — vary schemas, rules, opcodes,
text wording; invent new schemas/rules where useful):
{examples}

Generate exactly {n} NEW rows as a JSON array. Constraints:
  - Roughly {int(en*100)}% English, {int(fr*100)}% French, {int(mixed*100)}% EN/FR code-switch.
  - Roughly {repair_target} REPAIR rows (validity=0); the rest SCHEMA2RULE.
  - Cover at least 6 distinct schemas in the batch.
  - Cover at least 10 distinct opcodes in the batch.
  - Reuse opcode/schema/rule IDs consistently when names match prior rows.
  - text field must contain ALL required tags listed in the system prompt.
  - cubelang_program: include a small working .cube program per row.
  - Output the JSON array only — no markdown, no prose, parseable by json.loads.
"""


# ── Async worker pool ───────────────────────────────────────────────────

async def _generate_batch(model_name: str, few_shot: list[dict],
                          rows_per_call: int, lang_mix: tuple[float, float, float],
                          registry: LabelRegistry,
                          svc_source_rows: list[dict] | None = None,
                          retries: int = 4) -> list[dict]:
    """One Gemini call → list of canonicalized rows.

    Retries on transient errors. Each surviving row has its opcode_id
    rewritten to the canonical position-based ID and schema_id/rule_id
    rewritten to the LabelRegistry's stable assignment.
    """
    prompt = build_user_prompt(few_shot, rows_per_call, lang_mix,
                               svc_source_rows=svc_source_rows)
    cfg = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=1.0,
        top_p=0.95,
        max_output_tokens=8192,
        response_mime_type="application/json",
    )
    backoff = 2.0
    last_err = None
    for attempt in range(retries):
        try:
            resp = await asyncio.to_thread(
                GENAI_CLIENT.models.generate_content,
                model=model_name,
                contents=prompt,
                config=cfg,
            )
            text = (resp.text or "").strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError(f"expected array, got {type(data).__name__}")
            out: list[dict] = []
            for r in data:
                fixed = normalize_row(r, registry)
                if fixed is not None:
                    out.append(fixed)
            return out
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                await asyncio.sleep(backoff)
                backoff *= 2.0
    print(f"  WARN batch failed after {retries} retries: {last_err}",
          file=sys.stderr)
    return []


def _stream_svc(svc_path: Path, skip: int, limit: int | None):
    """Yield SVC rows from a JSONL file, skipping the first ``skip``
    (already-consumed on resume) and stopping at ``limit`` if given."""
    seen = 0
    with svc_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if seen < skip:
                seen += 1
                continue
            if limit is not None and (seen - skip) >= limit:
                return
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
            seen += 1


async def _producer(workers: int, target_rows: int, model_name: str,
                    sample_rows: list[dict], rows_per_call: int,
                    lang_mix: tuple[float, float, float],
                    output_path: Path, fewshot_pool: int = 8,
                    svc_source: Path | None = None,
                    svc_limit: int | None = None) -> int:
    """Spawn ``workers`` async tasks, collect rows, append to JSONL.

    If ``svc_source`` is given, rows are annotations of real
    instructions streamed from that file; ``target_rows`` then becomes
    an upper bound (capped by either ``target_rows`` or the file size).
    Otherwise, pure synthesis continues until ``target_rows`` is hit.
    """
    # google-genai is stateless; the model name is passed per-call.
    registry = LabelRegistry()
    written = _count_existing(output_path, registry=registry)
    if written:
        print(f"  resume: {written} rows already in {output_path.name} | "
              f"{len(registry.schema)} schemas, {len(registry.rule)} rules")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f = output_path.open("a", encoding="utf-8")
    sem = asyncio.Semaphore(workers)
    stats: Counter = Counter()
    rng = random.Random(0)
    t0 = time.time()

    # SVC source generator — empty iterator when not in annotation mode.
    svc_iter = None
    if svc_source is not None:
        svc_iter = iter(_stream_svc(svc_source, skip=written, limit=svc_limit))
        print(f"  svc source: {svc_source} (skip {written}, limit {svc_limit})")

    def _next_svc_batch(k: int) -> list[dict]:
        if svc_iter is None:
            return []
        batch: list[dict] = []
        for _ in range(k):
            try:
                batch.append(next(svc_iter))
            except StopIteration:
                break
        return batch

    async def _one_batch():
        nonlocal written
        async with sem:
            few_shot = rng.sample(sample_rows, k=min(fewshot_pool, len(sample_rows)))
            svc_rows = _next_svc_batch(rows_per_call) if svc_iter is not None else None
            # In annotation mode, bail out when the source is exhausted.
            if svc_iter is not None and not svc_rows:
                return 0
            rows = await _generate_batch(model_name, few_shot, rows_per_call,
                                         lang_mix, registry,
                                         svc_source_rows=svc_rows)
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats[r["schema_name"]] += 1
        f.flush()
        written += len(rows)
        return len(rows)

    in_flight: list[asyncio.Task] = []
    svc_exhausted = False
    while written < target_rows and not svc_exhausted:
        while (len(in_flight) < workers * 2 and written < target_rows
               and not svc_exhausted):
            in_flight.append(asyncio.create_task(_one_batch()))
        done, pending = await asyncio.wait(
            in_flight, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            # A task that returned 0 in annotation mode means the SVC
            # source ran out — signal shutdown after draining.
            if svc_iter is not None and t.result() == 0:
                svc_exhausted = True
        in_flight = list(pending)
        elapsed = time.time() - t0
        rate = written / max(elapsed, 1.0)
        eta = (target_rows - written) / max(rate, 1e-6)
        print(f"  [{written:>6,}/{target_rows:,}]  "
              f"{rate:.1f} rows/s  ETA {eta/60:.1f}m  "
              f"top schemas: {stats.most_common(3)}")

    if in_flight:
        await asyncio.gather(*in_flight)
    f.close()
    return written


def _count_existing(path: Path, registry: LabelRegistry | None = None) -> int:
    """Count valid rows in ``path`` and (optionally) seed ``registry``
    so a resumed run keeps the same schema/rule ID assignments."""
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_well_formed(row):
                if registry is not None:
                    registry.absorb(row)
                n += 1
    return n


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", type=Path, required=True,
                    help="Few-shot seed JSONL showing the desired output format "
                         "(use sandbox/mingru_baseline/data/seed_enriched.jsonl)")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "sandbox/mingru_baseline/data/multitask_gemini_v1.jsonl")
    ap.add_argument("--target", type=int, default=50000,
                    help="Upper bound on rows written (stops earlier if SVC "
                         "source is exhausted in annotation mode)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent Gemini calls in flight")
    ap.add_argument("--rows-per-call", type=int, default=25)
    ap.add_argument("--model", default="gemini-3-flash-preview",
                    help="Gemini model name (e.g. gemini-3-flash-preview, "
                         "gemini-3-pro-preview, gemini-2.5-flash)")
    ap.add_argument("--lang-mix", default="0.6,0.3,0.1",
                    help="EN,FR,code-switch fractions (synthesis mode only)")
    ap.add_argument("--svc-source", type=Path, default=None,
                    help="Optional JSONL with real instructions in SVC format "
                         "(e.g. D:/grillcheese_training_data/instruct_svc_semantic.jsonl). "
                         "When given, switches to annotation mode — one output row "
                         "per input instruction, grounded in real text.")
    ap.add_argument("--svc-limit", type=int, default=None,
                    help="Max SVC rows to consume (None = whole file)")
    args = ap.parse_args()

    sample_rows = []
    with args.sample.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if is_well_formed(row):
                sample_rows.append(row)
    if not sample_rows:
        sys.exit(f"no valid rows in {args.sample}")
    print(f"  seed: {len(sample_rows)} valid rows from {args.sample.name}")

    parts = [float(x) for x in args.lang_mix.split(",")]
    assert len(parts) == 3, "--lang-mix must be EN,FR,mixed"
    lang_mix = (parts[0], parts[1], parts[2])

    print(f"  model: {args.model}  workers: {args.workers}  "
          f"rows/call: {args.rows_per_call}")
    print(f"  target: {args.target:,} rows -> {args.output}")
    print(f"  lang mix EN/FR/mixed: {lang_mix}\n")

    written = asyncio.run(_producer(
        workers=args.workers,
        target_rows=args.target,
        model_name=args.model,
        sample_rows=sample_rows,
        rows_per_call=args.rows_per_call,
        lang_mix=lang_mix,
        output_path=args.output,
        svc_source=args.svc_source,
        svc_limit=args.svc_limit,
    ))
    print(f"\n  done — {written:,} valid rows in {args.output}")


if __name__ == "__main__":
    main()
