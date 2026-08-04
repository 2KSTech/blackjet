#!/usr/bin/env python3
"""
measure_token_preservation.py
=============================

The first real test of the design's central assumption: **does the model
reproduce placeholder tokens verbatim?** (spec §9.8, handoff v0.2.1 §4.4).

For every corpus document this script anonymizes locally, sends the tokenized
text through the configured OpenAI-compatible backend (``AI_PROVIDER`` — ollama
by default), rehydrates, and reports per-document and aggregate:

* tokens sent / tokens returned / unmatched after rehydration
* exact-preservation rate per entity type (PERSON, LOCATION, EMAIL, ...)
* whether failures look systematic (delimiter stripped, case changed,
  digits altered) or random

Run it once per candidate model and keep the tables; together they are the
delimiter-choice evidence for open defect 2 (``<<..>>`` vs ``⟪..⟫`` vs
``[[..]]``).

Usage
-----
    # uses the same .env / environment as the app; run.sh-style sourcing:
    set -a; . ./.env; set +a
    .venv/bin/python3 scripts/measure_token_preservation.py
    .venv/bin/python3 scripts/measure_token_preservation.py --doc john_doe
    .venv/bin/python3 scripts/measure_token_preservation.py --json results.json

Cost note: one model call per document (6 total). With the default local
ollama that is free but can be slow on modest hardware; with a hosted provider
(openai, groq, together, openrouter) it is billed.

Exit codes: 0 = ran and all tokens preserved everywhere; 1 = ran but some
tokens were mangled (the interesting result — see the table); 2 = could not
run (configuration / connection).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from blackjet import config, model_client, pipeline  # noqa: E402
from blackjet.loader import list_corpus, load  # noqa: E402
from blackjet.logging_setup import setup_logging  # noqa: E402
from blackjet.vault import VAULT  # noqa: E402

TOKEN_RE = re.compile(
    re.escape(config.TOKEN_OPEN) + r"([A-Z_]+)_(\d{2})" + re.escape(config.TOKEN_CLOSE)
)


def tokens_in(text: str) -> Counter:
    """Multiset of full tokens present in *text*."""
    return Counter(m.group(0) for m in TOKEN_RE.finditer(text))


def entity_of(token: str) -> str:
    m = TOKEN_RE.fullmatch(token)
    return m.group(1) if m else "?"


def classify_mangling(token: str, response: str) -> str:
    """Best-effort description of *how* a missing token was altered."""
    inner = token[len(config.TOKEN_OPEN):-len(config.TOKEN_CLOSE)]
    if inner in response:
        return "delimiters stripped/altered"
    if inner.lower() in response.lower():
        return "case changed"
    stem = inner.rsplit("_", 1)[0]
    if stem in response:
        return "number altered/dropped"
    return "token absent entirely"


def measure_doc(entry: dict) -> dict:
    doc = load(Path(config.DATA_DIR) / entry["filename"])
    session = VAULT.create(uuid.uuid4().hex)
    try:
        anonymized, _ = pipeline.anonymize(doc.text, session)
        leaked = pipeline.scan_for_leaks(anonymized, session)
        if leaked:
            return {"doc_id": doc.doc_id, "error": f"tripwire fired ({len(leaked)} leaks) — not sent"}

        sent = tokens_in(anonymized)
        try:
            response, seconds = model_client.send(anonymized)
        except model_client.ModelError as exc:
            return {"doc_id": doc.doc_id, "error": str(exc)}

        returned = tokens_in(response)
        rehydrated, unmatched = pipeline.rehydrate(response, session)

        per_entity: dict[str, dict] = {}
        mangling: Counter = Counter()
        for token, n_sent in sent.items():
            ent = entity_of(token)
            slot = per_entity.setdefault(ent, {"sent": 0, "preserved": 0})
            n_ret = returned.get(token, 0)
            slot["sent"] += n_sent
            slot["preserved"] += min(n_sent, n_ret)
            if n_ret < n_sent:
                mangling[classify_mangling(token, response)] += n_sent - n_ret

        return {
            "doc_id": doc.doc_id,
            "model": config.AI_PROVIDER_MODEL,
            "provider": config.AI_PROVIDER,
            "seconds": round(seconds, 2),
            "tokens_sent": sum(sent.values()),
            "tokens_returned": sum(returned.values()),
            "unmatched_after_rehydrate": len(unmatched),
            "unmatched_tokens": unmatched,
            "per_entity": per_entity,
            "mangling_modes": dict(mangling),
            "response_chars": len(response),
        }
    finally:
        VAULT.destroy(session.session_id)


def print_table(results: list[dict]) -> None:
    print(f"\nprovider={config.AI_PROVIDER}  model={config.AI_PROVIDER_MODEL}  "
          f"delimiters={config.TOKEN_OPEN}...{config.TOKEN_CLOSE}")
    header = f"{'doc':22} {'sent':>5} {'ret':>5} {'unmat':>5} {'sec':>7}  per-entity preserved/sent"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['doc_id']:22} ERROR: {r['error']}")
            continue
        ents = "  ".join(
            f"{e}:{v['preserved']}/{v['sent']}" for e, v in sorted(r["per_entity"].items())
        )
        print(f"{r['doc_id']:22} {r['tokens_sent']:>5} {r['tokens_returned']:>5} "
              f"{r['unmatched_after_rehydrate']:>5} {r['seconds']:>7.2f}  {ents}")
        if r["mangling_modes"]:
            print(f"{'':22} mangling: {r['mangling_modes']}")

    ok = [r for r in results if "error" not in r]
    if ok:
        total_sent = sum(r["tokens_sent"] for r in ok)
        total_preserved = sum(
            v["preserved"] for r in ok for v in r["per_entity"].values()
        )
        total_unmatched = sum(r["unmatched_after_rehydrate"] for r in ok)
        rate = 100.0 * total_preserved / total_sent if total_sent else 0.0
        print(f"\nAggregate: {total_sent} tokens sent, {total_preserved} returned "
              f"verbatim → {rate:.1f}% exact preservation; "
              f"{total_unmatched} additionally flagged unmatched by rehydrate")
        if total_preserved < total_sent and total_unmatched < (total_sent - total_preserved):
            print("NOTE: some mangled tokens (e.g. lowercased or delimiter-stripped) "
                  "no longer match the strict token pattern, so rehydrate cannot "
                  "even flag them as unmatched — they simply persist in the output. "
                  "Exact preservation above is the true survival metric.")
        modes: Counter = Counter()
        for r in ok:
            modes.update(r["mangling_modes"])
        if modes:
            print(f"Failure modes (systematic if concentrated): {dict(modes)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--doc", help="run a single doc_id instead of all")
    parser.add_argument("--json", metavar="PATH", help="also write results as JSON")
    args = parser.parse_args()

    setup_logging(config.LOG_LEVEL)

    if not model_client.is_configured():
        print(f"FATAL: {model_client.not_configured_reason()}", file=sys.stderr)
        return 2

    docs = list_corpus(config.DATA_DIR)
    if args.doc:
        docs = [d for d in docs if d["doc_id"] == args.doc]
        if not docs:
            print(f"FATAL: no such doc_id {args.doc!r}", file=sys.stderr)
            return 2

    results = [measure_doc(d) for d in docs]
    print_table(results)

    if args.json:
        try:
            Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nJSON written to {args.json}")
        except OSError as exc:
            print(f"WARNING: could not write {args.json}: {exc}", file=sys.stderr)

    if any("error" in r for r in results):
        return 2
    for r in results:
        for v in r["per_entity"].values():
            if v["preserved"] < v["sent"]:
                return 1
        if r["unmatched_after_rehydrate"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
