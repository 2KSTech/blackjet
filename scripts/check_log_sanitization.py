#!/usr/bin/env python3
"""
check_log_sanitization.py
=========================

Regression test for the log pane's privacy claim (handoff v0.2.1 §3.4 item 1).

Runs the full local pipeline for every corpus document IN-PROCESS, then greps
the captured ring-buffer log — the exact records `GET /api/logs` would serve —
for every labelled PII value in ``data/labels/*.json``.

Pass condition: **zero hits**.

Values shorter than ``logging_setup.MIN_SECRET_LENGTH`` (4) are exempt by
design: redacting a 2-character state code would mangle unrelated lines, so
short values can legitimately appear (documented in the UI).

Usage
-----
    .venv/bin/python3 scripts/check_log_sanitization.py
    # exit 0 = pass, exit 1 = leak(s) found, exit 2 = setup problem

No model call is made and no network is touched: the model round trip is not
part of the local log pipeline under test (and its egress lines contain only
token counts and a redacted URL).
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from blackjet import config, pipeline  # noqa: E402
from blackjet.loader import list_corpus, load  # noqa: E402
from blackjet.logging_setup import (  # noqa: E402
    MIN_SECRET_LENGTH,
    get_ring_buffer,
    setup_logging,
)
from blackjet.vault import VAULT  # noqa: E402

LABELS_DIR = PROJECT_ROOT / "data" / "labels"


def collect_label_values() -> dict[str, list[tuple[str, str]]]:
    """Return {doc_id: [(span_id, value), ...]} from every label file."""
    out: dict[str, list[tuple[str, str]]] = {}
    if not LABELS_DIR.is_dir():
        print(f"FATAL: labels directory missing: {LABELS_DIR}", file=sys.stderr)
        sys.exit(2)
    for path in sorted(LABELS_DIR.glob("*_labels.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"FATAL: cannot read {path.name}: {exc}", file=sys.stderr)
            sys.exit(2)
        values = [
            (span.get("id", "?"), span["value"])
            for span in data.get("spans", [])
            if isinstance(span.get("value"), str) and span["value"].strip()
        ]
        out[data.get("doc_id", path.stem)] = values
    return out


def main() -> int:
    setup_logging("DEBUG")  # worst case: most verbose, most chances to leak
    ring = get_ring_buffer()
    if ring is None:
        print("FATAL: ring buffer not installed by setup_logging()", file=sys.stderr)
        return 2

    labels = collect_label_values()
    docs = list_corpus(config.DATA_DIR)
    if not docs:
        print(f"FATAL: no corpus documents in {config.DATA_DIR}", file=sys.stderr)
        return 2

    print(f"Corpus: {len(docs)} document(s); label files: {len(labels)}")

    # Run the full local pipeline for every document.
    for entry in docs:
        doc = load(Path(config.DATA_DIR) / entry["filename"])
        session = VAULT.create(uuid.uuid4().hex)
        try:
            anonymized, _findings = pipeline.anonymize(doc.text, session)
            pipeline.scan_for_leaks(anonymized, session)
            # Exercise rehydrate too (against our own output — the model path
            # is out of scope here and adds no log lines with real values).
            pipeline.rehydrate(anonymized, session)
        finally:
            VAULT.destroy(session.session_id)

    records, latest = ring.records_since(0)
    log_text = "\n".join(r["message"] for r in records)
    print(f"Captured {len(records)} log record(s) (latest seq {latest})")

    leaks: list[tuple[str, str, str]] = []
    skipped_short = 0
    for doc_id, values in labels.items():
        for span_id, value in values:
            if len(value.strip()) < MIN_SECRET_LENGTH:
                skipped_short += 1
                continue
            if value in log_text:
                leaks.append((doc_id, span_id, value))

    print(f"Checked labelled values; {skipped_short} value(s) under "
          f"{MIN_SECRET_LENGTH} chars exempt by design")

    if leaks:
        print(f"\nFAIL: {len(leaks)} labelled value(s) found in the log:")
        for doc_id, span_id, value in leaks:
            # Print the value here deliberately — this is the test harness's
            # stdout, not the application log, and the maintainer needs to see
            # exactly what leaked.
            print(f"  {doc_id} {span_id}: {value!r}")
        return 1

    print("\nPASS: no labelled PII value appears in the captured log.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
