#!/usr/bin/env python3
"""blackjet regression — every corpus document through the real pipeline.

Loads each document with ``blackjet.loader``, anonymizes and rehydrates it with
``blackjet.pipeline``, and asserts:

  round trip      rehydrate(anonymize(x)) reproduces the loader's text exactly;
                  for JSON, that the result also re-parses equal to the file
  no loss         JSON keeps every leaf; text keeps every line
  tripwire        no vaulted value present in the anonymized payload
  recall          every phase-1 span in data/labels/ is absent from it
  labels          every span resolves against the loader's own output

Exit 0 only if all documents pass. Needs the spaCy model for the NER tier; run
with BLACKJET_USE_NER=0 to check the deterministic tiers alone.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.ERROR)

from blackjet.loader import load                      # noqa: E402
from blackjet import pipeline, config                 # noqa: E402
from blackjet.vault import Session                    # noqa: E402

CORPUS = [
    ("thomasdavis.json", "thomasdavis_labels.json"),
    ("sample-resume.json", "sarah_johnson_labels.json"),
    ("rosie_miller.txt", "rosie_miller_labels.json"),
    ("john_doe.pdf", "john_doe_labels.json"),
    ("Ivy_Haddington.pdf", "ivy_haddington_labels.json"),
    ("alfred_pennyworth_pm.pdf", "alfred_pennyworth_labels.json"),
]
ROOT = Path(__file__).resolve().parent.parent


def check(doc_name, label_name):
    src = ROOT / "data" / "resumes" / doc_name
    doc = load(src)
    session = Session(session_id="regression")
    anon, findings = pipeline.anonymize(doc.text, session, value_spans=doc.value_spans)
    leaked = pipeline.scan_for_leaks(anon, session)
    rehydrated, unmatched = pipeline.rehydrate(anon, session)

    labels = json.loads((ROOT / "data" / "labels" / label_name).read_text(encoding="utf-8"))
    phase1 = [s for s in labels["spans"] if s["phase"] == 1]
    missed = [s["id"] for s in phase1 if s["value"] in anon]
    unresolvable = [s["id"] for s in labels["spans"]
                    if doc.text.count(s["value"]) < s.get("occurrence", 1)]

    results = [
        ("round trip", rehydrated == doc.text, "%d chars" % len(doc.text)),
        ("no unmatched tokens", not unmatched, ", ".join(unmatched) or "none"),
        ("tripwire clear", not leaked, ", ".join(leaked) or "none"),
        # Recall gates only when the NER tier is on. Regex-only is a documented
        # degraded mode: names and locations are not detectable without a model,
        # so failing there would be reporting the configuration, not a defect.
        ("phase-1 recall" if config.USE_NER else "phase-1 recall (not gating)",
         (not missed) or not config.USE_NER,
         "%d/%d%s" % (len(phase1) - len(missed), len(phase1),
                      "" if not missed else "  MISSED " + ", ".join(missed))),
        ("labels resolve", not unresolvable,
         "%d span(s)%s" % (len(labels["spans"]),
                           "" if not unresolvable else "  UNRESOLVABLE " + ", ".join(unresolvable))),
    ]
    if doc.source_format == "jsonresume":
        same = json.loads(rehydrated) == json.loads(src.read_text(encoding="utf-8"))
        leaves_in = len(json.loads(src.read_text(encoding="utf-8")) and doc.value_spans)
        results.insert(1, ("rehydrated == source JSON", same, "%d data values" % leaves_in))
    else:
        results.insert(1, ("line parity", doc.text.count("\n") == rehydrated.count("\n"),
                           "%d lines" % (doc.text.count("\n") + 1)))

    print("### %s   (%d findings, %d tokens)" % (doc_name, len(findings), len(session.reverse)))
    for warn in doc.warnings:
        print("    note: %s" % warn)
    failed = 0
    for name, ok, detail in results:
        print("    [%s] %-26s %s" % ("PASS" if ok else "FAIL", name, detail))
        failed += not ok
    return failed


def main():
    print("blackjet regression — NER %s, model %s\n"
          % ("on" if config.USE_NER else "OFF", config.SPACY_MODEL))
    failed = sum(check(d, l) for d, l in CORPUS)
    print()
    if failed:
        print("%d check(s) FAILED" % failed)
        return 1
    print("all %d documents passed" % len(CORPUS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
