"""
blackjet.detect
===============

Finds PII spans in plain text and returns them as a list of ``Finding`` objects.

Two tiers
---------
**Tier 1 — regex/validator.** Deterministic patterns: email, phone, URL. Fast,
no model, no dependencies. Handles the bulk of a typical resume's contact data.

**Tier 2 — Presidio NER.** Statistical detection of PERSON and LOCATION, which
regex cannot do. Optional: if ``presidio-analyzer`` or the spaCy model is not
installed, the app logs a clear warning and runs regex-only rather than failing.

Overlap resolution
------------------
Findings from both tiers are merged with **longest-match-first**. A LOCATION
finding for "Portland" that sits inside an ADDRESS finding for
"1 Elm St, Portland, ME" is discarded, so the text is never double-tokenized.

Everything here is local. No network call is made by detection, ever.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable

from . import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Finding:
    """One detected PII span in a document.

    Attributes
    ----------
    start, end : character offsets into the source text (end exclusive)
    entity_type: PERSON | EMAIL | PHONE | URL | LOCATION
    value      : the matched substring
    tier       : "regex" or "ner"
    score      : detector confidence, 1.0 for deterministic regex matches
    """

    start: int
    end: int
    entity_type: str
    value: str
    tier: str
    score: float = 1.0

    @property
    def length(self) -> int:
        return self.end - self.start


# --------------------------------------------------------------------------
# Tier 1 — regex
# --------------------------------------------------------------------------

# Note on phone: deliberately permissive about separators and includes an
# optional country prefix, because resumes carry non-US formats. A US-only
# pattern fails silently on e.g. an Australian number, which looks like a clean
# pass and is the worst kind of bug in this application.
_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(
        r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    ),
    "URL": re.compile(
        r"\b(?:https?://|www\.)[^\s<>\"')\]]+"
        r"|\b(?:linkedin\.com|github\.com)/[^\s<>\"')\]]+",
        re.IGNORECASE,
    ),
    "PHONE": re.compile(
        r"(?<![\w.])"
        r"(?:\+\d{1,3}[\s.\-]?)?"          # optional country code
        r"(?:\(\d{1,4}\)[\s.\-]?|\d{1,4}[\s.\-])?"  # optional area code
        r"\d{3}[\s.\-]?\d{3,4}"
        r"(?:[\s.\-]?\d{2,4})?"
        r"(?![\w.])"
    ),
}

# A phone match must contain at least this many digits to be believable.
# Without this, date ranges and metrics ("2019 - 2022", "increased 30 000")
# produce false positives.
_MIN_PHONE_DIGITS = 9


def _regex_findings(text: str) -> list[Finding]:
    """Run every tier-1 pattern over ``text``."""
    findings: list[Finding] = []

    for entity_type, pattern in _PATTERNS.items():
        try:
            for match in pattern.finditer(text):
                value = match.group(0)

                if entity_type == "PHONE":
                    digits = sum(c.isdigit() for c in value)
                    if digits < _MIN_PHONE_DIGITS:
                        continue

                # Trim trailing punctuation that regex greedily absorbed.
                stripped = value.rstrip(".,;:)")
                end = match.start() + len(stripped)

                findings.append(
                    Finding(
                        start=match.start(),
                        end=end,
                        entity_type=entity_type,
                        value=stripped,
                        tier="regex",
                        score=1.0,
                    )
                )
        except re.error:
            log.exception("Regex failure for entity type %s; skipping", entity_type)

    log.debug("Tier 1 regex produced %d finding(s)", len(findings))
    return findings


# --------------------------------------------------------------------------
# Tier 0 — structural field rules
# --------------------------------------------------------------------------

# Fields whose value is personal because of WHERE it sits, not what it looks
# like. Content inspection cannot reach these: a username is an ordinary word,
# a street address is not a distinctive shape, and an avatar URL carrying a
# numeric account id has no name in it at all. The configured NER entity set is
# PERSON and LOCATION, so ADDRESS, POSTAL, HANDLE and IDENTIFIER are otherwise
# unreachable at any extraction quality.
#
# Matched against the JSON path of a value span, so this tier only ever applies
# to a loader that reports where its values came from.
_FIELD_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^basics\.name$"), "PERSON"),
    (re.compile(r"^basics\.email$"), "EMAIL"),
    (re.compile(r"^basics\.phone$"), "PHONE"),
    (re.compile(r"^basics\.(?:url|website)$"), "URL"),
    (re.compile(r"^basics\.image$"), "IDENTIFIER"),
    (re.compile(r"^basics\.location\.address$"), "ADDRESS"),
    (re.compile(r"^basics\.location\.postalCode$"), "POSTAL"),
    (re.compile(r"^basics\.location\.(?:city|region)$"), "LOCATION"),
    (re.compile(r"^basics\.profiles\[\d+\]\.username$"), "HANDLE"),
    (re.compile(r"^basics\.profiles\[\d+\]\.url$"), "URL"),
)

# basics.location.countryCode is deliberately absent. A country is not a
# quasi-identifier at this granularity once city and region are suppressed.


# A resume's first line is its owner's name. That is a layout convention, not a
# guess, and it is the one clue the NER tier cannot use: en_core_web_lg does not
# return 'Ivy Haddington' from Ivy_Haddington.pdf under either extraction mode,
# even though the string is intact and correctly spelled. The name most needing
# suppression is the one a statistical model is least anchored on, because a
# bare header line carries no surrounding sentence.
#
# Applied to the FIRST non-empty line only, and only when it still looks like a
# name. A wider "header zone" rule would take job titles with it — the line
# after the name is 'Product Manager' as often as not.
_HEADER_NAME_RE = re.compile(r"^[^\W\d_][\w'’\-\.]*(?: [^\W\d_][\w'’\-\.]*){1,3}$")


def header_name_findings(text: str) -> list[Finding]:
    """Tier 0 for unstructured documents: the name in the header line."""
    offset = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            start = offset + line.index(stripped)
            if (
                "," not in stripped
                and "@" not in stripped
                and not any(c.isdigit() for c in stripped)
                and _HEADER_NAME_RE.match(stripped)
            ):
                # Never log the value. This runs before vault.token_for has
                # registered it as a redaction secret, so anything written here
                # lands in the log in the clear.
                log.debug(
                    "Tier 0 header rule matched the first line (%d chars)",
                    len(stripped),
                )
                return [
                    Finding(
                        start=start,
                        end=start + len(stripped),
                        entity_type="PERSON",
                        value=stripped,
                        tier="header",
                        score=1.0,
                    )
                ]
            return []
        offset += len(line) + 1
    return []


def field_findings(text: str, value_spans) -> list[Finding]:
    """Tier 0: values that are personal by field identity.

    ``value_spans`` are ``loader.ValueSpan`` objects. Returns one finding per
    matching field, covering the whole value.
    """
    findings: list[Finding] = []
    for span in value_spans or ():
        for pattern, entity_type in _FIELD_RULES:
            if pattern.match(span.path):
                value = text[span.start : span.end]
                if value.strip():
                    findings.append(
                        Finding(
                            start=span.start,
                            end=span.end,
                            entity_type=entity_type,
                            value=value,
                            tier="field",
                            score=1.0,
                        )
                    )
                break

    log.debug("Tier 0 field rules produced %d finding(s)", len(findings))
    return findings


# --------------------------------------------------------------------------
# Tier 2 — Presidio NER
# --------------------------------------------------------------------------

_analyzer = None          # cached AnalyzerEngine
_analyzer_state = "cold"  # cold | ready | unavailable
_analyzer_error: str | None = None


def analyzer_status() -> dict:
    """Log-safe description of the NER tier, surfaced in the Admin tab."""
    return {
        "enabled_in_config": config.USE_NER,
        "state": _analyzer_state,
        "error": _analyzer_error,
        "threshold": config.NER_THRESHOLD,
        "entities": list(config.NER_ENTITIES),
    }


def _get_analyzer():
    """Load Presidio once and cache it.

    Returns None if Presidio is unavailable, having recorded why. The loading of
    the spaCy model is the expensive part (seconds), so this must happen once at
    process start, not per request.
    """
    global _analyzer, _analyzer_state, _analyzer_error

    if _analyzer_state == "ready":
        return _analyzer
    if _analyzer_state == "unavailable":
        return None

    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        from presidio_analyzer.nlp_engine import NlpEngineProvider  # type: ignore

        log.info(
            "Loading Presidio with spaCy model %s (one-time, several seconds)...",
            config.SPACY_MODEL,
        )
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": config.SPACY_MODEL}],
            }
        )

        # Presidio's SpacyNlpEngine calls _download_spacy_model_if_needed(),
        # which invokes spacy.cli.download() when the configured model is
        # absent. That is a ~400MB network fetch triggered by starting the app,
        # with no prompt and no way to decline — directly contrary to this
        # project's rule that nothing downloads without the user asking.
        #
        # Check first and refuse. Regex-only is the correct degraded mode here;
        # a surprise 400MB transfer is not.
        import spacy.util

        if not spacy.util.is_package(config.SPACY_MODEL):
            _analyzer_state = "unavailable"
            _analyzer_error = (
                f"spaCy model {config.SPACY_MODEL!r} is not installed in this "
                f"interpreter ({sys.executable}). NER is OFF; names and "
                f"locations will NOT be detected. Refusing to auto-download it "
                f"— run ./setup.sh, or install it yourself with: "
                f"{sys.executable} -m spacy download {config.SPACY_MODEL}"
            )
            log.warning(
                "NER tier unavailable: spaCy model %r absent from %s. "
                "Not downloading it automatically; continuing regex-only.",
                config.SPACY_MODEL,
                sys.executable,
            )
            return None

        _analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        _analyzer_state = "ready"
        _analyzer_error = None
        log.info("Presidio analyzer ready")
        return _analyzer

    except ImportError as exc:
        _analyzer_state = "unavailable"
        _analyzer_error = (
            "presidio-analyzer not installed. Running regex-only. "
            "Install with: pip3 install presidio-analyzer && "
            "python3 -m spacy download en_core_web_lg"
        )
        log.warning("NER tier unavailable: %s (%s)", _analyzer_error, exc)
        return None

    except Exception as exc:  # spaCy model missing, incompatible version, etc.
        _analyzer_state = "unavailable"
        _analyzer_error = (
            f"Presidio failed to initialise with spaCy model "
            f"{config.SPACY_MODEL!r}: {exc}. NER is OFF; names and locations "
            f"will NOT be detected. Install the model with: "
            f"python3 -m spacy download {config.SPACY_MODEL}"
        )
        log.exception("NER tier unavailable — continuing with regex only")
        return None


def warm_up() -> None:
    """Preload the analyzer at startup so the first request is not slow."""
    if config.USE_NER:
        _get_analyzer()
    else:
        log.info("NER tier disabled by config (BLACKJET_USE_NER=0)")


def _ner_findings(text: str) -> list[Finding]:
    """Run tier-2 NER over ``text``. Returns [] if the tier is unavailable."""
    if not config.USE_NER:
        return []

    analyzer = _get_analyzer()
    if analyzer is None:
        return []

    findings: list[Finding] = []
    try:
        results = analyzer.analyze(
            text=text,
            entities=list(config.NER_ENTITIES),
            language="en",
        )
        for r in results:
            if r.score < config.NER_THRESHOLD:
                log.debug(
                    "NER finding below threshold discarded: type=%s score=%.2f",
                    r.entity_type,
                    r.score,
                )
                continue
            findings.append(
                Finding(
                    start=r.start,
                    end=r.end,
                    entity_type=r.entity_type,
                    value=text[r.start : r.end],
                    tier="ner",
                    score=float(r.score),
                )
            )
    except Exception:
        # Detection failing must not take the request down silently, but it must
        # be loud: a silent NER failure means unprotected names in the payload.
        log.exception("NER analysis failed; regex findings only for this document")

    log.debug("Tier 2 NER produced %d finding(s) above threshold", len(findings))
    return findings


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------


def _resolve_overlaps(findings: Iterable[Finding]) -> list[Finding]:
    """Longest-match-first overlap resolution.

    Sorted by length descending, then start ascending. A finding is kept only if
    it does not overlap anything already kept.
    """
    kept: list[Finding] = []
    for f in sorted(findings, key=lambda x: (-x.length, x.start)):
        if any(f.start < k.end and k.start < f.end for k in kept):
            log.debug(
                "Discarding overlapping finding type=%s tier=%s at [%d,%d)",
                f.entity_type,
                f.tier,
                f.start,
                f.end,
            )
            continue
        kept.append(f)

    kept.sort(key=lambda x: x.start)
    return kept


# Minimum length for a derived handle to be searched for. Short forms collide
# with ordinary words; 'apennyworth' is safe, 'adavis' would be less so.
_MIN_DERIVATIVE_LEN = 8


def _derivatives(name: str) -> set[str]:
    """Handle-shaped forms a person's own name can produce."""
    parts = [p for p in re.split(r"[\s.]+", name.strip()) if p.isalpha()]
    if len(parts) < 2:
        return set()
    first, last = parts[0].lower(), parts[-1].lower()
    joined = {
        first + last,
        first[0] + last,
        first + last[0],
        last + first,
        f"{first}.{last}",
        f"{first}_{last}",
        f"{first}-{last}",
        f"{first[0]}.{last}",
    }
    return {d for d in joined if len(d) >= _MIN_DERIVATIVE_LEN}


def derivative_findings(text: str, persons: Iterable[str]) -> list[Finding]:
    """Handles and usernames built out of a name already detected in the document.

    A username is an ordinary-looking word: neither NER nor a string match
    against the full name will find 'apennyworth', and suppressing the name
    leaves it sitting in the clear. Seeding the search from a PERSON already
    found keeps this from being a guess — the only strings looked for are ones
    this document's own name could produce.

    The bare surname is deliberately not generated. It is identifying in
    isolation, but every label file in the corpus lists it as a value that must
    not fire on its own, and it appears inside the full name anyway.
    """
    wanted: set[str] = set()
    for person in persons:
        wanted |= _derivatives(person)
    if not wanted:
        return []

    findings: list[Finding] = []
    pattern = re.compile(
        r"(?<![\w.])(?:%s)(?![\w])" % "|".join(sorted(map(re.escape, wanted), key=len, reverse=True)),
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        findings.append(
            Finding(
                start=match.start(),
                end=match.end(),
                entity_type="HANDLE",
                value=match.group(0),
                tier="derivative",
                score=1.0,
            )
        )
    log.debug("Derivative pass produced %d finding(s)", len(findings))
    return findings


def _clip_to_values(findings: Iterable[Finding], text: str, value_spans) -> list[Finding]:
    """Discard findings that are not wholly inside one data value.

    Only meaningful when the loader reports value spans. A match that starts in
    one value and ends in another, or that lands on a key name or on JSON
    punctuation, is not an entity — it is an artifact of the substrate. Keeping
    it would tokenize across a record boundary and corrupt the document.
    """
    bounds = sorted((s.start, s.end) for s in value_spans)
    kept: list[Finding] = []
    dropped = 0
    for f in findings:
        inside = any(start <= f.start and f.end <= end for start, end in bounds)
        if inside:
            kept.append(f)
        else:
            dropped += 1
            log.debug(
                "Discarding finding type=%s tier=%s at [%d,%d): crosses a value "
                "boundary or lies outside any data value",
                f.entity_type, f.tier, f.start, f.end,
            )
    if dropped:
        log.info("Discarded %d finding(s) not contained in a single data value", dropped)
    return kept


def detect(text: str, value_spans=None) -> list[Finding]:
    """Detect all PII in ``text``, returned in document order.

    ``value_spans`` is optional. When the loader supplies it, tier 0 runs and
    every finding is required to sit wholly inside one data value.

    Raises
    ------
    TypeError
        If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"detect() expects str, got {type(text).__name__}")

    if not text.strip():
        log.warning("detect() called with empty text; nothing to do")
        return []

    if value_spans:
        field_hits = field_findings(text, value_spans)
    else:
        field_hits = header_name_findings(text)
    regex_hits = _regex_findings(text)
    ner_hits = _ner_findings(text)

    candidates = field_hits + regex_hits + ner_hits

    persons = {f.value for f in candidates if f.entity_type == "PERSON"}
    derived_hits = derivative_findings(text, persons)
    candidates += derived_hits

    if value_spans:
        candidates = _clip_to_values(candidates, text, value_spans)

    merged = _resolve_overlaps(candidates)

    log.info(
        "Detection complete: %d field + %d regex + %d ner + %d derivative -> "
        "%d after overlap resolution",
        len(field_hits),
        len(regex_hits),
        len(ner_hits),
        len(derived_hits),
        len(merged),
    )
    return merged
