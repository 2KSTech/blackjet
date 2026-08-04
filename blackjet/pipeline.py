"""
blackjet.pipeline
=================

The three steps the demo exists to show.

    anonymize(text, session)  ->  text with PII replaced by tokens
    rehydrate(text, session)  ->  tokens replaced by real values

Between the two sits the model call (``blackjet.model_client``). That is the only
point at which anything leaves this machine, and by then the text contains
tokens only.
"""

from __future__ import annotations

import logging
import re

from . import config
from .detect import Finding, detect
from .vault import Session, VaultError

log = logging.getLogger(__name__)

# Matches any token this app mints, e.g. <<PERSON_01>>
_TOKEN_RE = re.compile(
    re.escape(config.TOKEN_OPEN) + r"([A-Z_]+_\d{2})" + re.escape(config.TOKEN_CLOSE)
)

# Matches token-shaped remnants a model may have mangled: wrong case, missing
# or altered delimiters, or stray whitespace inside them. Deliberately looser
# than _TOKEN_RE — it exists to DETECT failures, never to resolve them.
_MANGLED_TOKEN_RE = re.compile(
    r"(?<![\w<])"
    r"(?:"
    + re.escape(config.TOKEN_OPEN)
    + r"|<|\[\[|\[|\{\{|⟪)?"
    r"\s*(?P<body>[A-Za-z_]+_\d{2})\s*"
    r"(?:" + re.escape(config.TOKEN_CLOSE) + r"|>|\]\]|\]|\}\}|⟫)?"
    r"(?![\w>])",
    re.UNICODE,
)


class PipelineError(RuntimeError):
    """Raised when anonymization or rehydration cannot complete safely."""


def _guarded_pattern(value: str) -> str:
    """Escape ``value`` for regex, adding word-boundary guards where meaningful.

    Spec v0.4 §5.3 requires that a real value map to the same token *throughout
    the document*, which means substituting every literal occurrence, not only
    the offsets the detector happened to tag. Substituting bare strings would be
    reckless: a two-letter state code such as ``CA`` would otherwise rewrite the
    ``CA`` inside ``CATEGORY``.

    Guards are applied conditionally, because ``(?<!\\w)`` in front of a value
    that begins with a non-word character would block legitimate matches. In
    this corpus ``alfred_pennyworth_pm.pdf`` extracts a phone number fused to an
    icon glyph (``...alt(123) 456-7890``); the preceding character is a letter,
    so an unconditional leading guard would refuse to replace the value at the
    very offset the detector found it.
    """
    pattern = re.escape(value)
    if value[:1].isalnum() or value[:1] == "_":
        pattern = r"(?<!\w)" + pattern
    if value[-1:].isalnum() or value[-1:] == "_":
        pattern = pattern + r"(?!\w)"
    return pattern


def anonymize(text: str, session: Session) -> tuple[str, list[Finding]]:
    """Replace every occurrence of every detected PII value with its token.

    Substitution is **document-level**, per spec v0.4 §5.3: "Same real value ->
    same token throughout the document."

    Prior versions substituted at detector offsets only. Because the vault keys
    tokens by value, that produced a document in which ``Berkeley`` was
    tokenized where spaCy tagged it and left in the clear everywhere else — both
    a leak and a direct hint to the model about the mapping. ``scan_for_leaks``
    caught it; this function is the fix.

    Implementation notes
    --------------------
    A **single** ``re.sub`` pass over an alternation of all vault values is used,
    rather than one pass per value. Sequential passes would rewrite text that a
    previous pass had already turned into a token, corrupting it. One pass with
    the alternation ordered **longest value first** also resolves containment
    correctly: ``Cambridge`` is tried before ``CA``, so ``Cambridge, MA`` does
    not become ``<<LOCATION_03>>mbridge, MA``.

    Returns
    -------
    (anonymized_text, findings)

    Raises
    ------
    PipelineError
        If the input is unusable, or if detection fails.
    """
    if not isinstance(text, str) or not text.strip():
        raise PipelineError("anonymize() requires non-empty text")

    try:
        findings = detect(text)
    except Exception as exc:
        # Fail closed. If detection breaks we must not fall through to sending
        # the original text — that is the exact failure this app exists to
        # prevent.
        log.exception("Detection failed; refusing to continue")
        raise PipelineError(f"Detection failed, nothing was sent: {exc}") from exc

    if not findings:
        log.warning("No PII detected; document sent unchanged")
        return text, findings

    # Mint tokens for every distinct detected value first, so the substitution
    # table is complete before any rewriting begins.
    value_to_token: dict[str, str] = {}
    for f in findings:
        if f.value not in value_to_token:
            value_to_token[f.value] = session.token_for(f.value, f.entity_type)

    # Longest first: both for the alternation's leftmost-first semantics and so
    # that a value containing another value wins.
    ordered = sorted(value_to_token, key=len, reverse=True)

    try:
        combined = re.compile("|".join(_guarded_pattern(v) for v in ordered))
    except re.error as exc:
        # A malformed span (e.g. one spanning a newline) must not silently
        # disable anonymization.
        log.exception("Could not build substitution pattern; refusing to continue")
        raise PipelineError(
            f"Could not build the anonymization pattern, nothing was sent: {exc}"
        ) from exc

    replaced = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal replaced
        token = value_to_token.get(match.group(0))
        if token is None:
            # Unreachable unless the alternation and the table disagree. Leaving
            # the raw value in place would be a leak, so fail loudly instead.
            log.error("Substitution table miss for a matched span; failing closed")
            raise PipelineError("Substitution table miss during anonymization")
        replaced += 1
        return token

    out = combined.sub(_sub, text)

    log.info(
        "Anonymized document: %d detected span(s), %d distinct value(s), "
        "%d occurrence(s) replaced document-wide, %d unique token(s)",
        len(findings),
        len(value_to_token),
        replaced,
        len(session.reverse),
    )

    extra = replaced - len(findings)
    if extra > 0:
        log.info(
            "Document-level pass caught %d occurrence(s) the detector did not "
            "tag; these would have leaked under span-level substitution",
            extra,
        )

    return out, findings


def scan_for_leaks(text: str, session: Session) -> list[str]:
    """Return any real vault value found verbatim in ``text``.

    Run against the outbound payload as a tripwire immediately before the
    network call. It should always return an empty list. If it does not, the
    request must be aborted.

    Tokens are blanked out before the search. Without this the tripwire matches
    its own output: the two-letter state code ``CA`` occurs inside
    ``<<LOCATION_02>>`` and ``MA`` inside ``<<EMAIL_01>>``, which blocks a
    correctly anonymized payload. Blanking (rather than deleting) preserves
    offsets, so nothing new is created at the seams.

    This is deliberately a plain substring search, not a word-boundary one. A
    tripwire guarding an outbound payload should over-report rather than
    under-report; the only matches suppressed here are those provably inside a
    token this application minted.
    """
    scannable = _TOKEN_RE.sub(lambda m: " " * len(m.group(0)), text)

    leaked = [value for value in session.forward if value in scannable]
    if leaked:
        log.error(
            "TRIPWIRE: %d real value(s) present in outbound payload — blocking send",
            len(leaked),
        )
    else:
        log.debug("Tripwire clear: no vault value present in the outbound payload")
    return leaked


def rehydrate(text: str, session: Session) -> tuple[str, list[str]]:
    """Replace every token in ``text`` with its real value.

    Returns
    -------
    (rehydrated_text, unmatched_tokens)

    Unmatched tokens are left in place rather than deleted, so the failure is
    visible in the output instead of silently producing a plausible-looking
    document with a hole in it.
    """
    if not isinstance(text, str):
        raise PipelineError("rehydrate() requires a string")

    unmatched: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            return session.resolve(token)
        except VaultError:
            log.error("Unmatched token in model response: %s", token)
            unmatched.append(token)
            return token

    out = _TOKEN_RE.sub(_replace, text)

    # --- mangled-token sweep -------------------------------------------------
    # The strict pattern above only sees tokens the model reproduced exactly.
    # A token it lowercased (<<person_01>>) or stripped delimiters from
    # (PERSON_01) no longer matches, so it would be neither restored NOR
    # flagged: the run would report success with a dead placeholder sitting in
    # the output. Sweep for token-shaped remnants and report them.
    expected = set(session.reverse)
    for match in _MANGLED_TOKEN_RE.finditer(out):
        remnant = match.group(0)
        if remnant in unmatched:
            continue
        canonical = (
            f"{config.TOKEN_OPEN}{match.group('body').upper()}{config.TOKEN_CLOSE}"
        )
        if canonical in expected:
            log.error(
                "Mangled token in model response: %r (expected %s) — the model "
                "altered it, so it cannot be safely resolved",
                remnant,
                canonical,
            )
            unmatched.append(remnant)

    # Tokens that never came back at all are also failures: the real value is
    # simply missing from the output rather than visibly wrong.
    still_present = set(_TOKEN_RE.findall(out)) | {
        f"{config.TOKEN_OPEN}{m.group('body').upper()}{config.TOKEN_CLOSE}"
        for m in _MANGLED_TOKEN_RE.finditer(out)
    }
    restored = {t for t in expected if t not in still_present}
    dropped = sorted(expected - still_present - restored)
    for token in dropped:  # pragma: no cover - defensive; set algebra makes this empty
        unmatched.append(token)

    log.info(
        "Rehydrated document: %d token(s) restored, %d unmatched",
        len(expected) - len(unmatched),
        len(unmatched),
    )
    return out, unmatched


def count_tokens(text: str) -> int:
    """How many tokens appear in ``text``. Used for round-trip accounting."""
    return len(_TOKEN_RE.findall(text))
