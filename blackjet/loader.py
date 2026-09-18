"""
blackjet.loader
===============

Turns a source document into plain text for detection.

Three formats, matching the project corpus:

* ``.pdf``  — text layer via pypdf, plus a metadata sweep (Author/Title/Producer/
  Subject/Creator). Metadata matters: a name can live in ``/Author`` and appear
  nowhere in the visible text, which is exactly the case a text-only pipeline
  misses.
* ``.txt``  — read as-is.
* ``.json`` — JSON Resume schema. Walked recursively so every string value is
  included regardless of nesting.

Scanned/image-only PDFs are **not** OCR'd. They are detected (no extractable
text) and reported, so the caller can warn rather than silently anonymize an
empty string and claim success.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .logging_setup import register_secret

log = logging.getLogger(__name__)


class LoaderError(RuntimeError):
    """Raised when a document cannot be read or contains no usable text."""


@dataclass
class ValueSpan:
    """One data value's location inside ``LoadedDocument.text``.

    ``start``/``end`` are character offsets into ``text``; ``path`` is where the
    value came from in the source document. This is the missing link the rest of
    the pipeline needs to answer "where in the original was this?" — without it
    a detection cannot be told apart from a match that straddles two unrelated
    fields, and no offset survives back to the source.
    """

    start: int
    end: int
    path: str


@dataclass
class LoadedDocument:
    """A source document reduced to text, with provenance for the UI.

    ``value_spans`` is populated only by loaders that know where the data values
    are (currently JSON Resume). When it is empty, ``text`` is an undifferentiated
    string and downstream code treats every offset as fair game, which is the
    historical behaviour.
    """

    doc_id: str
    source_path: Path
    source_format: str
    text: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    value_spans: list[ValueSpan] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

_PDF_METADATA_KEYS = ("/Author", "/Title", "/Subject", "/Creator", "/Producer")

# Substrings that mark a font as decorative rather than textual, matched
# case-insensitively against /BaseFont. Deliberately narrow: SFRM1000 draws the
# bullet characters in john_doe.pdf and is a text font, not an icon font.
_ICON_FONT_MARKERS = (
    "fontawesome", "wingding", "dingbat", "glyphicon",
    "materialicons", "octicon", "entypo", "ionicons",
)

# A run of this many spaces in layout-mode output is a column gap, not padding.
_COLUMN_GAP = 6


def _is_icon_font(basefont: str) -> bool:
    low = (basefont or "").lower()
    return any(marker in low for marker in _ICON_FONT_MARKERS)


def _collect_runs(page) -> list[tuple[str, str]]:
    """Return (basefont, text) for every text run on the page.

    Layout-mode extraction does not expose font identity, so this second pass
    is the only way to tell an icon glyph from a letter. Doing it by codepoint
    instead would be guesswork, and would also strip legitimate symbols.
    """
    runs: list[tuple[str, str]] = []

    def visit(text, cm, tm, font_dict, font_size):
        if not text.strip():
            return
        basefont = ""
        try:
            if font_dict:
                basefont = str(font_dict.get("/BaseFont", ""))
        except Exception:
            basefont = ""
        runs.append((basefont.split("+")[-1], text))

    page.extract_text(visitor_text=visit)
    return runs


def _strip_icon_runs(text: str, runs) -> tuple[str, list[str], list[str]]:
    """Remove icon-font runs from ``text``; report what was removed."""
    removed: list[str] = []
    ambiguous: list[str] = []
    for basefont, run in runs:
        if not _is_icon_font(basefont) or not run.strip():
            continue
        count = text.count(run)
        if count == 1:
            text = text.replace(run, "", 1)
            removed.append(run)
        elif count > 1:
            ambiguous.append(run)
    return text, removed, ambiguous


def _normalize_columns(text: str) -> str:
    """Collapse layout-mode padding; a wide run of spaces is a column break.

    Without this, a two-column header arrives as one line of mostly whitespace
    and the detector sees unrelated fields fused together.
    """
    out: list[str] = []
    for line in text.split("\n"):
        line = re.sub(r" {%d,}" % _COLUMN_GAP, "\n", line)
        line = re.sub(r" {2,5}", " ", line)
        out.extend(part.rstrip() for part in line.split("\n"))
    collapsed: list[str] = []
    for line in out:
        if not line and collapsed and not collapsed[-1]:
            continue
        collapsed.append(line)
    while collapsed and not collapsed[0]:
        collapsed.pop(0)
    while collapsed and not collapsed[-1]:
        collapsed.pop()
    return "\n".join(collapsed)


def _load_pdf(path: Path) -> LoadedDocument:
    """Extract the text layer and document metadata from a PDF."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise LoaderError(
            "pypdf is required to read PDFs. Install with: pip3 install pypdf"
        ) from exc

    warnings: list[str] = []
    metadata: dict[str, str] = {}

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        log.exception("Could not open PDF %s", path.name)
        raise LoaderError(f"Could not open PDF {path.name}: {exc}") from exc

    # --- text layer --------------------------------------------------------
    # Extraction mode is not a per-document guess. Layout mode is metric-aware,
    # which is what gets '771-555-0100' out of Ivy_Haddington.pdf instead of
    # '771 -555 -0100' — the phone is six text runs and the space before each
    # hyphen is synthesized by the default mode from a sub-space kerning gap.
    #
    # Mode alone is not enough. alfred_pennyworth_pm.pdf yields
    # '♂¶obile-alt(123) 456-7890' and '/linkedin-inapennyworth' under BOTH
    # modes, because that is not a layout problem: those runs are drawn in
    # FontAwesome fonts and concatenated into the same text stream, and
    # '/linkedin-in' is the icon's glyph name. Layout mode cannot report font
    # identity, so a second visitor pass collects it and the icon runs are
    # removed. Every removal is recorded — extraction loss must never be
    # indistinguishable from clean text.
    pages: list[str] = []
    removed_runs: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            runs = _collect_runs(page)
            page_text = page.extract_text(extraction_mode="layout") or ""
            page_text, removed, ambiguous = _strip_icon_runs(page_text, runs)
            removed_runs.extend(removed)
            for run in ambiguous:
                warnings.append(
                    f"Page {index + 1}: icon-font run {run!r} occurs more than once "
                    f"in the page text and was left in place rather than guessed at"
                )
            pages.append(_normalize_columns(page_text))
        except Exception:
            log.exception("Text extraction failed on page %d of %s", index + 1, path.name)
            warnings.append(f"Page {index + 1}: text extraction failed")

    text = "\n".join(p for p in pages if p.strip())

    if removed_runs:
        warnings.append(
            "Removed %d icon-font run(s) fused into the text layer: %s"
            % (len(removed_runs), ", ".join(repr(r) for r in removed_runs))
        )
        log.info("Removed icon-font runs from %s: %s", path.name, removed_runs)

    if not text.strip():
        warnings.append(
            "No text layer found. This is probably a scanned/image-only PDF. "
            "OCR is out of scope, so nothing can be detected in the page content."
        )
        log.warning("PDF %s has no extractable text layer", path.name)

    # --- metadata sweep ----------------------------------------------------
    try:
        raw_meta = reader.metadata or {}
        for key in _PDF_METADATA_KEYS:
            value = raw_meta.get(key)
            if value and str(value).strip():
                metadata[key.lstrip("/")] = str(value).strip()
    except Exception:
        log.exception("Could not read PDF metadata from %s", path.name)
        warnings.append("PDF metadata could not be read")

    # Metadata values are appended to the text so detection sees them. Without
    # this, a name present only in /Author is never examined.
    if metadata:
        meta_block = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        text = f"{text}\n\n[PDF METADATA]\n{meta_block}" if text.strip() else meta_block
        log.info("PDF metadata swept from %s: %s", path.name, list(metadata))

    return LoadedDocument(
        doc_id=path.stem,
        source_path=path,
        source_format="pdf",
        text=text,
        warnings=warnings,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# JSON Resume
# ---------------------------------------------------------------------------

def _count_leaves(node) -> int:
    """Every scalar in the parsed document, including nulls and booleans."""
    if isinstance(node, dict):
        return sum(_count_leaves(v) for v in node.values())
    if isinstance(node, list):
        return sum(_count_leaves(v) for v in node)
    return 1


def _serialize_with_spans(node) -> tuple[str, list[ValueSpan]]:
    """Serialize parsed JSON, recording where every data value lands.

    The document stays JSON. Earlier versions flattened it to ``path: value``
    lines joined by ``\\n``, escaping neither the separator nor the newline, so a
    value containing a newline produced lines indistinguishable from record
    boundaries and a single detection could span a value, a record break and the
    next path key. Keeping the JSON means that cannot happen: a newline inside a
    value serializes as the two characters ``\\n`` and no structural character is
    ambiguous.

    Spans cover the *content* of each scalar — inside the quotes for strings —
    so a substitution confined to a span can never break the syntax, and no key
    name is ever a substitution target.
    """
    parts: list[str] = []
    spans: list[ValueSpan] = []
    pos = 0

    def write(chunk: str) -> None:
        nonlocal pos
        parts.append(chunk)
        pos += len(chunk)

    def emit(value, path: str, level: int) -> None:
        nonlocal pos
        pad, pad_inner = "  " * level, "  " * (level + 1)

        if isinstance(value, dict):
            if not value:
                write("{}")
                return
            write("{\n")
            items = list(value.items())
            for index, (key, child) in enumerate(items):
                write(pad_inner + json.dumps(key, ensure_ascii=False) + ": ")
                emit(child, f"{path}.{key}" if path else key, level + 1)
                write(",\n" if index < len(items) - 1 else "\n")
            write(pad + "}")

        elif isinstance(value, list):
            if not value:
                write("[]")
                return
            write("[\n")
            for index, child in enumerate(value):
                write(pad_inner)
                emit(child, f"{path}[{index}]", level + 1)
                write(",\n" if index < len(value) - 1 else "\n")
            write(pad + "]")

        elif isinstance(value, str):
            encoded = json.dumps(value, ensure_ascii=False)
            write('"')
            start = pos
            write(encoded[1:-1])
            spans.append(ValueSpan(start, pos, path))
            write('"')

        elif value is None or isinstance(value, bool):
            # Not data the user wrote; nothing to detect and nothing to protect.
            write(json.dumps(value))

        else:
            encoded = json.dumps(value)
            start = pos
            write(encoded)
            spans.append(ValueSpan(start, pos, path))

    emit(node, "", 0)
    return "".join(parts) + "\n", spans


def _load_json_resume(path: Path) -> LoadedDocument:
    """Load a JSON Resume, keeping it JSON.

    Every scalar in the file reaches ``text``. An earlier version skipped any key
    named ``startDate``, ``endDate``, ``date``, ``level`` or ``score`` at any
    depth, on the grounds that such values were "structural rather than
    personal". They are neither: ``score`` is a GPA, ``level`` is a
    self-assessment, and employment date ranges are a textbook quasi-identifier.
    Worse, a value that never reaches ``text`` is never detected, never vaulted
    and therefore cannot be rehydrated — the document handed back to the user was
    permanently missing 14-15% of its content, with nothing in the UI to show it.
    """
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.exception("Malformed JSON in %s", path.name)
        raise LoaderError(f"Malformed JSON in {path.name}: {exc}") from exc
    except OSError as exc:
        raise LoaderError(f"Could not read {path.name}: {exc}") from exc

    text, spans = _serialize_with_spans(data)
    if not spans:
        raise LoaderError(f"{path.name} contained no data values")

    leaves = _count_leaves(data)
    warnings: list[str] = []
    if len(spans) != leaves:
        # Only nulls and booleans are legitimately unspanned. Anything else is a
        # loader bug, and a loader bug here is silent data loss.
        nulls = leaves - len(spans)
        log.info("%s: %d non-data leaf/leaves (null/boolean) not spanned", path.name, nulls)

    log.info(
        "Loaded JSON resume %s: %d leaf/leaves, %d data value(s), %d chars",
        path.name, leaves, len(spans), len(text),
    )

    return LoadedDocument(
        doc_id=path.stem,
        source_path=path,
        source_format="jsonresume",
        text=text,
        warnings=warnings,
        metadata={},
        value_spans=spans,
    )


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def _load_txt(path: Path) -> LoadedDocument:
    """Read a plain-text document, reporting any undecodable bytes.

    An earlier version decoded with ``errors="replace"`` and no warning. A
    mis-encoded name becomes an unmatchable string containing U+FFFD, the
    detector finds nothing there, and the document reports a clean pass — the
    failure mode this application exists to prevent. Substitution still happens
    (a document with bad bytes is better anonymized than refused) but it is
    reported, and the report says how much was lost and where.
    """
    warnings: list[str] = []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LoaderError(f"Could not read {path.name}: {exc}") from exc

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        bad = text.count("�")
        first = text.find("�")
        line = text.count("\n", 0, first) + 1 if first >= 0 else 0
        warnings.append(
            f"{bad} byte sequence(s) could not be decoded as UTF-8 and were "
            f"replaced with U+FFFD (first at line {line}). Any PII in those "
            f"positions is unmatchable and will NOT be detected."
        )
        log.warning(
            "%s: %d undecodable byte sequence(s); detection over those "
            "positions is unreliable", path.name, bad,
        )

    if not text.strip():
        raise LoaderError(f"{path.name} is empty")

    return LoadedDocument(
        doc_id=path.stem,
        source_path=path,
        source_format="txt",
        text=text,
        warnings=warnings,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_LOADERS = {
    ".pdf": _load_pdf,
    ".txt": _load_txt,
    ".json": _load_json_resume,
}

SUPPORTED_SUFFIXES = tuple(_LOADERS)


def _register_name_secrets(path: Path) -> None:
    """Register a corpus file's name-bearing forms as log secrets.

    Registers the stem (``Ivy_Haddington``) and its space-joined form
    (``Ivy Haddington``) so neither the filename nor a prose mention of the
    doc_id can reach a log line. Values shorter than MIN_SECRET_LENGTH are
    ignored by register_secret itself.
    """
    try:
        register_secret(path.stem)
        spaced = path.stem.replace("_", " ").replace("-", " ")
        if spaced != path.stem:
            register_secret(spaced)
    except Exception:  # pragma: no cover - registration must never break loading
        log.exception("Could not register filename secrets (continuing)")


def load(path: Path) -> LoadedDocument:
    """Load ``path`` into a LoadedDocument.

    Raises
    ------
    LoaderError
        If the file is missing, of an unsupported type, or unreadable.
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError(f"No such file: {path}")
    if not path.is_file():
        raise LoaderError(f"Not a file: {path}")

    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise LoaderError(
            f"Unsupported file type {path.suffix!r}. "
            f"Supported: {', '.join(SUPPORTED_SUFFIXES)}"
        )

    # Corpus filenames are people's names. Register them as log secrets BEFORE
    # the first log line below, or a log pane would display e.g.
    # "Loading Ivy_Haddington.pdf" verbatim — a correctness defect in the
    # sanitization claim (handoff v0.2.1 §3.2, resolution (a)).
    _register_name_secrets(path)

    log.info("Loading %s (%s)", path.name, path.suffix.lower())
    doc = loader(path)
    log.info(
        "Loaded %s: format=%s chars=%d warnings=%d",
        doc.doc_id,
        doc.source_format,
        doc.char_count,
        len(doc.warnings),
    )
    return doc


def list_corpus(directory: Path) -> list[dict]:
    """List loadable documents in ``directory``, sorted by filename."""
    directory = Path(directory)
    if not directory.is_dir():
        log.error("Corpus directory does not exist: %s", directory)
        return []

    entries = []
    for p in sorted(directory.iterdir()):
        if not (p.is_file() and p.suffix.lower() in _LOADERS):
            continue
        # Register every corpus filename as a log secret at listing time, so
        # names are protected from the very first log line of the process —
        # including "Pipeline run requested: doc_id=..." in the server, which
        # fires before load(). API responses are unaffected; only logs are.
        _register_name_secrets(p)
        entries.append(
            {
                "doc_id": p.stem,
                "filename": p.name,
                "format": p.suffix.lstrip(".").lower(),
                "bytes": p.stat().st_size,
            }
        )
    log.info("Corpus at %s: %d document(s)", directory, len(entries))
    return entries
