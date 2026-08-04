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
from dataclasses import dataclass, field
from pathlib import Path

from .logging_setup import register_secret

log = logging.getLogger(__name__)


class LoaderError(RuntimeError):
    """Raised when a document cannot be read or contains no usable text."""


@dataclass
class LoadedDocument:
    """A source document reduced to text, with provenance for the UI."""

    doc_id: str
    source_path: Path
    source_format: str
    text: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

_PDF_METADATA_KEYS = ("/Author", "/Title", "/Subject", "/Creator", "/Producer")


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
    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            log.exception("Text extraction failed on page %d of %s", index + 1, path.name)
            warnings.append(f"Page {index + 1}: text extraction failed")

    text = "\n".join(p for p in pages if p.strip())

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

# Keys whose values are structural rather than personal. Skipping them keeps
# noise out of detection without hiding anything identifying.
_JSON_SKIP_KEYS = {"startDate", "endDate", "date", "level", "score"}


def _walk_json(node, path: str = "") -> list[tuple[str, str]]:
    """Recursively collect (json_path, string_value) pairs."""
    out: list[tuple[str, str]] = []

    if isinstance(node, dict):
        for key, value in node.items():
            if key in _JSON_SKIP_KEYS:
                continue
            out.extend(_walk_json(value, f"{path}.{key}" if path else key))

    elif isinstance(node, list):
        for i, value in enumerate(node):
            out.extend(_walk_json(value, f"{path}[{i}]"))

    elif isinstance(node, str):
        if node.strip():
            out.append((path, node))

    elif isinstance(node, (int, float)):
        out.append((path, str(node)))

    return out


def _load_json_resume(path: Path) -> LoadedDocument:
    """Flatten a JSON Resume into labelled ``path: value`` lines."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.exception("Malformed JSON in %s", path.name)
        raise LoaderError(f"Malformed JSON in {path.name}: {exc}") from exc
    except OSError as exc:
        raise LoaderError(f"Could not read {path.name}: {exc}") from exc

    pairs = _walk_json(data)
    if not pairs:
        raise LoaderError(f"{path.name} contained no text values")

    text = "\n".join(f"{key}: {value}" for key, value in pairs)
    log.info("Loaded JSON resume %s: %d field(s)", path.name, len(pairs))

    return LoadedDocument(
        doc_id=path.stem,
        source_path=path,
        source_format="jsonresume",
        text=text,
        warnings=[],
        metadata={},
    )


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def _load_txt(path: Path) -> LoadedDocument:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise LoaderError(f"Could not read {path.name}: {exc}") from exc

    if not text.strip():
        raise LoaderError(f"{path.name} is empty")

    return LoadedDocument(
        doc_id=path.stem,
        source_path=path,
        source_format="txt",
        text=text,
        warnings=[],
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
