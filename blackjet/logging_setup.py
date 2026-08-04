"""
blackjet.logging_setup
======================

Logging for a privacy demo has one hard rule: **real PII must never reach a log
line.** This module enforces that structurally rather than by discipline.

How it works
------------
1. A ``RedactingFilter`` is attached to every handler on the root logger.
2. Any value registered via ``register_secret()`` is replaced with ``[REDACTED]``
   in the formatted message of every record, wherever it appears.
3. The pipeline registers each detected PII value the moment it is detected, so
   from that point on the value cannot be logged even by accident.

This is a backstop, not a licence to log carelessly. Call sites should still log
token ids and counts rather than values.

Usage
-----
    from blackjet.logging_setup import setup_logging, register_secret, clear_secrets
    setup_logging()
    log = logging.getLogger(__name__)
"""

from __future__ import annotations

import logging
import sys
import threading
import time

# Thread-safe set of literal strings that must never appear in log output.
_SECRETS: set[str] = set()
_SECRETS_LOCK = threading.Lock()

REDACTED = "[REDACTED]"

# Values shorter than this are not registered: redacting a 2-character string
# would mangle unrelated log lines for no privacy benefit.
MIN_SECRET_LENGTH = 4


def register_secret(value: str) -> None:
    """Register a literal value that must be scrubbed from all future log output.

    Safe to call repeatedly with the same value. Short or empty values are
    ignored to avoid over-redacting unrelated text.
    """
    if not value or not isinstance(value, str):
        return
    if len(value.strip()) < MIN_SECRET_LENGTH:
        return
    with _SECRETS_LOCK:
        _SECRETS.add(value)


def clear_secrets() -> None:
    """Drop all registered secrets. Called on session teardown."""
    with _SECRETS_LOCK:
        _SECRETS.clear()


def secret_count() -> int:
    """Number of registered secrets. Safe to log."""
    with _SECRETS_LOCK:
        return len(_SECRETS)


class RedactingFilter(logging.Filter):
    """Replaces every registered secret in a record's message with [REDACTED]."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            # If the record cannot even be formatted, let it through unchanged
            # rather than swallowing a diagnostic.
            return True

        with _SECRETS_LOCK:
            secrets = tuple(_SECRETS)

        if not secrets:
            return True

        redacted = message
        for secret in secrets:
            if secret in redacted:
                redacted = redacted.replace(secret, REDACTED)

        if redacted != message:
            # Overwrite msg and clear args so the substitution survives formatting.
            record.msg = redacted
            record.args = ()

        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with the redaction filter attached.

    Idempotent: calling twice will not duplicate handlers.
    """
    root = logging.getLogger()

    # Remove handlers we previously installed so re-running is safe.
    for handler in list(root.handlers):
        if getattr(handler, "_blackjet", False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)-24s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    handler.addFilter(RedactingFilter())
    handler._blackjet = True  # type: ignore[attr-defined]

    try:
        root.setLevel(getattr(logging, level.upper()))
    except AttributeError:
        root.setLevel(logging.INFO)
        logging.getLogger(__name__).warning(
            "Unknown log level %r, falling back to INFO", level
        )

    root.addHandler(handler)

    # Ring buffer feeding the UI log pane (GET /api/logs). Installed here, with
    # the RedactingFilter attached, so it cannot be forgotten: a handler added
    # without the filter would emit unredacted text into the browser.
    ring = RingBufferHandler()
    ring.setFormatter(logging.Formatter(fmt="%(message)s"))
    ring.addFilter(RedactingFilter())
    ring._blackjet = True  # type: ignore[attr-defined]
    root.addHandler(ring)
    global _RING
    _RING = ring

    # Third-party libraries are extremely noisy at DEBUG (presidio-analyzer
    # alone emits ~40 lines per analyzer init). Pin them to WARNING so
    # BLACKJET_LOG_LEVEL=DEBUG means "more blackjet.* detail", not a flood.
    # Override with BLACKJET_DEBUG_THIRD_PARTY=1 if that flood is wanted.
    import os

    if os.environ.get("BLACKJET_DEBUG_THIRD_PARTY", "0") != "1":
        for name in _NOISY_THIRD_PARTY:
            logging.getLogger(name).setLevel(logging.WARNING)
        # Presidio emits ~20 lines of harmless config chatter at WARNING on
        # every analyzer init AND per-analysis "Entity X is not mapped"
        # warnings — enough to drown the log pane. ERROR only.
        for name in ("presidio-analyzer", "presidio_analyzer"):
            logging.getLogger(name).setLevel(logging.ERROR)


# Loggers pinned to WARNING by default (see setup_logging).
_NOISY_THIRD_PARTY = (
    "presidio-analyzer",
    "presidio_analyzer",
    "spacy",
    "urllib3",
    "filelock",
    "matplotlib",
    "asyncio",
)


class RingBufferHandler(logging.Handler):
    """Bounded in-memory log store backing the UI log pane.

    Thread-safe (the server is a ThreadingHTTPServer). Each record is stored
    already formatted, *after* the RedactingFilter has run, as::

        {"seq": int, "ts": float, "level": str, "logger": str, "message": str}

    ``records_since(seq)`` returns everything newer than ``seq`` so the client
    can poll cheaply with ``GET /api/logs?since=<seq>``.
    """

    def __init__(self, capacity: int = 2000) -> None:
        super().__init__()
        import collections

        self._records: "collections.deque[dict]" = collections.deque(maxlen=capacity)
        self._seq = 0
        self._lock2 = threading.Lock()
        # Signalled on every new record so readers can block instead of poll.
        self._new_record = threading.Condition(self._lock2)

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - defensive
            self.handleError(record)
            return
        with self._new_record:
            self._seq += 1
            self._records.append(
                {
                    "seq": self._seq,
                    "ts": record.created,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": message,
                }
            )
            self._new_record.notify_all()

    def records_since(self, seq: int) -> tuple[list[dict], int]:
        """Return (records newer than *seq*, latest seq). Never blocks."""
        with self._lock2:
            return [r for r in self._records if r["seq"] > seq], self._seq

    def wait_for_records(self, seq: int, timeout: float) -> tuple[list[dict], int]:
        """Block until a record newer than *seq* exists, or *timeout* elapses.

        This is what turns the UI log pane from a polling client into a
        long-polling one: one request per burst of activity instead of one
        every 400ms. A 40-second model call previously produced ~100 requests
        and ~100 access-log lines; it now produces one blocked request that
        returns the moment something is actually logged.

        Returns the same shape as records_since. An empty list means the wait
        timed out with nothing new, and the client should simply ask again.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._new_record:
            while True:
                records = [r for r in self._records if r["seq"] > seq]
                if records:
                    return records, self._seq
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return [], self._seq
                # Wake periodically regardless, so a lost notify cannot hang
                # the request for the full timeout.
                self._new_record.wait(min(remaining, 1.0))


# The ring buffer installed by setup_logging(), for the /api/logs endpoint.
_RING: RingBufferHandler | None = None


def get_ring_buffer() -> RingBufferHandler | None:
    """Return the installed ring buffer handler, or None before setup."""
    return _RING
