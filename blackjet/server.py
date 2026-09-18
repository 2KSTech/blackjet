"""
blackjet.server
===============

Stdlib-only HTTP server (no FastAPI, no uvicorn, no installs).

Endpoints
---------
GET  /                      static UI
GET  /api/corpus            list sample documents
POST /api/run               {"doc_id": ...} -> full pipeline result
GET  /api/status            config + detector state

POST /api/run performs: load -> detect -> tokenize -> tripwire -> model call
-> rehydrate, and returns every intermediate state so the UI can show
Original / Anonymized / Rehydrated side by side.

If the provider is not configured — no endpoint URL, or no AI_PROVIDER_APIKEY
for a provider that needs one — the round trip stops after anonymization and
says so explicitly. It does not fake a model response.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, model_client, pipeline
from .detect import analyzer_status, warm_up
from .loader import LoaderError, list_corpus, load
from .logging_setup import get_ring_buffer, setup_logging
from .vault import VAULT

log = logging.getLogger(__name__)

MAX_BODY_BYTES = 1_000_000



def run_pipeline(doc_id: str) -> dict:
    """Execute the full round trip for one corpus document.

    Returns a JSON-serializable dict with every stage. Real values appear only
    in fields the browser needs to render (original text, rehydrated text) —
    they are never logged.
    """
    started = time.monotonic()
    result: dict = {"doc_id": doc_id, "ok": False, "stages": {}}

    # --- resolve & load ----------------------------------------------------
    log.info("--- stage: load ---")
    matches = [
        p for p in Path(config.DATA_DIR).iterdir()
        if p.is_file() and p.stem == doc_id
    ]
    if not matches:
        result["error"] = f"Unknown doc_id: {doc_id}"
        return result

    try:
        doc = load(matches[0])
    except LoaderError as exc:
        result["error"] = str(exc)
        return result

    result["stages"]["original"] = {
        "text": doc.text,
        "format": doc.source_format,
        "chars": doc.char_count,
        "metadata": doc.metadata,
        "warnings": doc.warnings,
    }

    # --- anonymize ---------------------------------------------------------
    log.info("--- stage: detect + anonymize ---")
    session_id = uuid.uuid4().hex
    session = VAULT.create(session_id)
    try:
        anonymized, findings = pipeline.anonymize(
            doc.text, session, value_spans=doc.value_spans
        )
    except pipeline.PipelineError as exc:
        VAULT.destroy(session_id)
        result["error"] = f"Anonymization failed (nothing was sent): {exc}"
        return result

    result["stages"]["anonymized"] = {
        "text": anonymized,
        "findings": [
            {
                "type": f.entity_type,
                "tier": f.tier,
                "score": round(f.score, 2),
                "start": f.start,
                "end": f.end,
                "token": session.forward.get(f.value, "?"),
            }
            for f in findings
        ],
        "token_count": pipeline.count_tokens(anonymized),
        "unique_tokens": len(session.reverse),
    }

    # --- tripwire ----------------------------------------------------------
    log.info("--- stage: tripwire ---")
    leaked = pipeline.scan_for_leaks(anonymized, session)
    result["stages"]["tripwire"] = {"fired": bool(leaked), "leak_count": len(leaked)}
    if leaked:
        VAULT.destroy(session_id)
        result["error"] = (
            f"Tripwire blocked the send: {len(leaked)} real value(s) were still "
            "present in the outbound payload."
        )
        return result

    # --- model round trip --------------------------------------------------
    log.info("--- stage: model round trip ---")
    if not model_client.is_configured():
        VAULT.destroy(session_id)
        log.info("Round trip skipped: %s", model_client.not_configured_reason())
        result["stages"]["model"] = {
            "sent": False,
            "reason": model_client.not_configured_reason(),
        }
        result["ok"] = True
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        return result

    try:
        response_text, model_seconds = model_client.send(anonymized)
    except model_client.ModelError as exc:
        VAULT.destroy(session_id)
        result["error"] = f"Model call failed: {exc}"
        return result

    tokens_sent = pipeline.count_tokens(anonymized)
    tokens_returned = pipeline.count_tokens(response_text)
    result["stages"]["model"] = {
        "sent": True,
        "provider": config.AI_PROVIDER,
        "model": config.AI_PROVIDER_MODEL,
        "seconds": round(model_seconds, 2),
        "tokens_sent": tokens_sent,
        "tokens_returned": tokens_returned,
    }

    # --- rehydrate ---------------------------------------------------------
    log.info("--- stage: rehydrate ---")
    rehydrated, unmatched = pipeline.rehydrate(response_text, session)
    result["stages"]["rehydrated"] = {
        "text": rehydrated,
        "unmatched_tokens": unmatched,
    }

    VAULT.destroy(session_id)
    log.info("--- stage: done (%.2fs) ---", time.monotonic() - started)
    result["ok"] = not unmatched
    if unmatched:
        result["error"] = (
            f"{len(unmatched)} token(s) came back altered or missing and could "
            "not be rehydrated. They are left visible in the output."
        )
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = "blackjet/0.2"

    # -- helpers ------------------------------------------------------------

    def _json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            log.exception("Static file unreadable: %s", path)
            self._json({"error": "not found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _logs(self) -> None:
        """GET /api/logs?since=<seq> — records newer than seq, for the UI pane.

        Every record has already passed through the RedactingFilter attached
        to the ring buffer handler in setup_logging(); nothing is formatted
        here, only serialized.
        """
        from urllib.parse import parse_qs, urlsplit

        try:
            query = parse_qs(urlsplit(self.path).query)
            since = int(query.get("since", ["0"])[0])
        except (ValueError, TypeError):
            self._json({"error": "invalid 'since' parameter"}, 400)
            return

        ring = get_ring_buffer()
        if ring is None:
            # setup_logging() has not run — should be impossible via main().
            self._json({"records": [], "latest_seq": 0, "warning": "log buffer not initialized"})
            return

        records, latest = ring.records_since(since)
        self._json({"records": records, "latest_seq": latest})

    def log_message(self, fmt: str, *args) -> None:
        # Route access logs through logging (and its redaction filter).
        log.info("%s %s", self.address_string(), fmt % args)

    # -- routes -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path in ("/", "/index.html"):
                self._file(config.STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif self.path == "/api/corpus":
                self._json({"documents": list_corpus(config.DATA_DIR)})
            elif self.path.startswith("/api/logs"):
                self._logs()
            elif self.path == "/api/status":
                self._json(
                    {
                        "config": config.describe(),
                        "detector": analyzer_status(),
                        "model_configured": model_client.is_configured(),
                        "vault_sessions": VAULT.count(),
                    }
                )
            else:
                self._json({"error": "not found"}, 404)
        except Exception:
            log.exception("Unhandled error on GET %s", self.path)
            self._json({"error": "internal error"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path != "/api/run":
                self._json({"error": "not found"}, 404)
                return

            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._json({"error": "bad request body"}, 400)
                return

            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                doc_id = str(payload["doc_id"])
            except (ValueError, KeyError) as exc:
                self._json({"error": f"invalid request: {exc}"}, 400)
                return

            log.info("Pipeline run requested: doc_id=%s", doc_id)
            self._json(run_pipeline(doc_id))

        except Exception:
            log.exception("Unhandled error on POST %s", self.path)
            self._json({"error": "internal error"}, 500)


def main() -> None:
    setup_logging(config.LOG_LEVEL)
    log.info("blackjet starting: %s", config.describe())
    # Enumerate the corpus once at startup: list_corpus registers every
    # corpus filename as a log secret, so names are protected from the first
    # request onward (handoff v0.2.1 §3.2, resolution (a)).
    list_corpus(config.DATA_DIR)
    warm_up()  # load NER once, or log clearly why it is unavailable
    server = ThreadingHTTPServer((config.HOST, config.PORT), Handler)
    log.info("Listening on http://%s:%d", config.HOST, config.PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
