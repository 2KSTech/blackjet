"""
blackjet.model_client
=====================

**The only module in this application that touches the network.**

Everything else — detection, tokenization, the vault, rehydration — is local.
Keeping egress in one file makes the trust boundary auditable: if you want to
know what leaves the machine, you read this file and nothing else.

One protocol
------------
blackjet speaks **the OpenAI Chat Completions API and nothing else**:

    POST <base>/v1/chat/completions
    Authorization: Bearer <key>          (when the endpoint wants one)
    {"model": ..., "messages": [...], "stream": false}

That is the lingua franca of local and hosted inference alike, so one request
builder and one response parser cover every backend worth demoing. ``Ollama``
is the default and the premier target: it serves this exact API, needs no key,
and runs on the same machine as the vault — which is the whole point of a demo
about not letting PII leave your machine.

``AI_PROVIDER`` picks a preset (URL + default model); ``AI_PROVIDER_URL``
overrides it for anything not listed. See ``config.PROVIDER_PRESETS``.

The URL may be a bare host (``http://ip:port``), a base ending in ``/v1``, or
the full endpoint path. The missing suffix is appended; query parameters
(``?token=...``) survive either way.

The call is non-streamed. Streaming would split placeholder tokens across
chunk boundaries and require a hold-back buffer; that is deliberately out of
scope (spec §9.3).

Security note: because the egress URL is configurable, it is logged at startup
and on every send — with any embedded token redacted — so a misdirected
payload is visible rather than silent.
"""

from __future__ import annotations

import json
import logging
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config
from .logging_setup import register_secret

log = logging.getLogger(__name__)

# The model is asked to return the document unchanged. The demo is about the
# round trip, not about generation: a verbatim echo makes it obvious that tokens
# survived the journey, and any drift is immediately visible as an unmatched
# token in the UI.
SYSTEM_PROMPT = (
    "You are a document round-trip service. Return the user's document back to "
    "them verbatim, preserving all formatting, line breaks and spacing exactly. "
    "The document contains placeholder tokens of the form <<TYPE_NN>>, for "
    "example <<PERSON_01>>. These placeholders are deliberate. Reproduce every "
    "placeholder character-for-character. Never expand, rename, translate, "
    "reformat or guess the meaning of a placeholder. Never invent new "
    "placeholders. Output the document only, with no preamble or commentary."
)

# The one API path this application knows how to speak.
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# Query parameter names that plausibly carry an auth token; their values are
# registered as secrets so they can never appear in the log pane.
_TOKEN_PARAMS = ("token", "api_key", "apikey", "key", "auth")

# Low temperature: the model is asked to echo text verbatim, not to be creative.
TEMPERATURE = 0.2


def _token_from_url(url: str) -> str | None:
    """Return a token-like query parameter value from *url*, if present."""
    try:
        for key, value in urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query):
            if key.lower() in _TOKEN_PARAMS and value:
                return value
    except ValueError:  # pragma: no cover - defensive
        log.debug("Could not parse query string for a token", exc_info=True)
    return None


class ModelError(RuntimeError):
    """Raised when the model call cannot be completed."""


def _ssl_context() -> ssl.SSLContext:
    """Build a TLS context with a usable CA bundle.

    Python installed from python.org on macOS does not use the system keychain
    and ships without CA certificates, so HTTPS fails with
    CERTIFICATE_VERIFY_FAILED until certifi is present or the bundled
    "Install Certificates.command" has been run. certifi is a dependency of
    this project precisely so this works out of the box.

    Verification is never disabled. If no CA bundle can be found the call fails
    with an actionable message rather than falling back to an unverified
    connection.
    """
    try:
        import certifi  # type: ignore

        ctx = ssl.create_default_context(cafile=certifi.where())
        log.debug("TLS context using certifi bundle at %s", certifi.where())
        return ctx
    except ImportError:
        log.debug("certifi not installed; falling back to system CA store")
        return ssl.create_default_context()


def _chat_completions_path(path: str) -> str:
    """Return *path* extended to the chat-completions endpoint.

    Accepts the three shapes people actually put in a config file:

    * ``""`` / ``"/"``          — a bare host        -> ``/v1/chat/completions``
    * ``"/v1"``, ``"/openai/v1"`` — an OpenAI base   -> ``<base>/chat/completions``
    * ``".../chat/completions"`` — the full endpoint -> unchanged

    A base URL ending in ``/v1`` is what every provider's docs print, so
    treating it as "already half way there" avoids the ``/v1/v1/...`` 404 that
    naive appending produces.
    """
    trimmed = path.rstrip("/")

    if not trimmed:
        return CHAT_COMPLETIONS_PATH
    if trimmed.endswith("/chat/completions"):
        return trimmed
    if trimmed.endswith("/v1"):
        return trimmed + "/chat/completions"
    return trimmed + CHAT_COMPLETIONS_PATH


def resolve_url() -> str:
    """Resolve the effective chat-completions URL for the configured endpoint.

    * Extends the configured URL to the chat-completions path when needed.
    * Preserves any query string (e.g. ``?token=...``) in either case.
    * Registers token-like query values as log secrets.
    """
    raw = config.AI_PROVIDER_URL
    parts = urllib.parse.urlsplit(raw)

    if parts.scheme not in ("http", "https"):
        raise ModelError(
            f"AI_PROVIDER_URL has unsupported scheme {parts.scheme!r} "
            f"(expected http or https): {redacted_url(raw)}"
        )

    # Redact any token-like query values from future log output.
    for key, value in urllib.parse.parse_qsl(parts.query):
        if key.lower() in _TOKEN_PARAMS:
            register_secret(value)

    return urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            _chat_completions_path(parts.path),
            parts.query,
            "",
        )
    )


def redacted_url(url: str) -> str:
    """Return *url* with token-like query values replaced, for safe logging."""
    parts = urllib.parse.urlsplit(url)
    if not parts.query:
        return url
    pairs = [
        (k, "[REDACTED]" if k.lower() in _TOKEN_PARAMS else v)
        for k, v in urllib.parse.parse_qsl(parts.query)
    ]
    query = urllib.parse.urlencode(pairs, safe="[]")
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, query, "")
    )


def is_configured() -> bool:
    """True if the configured endpoint has what it needs to attempt a request.

    A URL is always required. A key is required only for providers that
    authenticate — Ollama and the other local servers do not, which is why the
    default backend is always "configured" and a dead endpoint is reported as a
    connection error with a fix rather than as silence.
    """
    if not config.AI_PROVIDER_URL:
        return False
    if config.PROVIDER_REQUIRES_KEY and not config.AI_PROVIDER_APIKEY:
        return False
    return True


def not_configured_reason() -> str:
    """Human-readable reason ``is_configured()`` is False, for the UI."""
    if not config.AI_PROVIDER_URL:
        return (
            f"AI_PROVIDER_URL is not set and {config.AI_PROVIDER!r} is not a "
            "known provider preset — round trip skipped. Set AI_PROVIDER_URL "
            "to any OpenAI-compatible endpoint, or set AI_PROVIDER to one of: "
            f"{', '.join(sorted(config.PROVIDER_PRESETS))}."
        )
    if config.PROVIDER_REQUIRES_KEY and not config.AI_PROVIDER_APIKEY:
        return (
            f"AI_PROVIDER_APIKEY is not set and provider {config.AI_PROVIDER!r} "
            "requires a key — round trip skipped. For a keyless local round "
            "trip use AI_PROVIDER=ollama."
        )
    return "Model backend is not configured — round trip skipped."


def _build_request(url: str, anonymized_text: str) -> urllib.request.Request:
    """Build the OpenAI chat-completions request. Never logs the payload text."""
    payload = {
        "model": config.AI_PROVIDER_MODEL,
        "stream": False,
        "temperature": TEMPERATURE,
        "max_tokens": config.MODEL_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": anonymized_text},
        ],
    }

    headers = {"content-type": "application/json"}

    # A token carried in the URL query is honored too, and used as the bearer
    # value when AI_PROVIDER_APIKEY is unset, because gateways differ in which
    # they accept.
    bearer = config.AI_PROVIDER_APIKEY or _token_from_url(url)
    if bearer:
        headers["authorization"] = f"Bearer {bearer}"

    body = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(url, data=body, method="POST", headers=headers)


def _parse_response(raw: str) -> str:
    """Extract the assistant message text from an OpenAI chat-completions body.

    Two Ollama-native shapes (``response`` from ``/api/generate``,
    ``message.content`` from ``/api/chat``) are accepted as a fallback. blackjet
    never *sends* those shapes, but a proxy in front of an instance may answer
    in them, and tolerating a response costs nothing while refusing one costs
    the user a working demo.
    """
    try:
        data = json.loads(raw)
    except ValueError as exc:
        log.exception("Model response is not JSON")
        raise ModelError(f"Model response is not JSON: {exc}") from exc

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise ModelError(
                "Chat-completions response has no message content. "
                f"Body starts: {raw[:200]!r}"
            )
        return content

    # Native-Ollama fallbacks.
    if isinstance(data.get("response"), str):
        log.debug("Parsed a native /api/generate response shape")
        return data["response"]

    message = data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        log.debug("Parsed a native /api/chat response shape")
        return message["content"]

    raise ModelError(
        "Response has no 'choices' array, so it is not an OpenAI "
        "chat-completions body. Is AI_PROVIDER_URL an OpenAI-compatible "
        f"endpoint? Body starts: {raw[:200]!r}"
    )


def _explain_http_error(code: int, detail: str) -> str:
    """Map an HTTP error to a message naming the likely cause and the fix."""
    base = f"Model API error {code}: {detail}"
    lowered = detail.lower()

    if code == 404 and ("model" in lowered or "not found" in lowered):
        return (
            f"{base} — the endpoint does not have a model called "
            f"{config.AI_PROVIDER_MODEL!r}. Fix: for Ollama run "
            f"`ollama pull {config.AI_PROVIDER_MODEL}` on the host and check "
            "`ollama list`; for a hosted provider set AI_PROVIDER_MODEL to an "
            "id that provider actually serves."
        )
    if code in (404, 405):
        return (
            f"{base} — that path does not accept this request. blackjet speaks "
            "only the OpenAI chat-completions API and posts to "
            f"{CHAT_COMPLETIONS_PATH}. Point AI_PROVIDER_URL at the base URL of "
            "an OpenAI-compatible server (Ollama 0.1.24+ serves one at "
            "http://127.0.0.1:11434/v1). A native-only /api/generate gateway is "
            "no longer supported."
        )
    if code in (401, 403):
        return (
            f"{base} — the endpoint rejected authentication. Check "
            "AI_PROVIDER_APIKEY (sent as a Bearer token) and any ?token=... in "
            "AI_PROVIDER_URL."
        )
    if code == 429:
        return f"{base} — rate limited; retry later."
    return base


def send(anonymized_text: str) -> tuple[str, float]:
    """Send tokenized text to the configured endpoint and return its response.

    Parameters
    ----------
    anonymized_text
        Text that has already passed through ``pipeline.anonymize`` **and** the
        ``scan_for_leaks`` tripwire. This function does not re-check; the caller
        is responsible for never handing it raw PII.

    Returns
    -------
    (response_text, elapsed_seconds)

    Raises
    ------
    ModelError
        On missing configuration, connection failure, HTTP error, timeout, or
        a response that is not an OpenAI chat-completions body.
    """
    if not is_configured():
        raise ModelError(not_configured_reason())

    if config.AI_PROVIDER_APIKEY:
        # Belt and braces: the key must never survive into the log pane even
        # if a future log line interpolates the wrong variable.
        register_secret(config.AI_PROVIDER_APIKEY)

    url = resolve_url()
    request = _build_request(url, anonymized_text)

    log.info(
        "Egress: POST %s provider=%s model=%s payload_bytes=%d timeout=%.0fs",
        redacted_url(url),
        config.AI_PROVIDER,
        config.AI_PROVIDER_MODEL,
        len(request.data or b""),
        config.MODEL_TIMEOUT_SECONDS,
    )

    is_https = url.lower().startswith("https:")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(
            request,
            timeout=config.MODEL_TIMEOUT_SECONDS,
            context=_ssl_context() if is_https else None,
        ) as response:
            raw = response.read().decode("utf-8")

    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:500]
        except Exception:
            pass
        message = _explain_http_error(exc.code, detail or str(exc.reason))
        log.error("%s", message)
        raise ModelError(message) from exc

    except urllib.error.URLError as exc:
        reason = str(exc.reason)
        if "CERTIFICATE_VERIFY_FAILED" in reason:
            log.error("TLS certificate verification failed: %s", reason)
            raise ModelError(
                "TLS certificate verification failed. This Python has no CA "
                "bundle. Fix with:  pip3 install certifi   "
                "(on macOS you can instead run the 'Install Certificates.command' "
                "in your /Applications/Python 3.x/ folder). "
                f"Original error: {reason}"
            ) from exc
        if "refused" in reason.lower():
            hint = (
                "nothing is listening at that address. "
                "If AI_PROVIDER=ollama: is Ollama running on that host? "
                "(`ollama serve`, or check the container / remote instance), "
                "and does AI_PROVIDER_URL point at the right ip:port?"
            )
            log.error("Connection refused to %s — %s", redacted_url(url), hint)
            raise ModelError(
                f"Connection refused to {redacted_url(url)} — {hint}"
            ) from exc
        if isinstance(exc.reason, TimeoutError) or "timed out" in reason.lower():
            log.error("Model request timed out after %.0fs", config.MODEL_TIMEOUT_SECONDS)
            raise ModelError(
                f"Model request timed out after {config.MODEL_TIMEOUT_SECONDS:.0f}s. "
                "Local models on modest hardware can be slow — raise "
                "BLACKJET_MODEL_TIMEOUT if the host is still working."
            ) from exc
        log.exception("Could not reach the model endpoint")
        raise ModelError(
            f"Could not reach {redacted_url(url)}: {reason}"
        ) from exc

    except TimeoutError as exc:
        log.exception("Model request timed out")
        raise ModelError(
            f"Model request timed out after {config.MODEL_TIMEOUT_SECONDS:.0f}s. "
            "Local models on modest hardware can be slow — raise "
            "BLACKJET_MODEL_TIMEOUT if the host is still working."
        ) from exc

    elapsed = time.monotonic() - started

    text = _parse_response(raw)

    if not text:
        log.error("Model returned an empty response body")
        raise ModelError(
            "Model returned no text content (response parsed cleanly but the "
            "text was empty)."
        )

    log.info("Egress complete in %.2fs, %d chars returned", elapsed, len(text))
    return text, elapsed
