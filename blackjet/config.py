"""
blackjet.config
===============

All runtime configuration in one place, read from environment variables.

No secrets are ever hard-coded. The only secret this app uses is
AI_PROVIDER_APIKEY, which is read once at import and never logged.

Environment variables
---------------------
AI_PROVIDER         OpenAI-compatible provider preset. Default: "ollama".
AI_PROVIDER_URL     Endpoint URL (base or full; ?token=... query preserved).
AI_PROVIDER_APIKEY  Bearer token. Falls back to HOSTED_OLLAMA_AUTH_TOKEN.
AI_PROVIDER_MODEL   Model id/tag. Falls back to OLLAMA_MODEL.
BLACKJET_HOST       Bind address. Default: 127.0.0.1
BLACKJET_PORT       Port. Default: 8080
BLACKJET_LOG_LEVEL  DEBUG|INFO|WARNING|ERROR. Default: INFO
BLACKJET_USE_NER    "1" to enable Presidio NER tier. Default: "1"
                    Falls back to regex-only automatically if Presidio
                    or the spaCy model is not installed.
BLACKJET_DATA_DIR   Directory of sample resumes. Default: ./data/resumes
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths ------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
STATIC_DIR: Path = PROJECT_ROOT / "static"
DATA_DIR: Path = Path(os.environ.get("BLACKJET_DATA_DIR", PROJECT_ROOT / "data" / "resumes"))

# --- Server -----------------------------------------------------------------

HOST: str = os.environ.get("BLACKJET_HOST", "127.0.0.1")
PORT: int = int(os.environ.get("BLACKJET_PORT", "8080"))
LOG_LEVEL: str = os.environ.get("BLACKJET_LOG_LEVEL", "INFO").upper()

# --- Detection --------------------------------------------------------------

USE_NER: bool = os.environ.get("BLACKJET_USE_NER", "1") == "1"

# Presidio confidence floor. Findings below this are recorded but not tokenized.
NER_THRESHOLD: float = float(os.environ.get("BLACKJET_NER_THRESHOLD", "0.5"))

# Entity types the NER tier is allowed to tokenize. Regex tier handles the rest.
NER_ENTITIES: tuple[str, ...] = ("PERSON", "LOCATION")

# spaCy model backing Presidio. en_core_web_lg (~600MB, ~1GB RAM) has markedly
# better name recall than en_core_web_sm (~12MB, ~200MB RAM). Neither is
# downloaded automatically; see run.sh.
SPACY_MODEL: str = os.environ.get("BLACKJET_SPACY_MODEL", "en_core_web_lg")

# --- Model provider ---------------------------------------------------------
#
# blackjet speaks exactly one protocol: the OpenAI Chat Completions API. Every
# provider below serves it, so there is one request builder and one response
# parser in model_client.py rather than a branch per vendor.
#
#   AI_PROVIDER         Preset name. Default "ollama" — keyless, local, and the
#                       backend this demo is built around. Any other name is
#                       treated as a generic OpenAI-compatible endpoint and
#                       requires AI_PROVIDER_URL.
#   AI_PROVIDER_URL     Endpoint URL, overriding the preset. May be a bare host
#                       (http://ip:port), an OpenAI base (.../v1), or the full
#                       .../v1/chat/completions path; the missing suffix is
#                       appended. A ?token=xxx query parameter is preserved.
#   AI_PROVIDER_APIKEY  Bearer token. Not needed by the local presets.
#   AI_PROVIDER_MODEL   Model id / tag, overriding the preset default.
#
# Fallbacks (so existing .env files keep working):
#   HOSTED_OLLAMA_URL, OLLAMA_URL, OLLAMA_HOST/OLLAMA_PORT,
#   HOSTED_OLLAMA_AUTH_TOKEN, OLLAMA_MODEL
#
# SECURITY NOTE: a settable egress URL is a place a payload could be
# redirected. The URL in effect is logged at startup and on every send
# (with any embedded token redacted). See README "Using a local model".

AI_PROVIDER: str = os.environ.get("AI_PROVIDER", "ollama").strip().lower()

_OLLAMA_DEFAULT_URL = "http://{host}:{port}/v1".format(
    host=os.environ.get("OLLAMA_HOST", "127.0.0.1"),
    port=os.environ.get("OLLAMA_PORT", "11434"),
)

# name -> (base URL, default model, needs an API key, runs on your hardware)
PROVIDER_PRESETS: dict[str, dict] = {
    # Local, keyless. The default and the one this demo is designed around.
    "ollama": {
        "url": _OLLAMA_DEFAULT_URL,
        "model": "mistral:7b",
        "requires_key": False,
        "local": True,
    },
    "vllm": {
        "url": "http://127.0.0.1:8000/v1",
        "model": "",
        "requires_key": False,
        "local": True,
    },
    "llamacpp": {
        "url": "http://127.0.0.1:8080/v1",
        "model": "local-model",
        "requires_key": False,
        "local": True,
    },
    "lmstudio": {
        "url": "http://127.0.0.1:1234/v1",
        "model": "local-model",
        "requires_key": False,
        "local": True,
    },
    # Hosted. Each needs its own key; each serves the same chat-completions API.
    "openai": {
        "url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "requires_key": True,
        "local": False,
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "requires_key": True,
        "local": False,
    },
    "together": {
        "url": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "requires_key": True,
        "local": False,
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "requires_key": True,
        "local": False,
    },
}


def _first_env(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment value among *names*."""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return default


# An unknown provider name is not an error: it is a generic OpenAI-compatible
# endpoint the user must name via AI_PROVIDER_URL. is_configured() reports the
# missing URL with the fix rather than failing at import time.
_preset: dict = PROVIDER_PRESETS.get(
    AI_PROVIDER, {"url": "", "model": "", "requires_key": False, "local": False}
)

AI_PROVIDER_URL: str = _first_env(
    "AI_PROVIDER_URL", "HOSTED_OLLAMA_URL", "OLLAMA_URL", default=_preset["url"]
)
AI_PROVIDER_APIKEY: str | None = _first_env(
    "AI_PROVIDER_APIKEY", "HOSTED_OLLAMA_AUTH_TOKEN"
)
AI_PROVIDER_MODEL: str = _first_env(
    "AI_PROVIDER_MODEL", "OLLAMA_MODEL", default=_preset["model"]
)
PROVIDER_REQUIRES_KEY: bool = bool(_preset["requires_key"])

# Local inference on modest hardware is far slower than a hosted API call.
_DEFAULT_TIMEOUT = "600" if _preset["local"] else "120"

MODEL_MAX_TOKENS: int = int(os.environ.get("BLACKJET_MAX_TOKENS", "4096"))
MODEL_TIMEOUT_SECONDS: float = float(
    os.environ.get("BLACKJET_MODEL_TIMEOUT", _DEFAULT_TIMEOUT)
)

# --- Tokens -----------------------------------------------------------------

# Delimiters for placeholder tokens. Chosen to be visually obvious in the UI and
# unlikely to occur naturally in a resume.
TOKEN_OPEN: str = "<<"
TOKEN_CLOSE: str = ">>"


def describe() -> dict:
    """Return a log-safe summary of config. Never includes the API key value."""
    return {
        "host": HOST,
        "port": PORT,
        "log_level": LOG_LEVEL,
        "use_ner": USE_NER,
        "ner_threshold": NER_THRESHOLD,
        "spacy_model": SPACY_MODEL,
        "provider": AI_PROVIDER,
        "model": AI_PROVIDER_MODEL,
        "requires_key": PROVIDER_REQUIRES_KEY,
        "api_key_present": bool(AI_PROVIDER_APIKEY),
        "data_dir": str(DATA_DIR),
    }
