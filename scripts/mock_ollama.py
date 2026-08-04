#!/usr/bin/env python3
"""
mock_ollama.py — a tiny OpenAI-compatible mock for testing the model client
WITHOUT a real model. Serves POST /v1/chat/completions, the one route blackjet
speaks, and returns the OpenAI shape (``choices[0].message.content``).

The native Ollama routes /api/generate and /api/chat are also served, returning
their native shapes (``response`` / ``message.content``). blackjet never calls
them, but model_client's parser accepts those shapes as a tolerant fallback for
gateways that answer chat-completions with a native body — serving them here is
what exercises that fallback.

**This is a test double.** It exercises blackjet's request building, auth,
response parsing, error mapping and rehydration-against-a-response. It does
NOT tell you anything about real model behaviour (token mangling etc.) — for
that, run scripts/measure_token_preservation.py against a real instance.

Behaviours, selected by the model tag in the request:
  echo            → returns the user content verbatim (happy path)
  slow            → echo, after a 4-second delay (for during-run poll tests)
  mangle          → lowercases one token and strips delimiters from another
  missing-model   → 404 with an Ollama-style error body
  badauth         → 401 unless Authorization: Bearer test-token-1234 is present
                    (header only — matching the maintainer's gateway, which
                    authenticates by header, not by ?token= query)
  native-shape    → echo, but answers /v1/chat/completions with a NATIVE Ollama
                    body instead of the OpenAI one, to test the fallback parser
  (any other tag) → echo

Usage:  python3 scripts/mock_ollama.py [port]     (default 11500)

Point the app at it with:
    AI_PROVIDER_URL=http://127.0.0.1:11500/v1
    AI_PROVIDER_MODEL=echo
"""

from __future__ import annotations

import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 11500
EXPECTED_BEARER = "Bearer test-token-1234"

OPENAI_ROUTE = "/v1/chat/completions"
NATIVE_ROUTES = ("/api/generate", "/api/chat")


def _user_content(route: str, req: dict) -> str:
    """Pull the user turn out of whichever request shape arrived."""
    if route == "/api/generate":
        return req.get("prompt", "")
    return next(
        (m["content"] for m in req.get("messages", []) if m.get("role") == "user"),
        "",
    )


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?")[0]
        if route != OPENAI_ROUTE and route not in NATIVE_ROUTES:
            self._json({"error": {"message": f"not found: {route}"}}, 404)
            return

        length = int(self.headers.get("Content-Length", 0) or 0)
        req = json.loads(self.rfile.read(length) or b"{}")
        model = req.get("model", "echo")
        user = _user_content(route, req)

        if model == "missing-model":
            self._json(
                {"error": {"message": f'model "{model}" not found, try pulling it first'}},
                404,
            )
            return
        if model == "badauth" and self.headers.get("Authorization") != EXPECTED_BEARER:
            self._json({"error": {"message": "invalid token"}}, 401)
            return

        text = user
        if model == "slow":
            time.sleep(4)
        if model == "mangle":
            tokens = re.findall(r"<<[A-Z_]+_\d{2}>>", user)
            if tokens:
                text = text.replace(tokens[0], tokens[0].lower(), 1)
            if len(tokens) > 1:
                text = text.replace(tokens[1], tokens[1][2:-2], 1)

        native_body = route == "/api/generate" or model == "native-shape"
        if route == OPENAI_ROUTE and not native_body:
            self._json(
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(user) // 4,
                        "completion_tokens": len(text) // 4,
                        "total_tokens": (len(user) + len(text)) // 4,
                    },
                }
            )
        elif route == "/api/generate":
            # Native shape: the text lives under "response".
            self._json({"model": model, "response": text, "done": True})
        else:
            self._json(
                {
                    "model": model,
                    "message": {"role": "assistant", "content": text},
                    "done": True,
                }
            )

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[mock_ollama] " + fmt % args + "\n")


if __name__ == "__main__":
    print(
        f"[mock_ollama] listening on 127.0.0.1:{PORT}  "
        f"(POST {OPENAI_ROUTE}; native {', '.join(NATIVE_ROUTES)})",
        flush=True,
    )
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
