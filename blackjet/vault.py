"""
blackjet.vault
==============

Holds the token -> real value map for a session.

The vault is the whole privacy claim. Two properties matter:

1. **It stays local.** Nothing in this module serializes to the network. The map
   is an in-memory dict, keyed by session id, and dies with the process.
2. **It never enters prompt context.** No code path sends the vault to the model.
   That is why prompt injection cannot exfiltrate it — there is nothing to leak.

Tokens are assigned sequentially per entity type (``<<PERSON_01>>``) rather than
derived from the value, so the token carries no information about what it
replaced. Identical values reuse the same token, which keeps the document
coherent for the model.

POC scope: in-memory, single process, no encryption at rest, no TTL sweeper.
Sessions are dropped explicitly via ``destroy()`` or when the process exits.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from . import config
from .logging_setup import register_secret

log = logging.getLogger(__name__)


class VaultError(RuntimeError):
    """Raised when a session is missing or a token cannot be resolved."""


@dataclass
class Session:
    """One document's token map.

    forward : real value -> token   (used during anonymization)
    reverse : token -> real value   (used during rehydration)
    """

    session_id: str
    forward: dict[str, str] = field(default_factory=dict)
    reverse: dict[str, str] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def token_for(self, value: str, entity_type: str) -> str:
        """Return the token for ``value``, minting a new one if needed."""
        existing = self.forward.get(value)
        if existing is not None:
            log.debug("Reusing token %s for repeated value", existing)
            return existing

        self.counters[entity_type] = self.counters.get(entity_type, 0) + 1
        token = (
            f"{config.TOKEN_OPEN}{entity_type}_"
            f"{self.counters[entity_type]:02d}{config.TOKEN_CLOSE}"
        )

        self.forward[value] = token
        self.reverse[token] = value

        # Register before the value can appear anywhere else, so an accidental
        # log of the raw document is scrubbed from this point forward.
        register_secret(value)

        log.debug("Minted token %s for a %s value", token, entity_type)
        return token

    def resolve(self, token: str) -> str:
        """Return the real value for ``token``.

        Raises
        ------
        VaultError
            If the token is not in this session. An unmatched token means the
            model altered or invented a placeholder, which must be surfaced, not
            silently passed through to the user.
        """
        try:
            return self.reverse[token]
        except KeyError:
            raise VaultError(f"Unknown token {token!r} in session {self.session_id}")

    def summary(self) -> list[dict]:
        """Log-safe listing for the Admin tab. Values are NOT included."""
        return [
            {"token": token, "length": len(value)}
            for token, value in self.reverse.items()
        ]


class Vault:
    """Thread-safe registry of sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str) -> Session:
        """Create (or replace) a session."""
        with self._lock:
            if session_id in self._sessions:
                log.warning("Session %s already existed; replacing", session_id)
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
            log.info("Vault session created: %s", session_id)
            return session

    def get(self, session_id: str) -> Session:
        """Fetch a session, raising VaultError if it is gone."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise VaultError(f"No such session: {session_id}")
        return session

    def destroy(self, session_id: str) -> bool:
        """Drop a session and its map. Returns True if something was removed."""
        with self._lock:
            removed = self._sessions.pop(session_id, None)
        if removed is not None:
            log.info(
                "Vault session destroyed: %s (%d token(s) discarded)",
                session_id,
                len(removed.reverse),
            )
            return True
        log.warning("destroy() called for unknown session %s", session_id)
        return False

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


# Module-level singleton. One process, one vault.
VAULT = Vault()
