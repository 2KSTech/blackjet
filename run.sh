#!/usr/bin/env bash
# blackjet launcher. Run from the project root.
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
[ -x .venv/bin/python3 ] && PY=.venv/bin/python3

# .env supplies the model backend settings (AI_PROVIDER*) and any BLACKJET_* overrides.
[ -f .env ] && set -a && . ./.env && set +a

# NER requires a spaCy model. This script NEVER downloads it for you:
# it is ~400MB and that is your decision, not this script's.
#
# When BLACKJET_USE_NER=0 the model is irrelevant, so say nothing: warning about
# a missing model in a mode that will never load one is just noise.
if [ "${BLACKJET_USE_NER:-1}" = "0" ]; then
  printf 'NER disabled (BLACKJET_USE_NER=0). Regex detection only: email, phone, URL.\n' >&2
elif ! "$PY" -c "import spacy,importlib.util as u; import sys; sys.exit(0 if u.find_spec('en_core_web_lg') or u.find_spec('en_core_web_sm') else 1)" 2>/dev/null; then

  # Before claiming the model is missing, check whether it is merely installed
  # into a DIFFERENT interpreter. Installing into system Python while run.sh
  # uses .venv/bin/python3 is the single most common setup failure here, and
  # the old message sent people to re-download 600MB they already had.
  OTHER=""
  if [ "$PY" != "python3" ] && command -v python3 >/dev/null 2>&1; then
    if python3 -c "import importlib.util as u; import sys; sys.exit(0 if u.find_spec('en_core_web_lg') or u.find_spec('en_core_web_sm') else 1)" 2>/dev/null; then
      OTHER="$(python3 -c 'import sys; print(sys.executable)' 2>/dev/null)"
    fi
  fi

  if [ -n "$OTHER" ]; then
    cat >&2 <<MSG
------------------------------------------------------------------
The spaCy model IS installed — but into a different interpreter.

  this app uses : $("$PY" -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PY")
  model found in: $OTHER

Do NOT download it again. Install it into the interpreter above:

  $PY -m spacy download en_core_web_lg

Or re-run ./setup.sh, which does this for you.
------------------------------------------------------------------
MSG
  else
    cat >&2 <<MSG
------------------------------------------------------------------
No spaCy model installed. NER (names, locations) will be OFF.
Regex detection (email, phone, URL) still works.

Interpreter in use: $("$PY" -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PY")

Easiest fix — run the setup script:

  ./setup.sh              large model, ~600MB, best recall
  ./setup.sh --small      small model, ~12MB, weaker on names

Or install manually, using THAT interpreter, not a bare python3:

  $PY -m spacy download en_core_web_lg
  $PY -m spacy download en_core_web_sm   # then set
                                         # BLACKJET_SPACY_MODEL=en_core_web_sm in .env

To silence this and stay regex-only, set BLACKJET_USE_NER=0 in .env
------------------------------------------------------------------
MSG
  fi
fi

exec "$PY" -m blackjet.server
