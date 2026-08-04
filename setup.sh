#!/usr/bin/env bash
#
# blackjet setup.
#
#   ./setup.sh                 interactive: preflight, then prompts for a mode
#   ./setup.sh --large         non-interactive: NER on, en_core_web_lg  (~400MB)
#   ./setup.sh --small         non-interactive: NER on, en_core_web_sm  (~13MB)
#   ./setup.sh --no-ner        non-interactive: regex only, no model
#   ./setup.sh --check         preflight only; installs nothing, exits
#   ./setup.sh --yes           accept defaults, never prompt (CI)
#   PYTHON=/path/to/python3 ./setup.sh      override interpreter discovery
#
# THE FOUR MODES
#   1. regex only        email/phone/URL. No model, no download, ~0 extra RAM.
#   2. NER + small       adds names/locations. ~13MB, ~200MB RAM, weaker recall.
#   3. NER + large       adds names/locations. ~400MB, ~1GB RAM, best recall.
#   4. reuse             a model is already present; nothing is downloaded.
#
# WHY THIS SCRIPT EXISTS
#   Every setup failure reported against this project has been one fault: the
#   right package installed into the wrong Python. This script resolves ONE
#   interpreter up front and performs every install with it. After that point
#   nothing here calls a bare `python3` or `pip3`.
#
#   It also writes the matching .env keys. That is not cosmetic: Presidio will
#   silently download a spaCy model at first use if the configured model is
#   absent, so the only way to honour "regex only" or "small model" is for the
#   config to agree with what was installed.

set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".venv"
MIN_PY_MAJOR=3
MIN_PY_MINOR=10
LARGE_MODEL="en_core_web_lg"
SMALL_MODEL="en_core_web_sm"
LARGE_MB=400
SMALL_MB=13

MODE=""            # ask | regex | small | large
MODE_WAS_EXPLICIT=0
DOWNLOAD_FAILED=0
ASSUME_YES=0
CHECK_ONLY=0

# ---------------------------------------------------------------------------
# Output helpers. Everything to stderr so stdout stays pipeable.
# Colour only when stderr is a terminal.
# ---------------------------------------------------------------------------
if [ -t 2 ]; then
  C_B=$'\033[1m'; C_R=$'\033[31m'; C_Y=$'\033[33m'; C_G=$'\033[32m'; C_0=$'\033[0m'
else
  C_B=""; C_R=""; C_Y=""; C_G=""; C_0=""
fi
say()  { printf '%s==>%s %s\n'   "$C_B" "$C_0" "$*" >&2; }
ok()   { printf '   %sok%s   %s\n' "$C_G" "$C_0" "$*" >&2; }
warn() { printf '%sWARN:%s %s\n' "$C_Y" "$C_0" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "$C_R" "$C_0" "$*" >&2; exit 1; }
rule() { printf -- '---------------------------------------------------------------\n' >&2; }

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --large)             MODE="large"; MODE_WAS_EXPLICIT=1 ;;
    --small)             MODE="small"; MODE_WAS_EXPLICIT=1 ;;
    --no-ner|--no-model) MODE="regex"; MODE_WAS_EXPLICIT=1 ;;
    --check)             CHECK_ONLY=1 ;;
    --yes|-y)            ASSUME_YES=1 ;;
    -h|--help)           sed -n '2,28p' "$0" | sed 's/^#\{1,\} \{0,1\}//'; exit 0 ;;
    *)                   die "Unknown option: $1   (try --help)" ;;
  esac
  shift
done

# ===========================================================================
# PREFLIGHT — check everything before changing anything
# ===========================================================================
say "Preflight"
PREFLIGHT_FAILED=0

# --- operating system, for later advice -----------------------------------
OS="$(uname -s 2>/dev/null || echo unknown)"
ok "platform: $OS $(uname -m 2>/dev/null || true)"

# --- bash version (macOS ships 3.2; this script must stay compatible) ------
ok "bash: ${BASH_VERSION:-unknown}"

# --- interpreter ----------------------------------------------------------
BASE_PY="${PYTHON:-python3}"
if ! command -v "$BASE_PY" >/dev/null 2>&1; then
  PREFLIGHT_FAILED=1
  warn "no '$BASE_PY' on PATH"
  case "$OS" in
    Darwin) warn "  install from https://www.python.org/downloads/macos/ or: brew install python@3.12" ;;
    Linux)  warn "  Debian/Ubuntu: sudo apt install python3 python3-venv python3-pip" ;;
  esac
else
  BASE_PY_PATH="$("$BASE_PY" -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$BASE_PY")"
  PY_VER="$("$BASE_PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo "?")"
  if "$BASE_PY" -c "import sys; sys.exit(0 if sys.version_info >= ($MIN_PY_MAJOR,$MIN_PY_MINOR) else 1)" 2>/dev/null; then
    ok "python: $PY_VER at $BASE_PY_PATH"
  else
    PREFLIGHT_FAILED=1
    warn "python $PY_VER is too old; ${MIN_PY_MAJOR}.${MIN_PY_MINOR}+ required"
  fi

  # --- venv module -------------------------------------------------------
  if "$BASE_PY" -c "import venv" 2>/dev/null; then
    ok "venv module present"
  else
    PREFLIGHT_FAILED=1
    warn "the 'venv' module is missing"
    warn "  Debian/Ubuntu: sudo apt install python3-venv"
  fi

  # --- pip ---------------------------------------------------------------
  if "$BASE_PY" -m pip --version >/dev/null 2>&1; then
    ok "pip present"
  else
    warn "pip not available to $BASE_PY; the venv normally supplies its own"
  fi

  # --- TLS trust store ---------------------------------------------------
  # INFORMATIONAL ONLY — never a preflight failure. History: v0.1.9 hard-
  # failed here when the base interpreter's default store was empty, which
  # permanently blocked macOS python.org builds. That gate was wrong: nothing
  # in this pipeline uses the base interpreter's default store. pip ships its
  # own vendored CA bundle, `spacy download` uses requests+certifi, and the
  # app's API call uses certifi explicitly (model_client, since v0.1.6).
  # certifi is installed into the venv from requirements.txt, so the check
  # that actually matters runs POST-INSTALL against the venv, below.
  TLS_STATUS="$("$BASE_PY" - <<'PYEOF' 2>/dev/null || echo "error|0|unknown"
import ssl
try:
    n = len(ssl.create_default_context().get_ca_certs())
    print(("ok" if n else "empty") + "|%d|" % n + (ssl.get_default_verify_paths().cafile or "none"))
except Exception as exc:
    print("error|0|%s" % exc)
PYEOF
)"
  TLS_STATE="${TLS_STATUS%%|*}"
  TLS_COUNT="$(printf '%s' "$TLS_STATUS" | cut -d'|' -f2)"
  case "$TLS_STATE" in
    ok)
      ok "TLS trust store: $TLS_COUNT root certificate(s)" ;;
    empty)
      ok "TLS: base interpreter's default trust store is empty (normal for"
      say "        python.org builds on macOS). Not a problem: pip and spaCy"
      say "        bring their own CA bundle, and the app uses certifi, which"
      say "        this script installs. Verified after install."
      ;;
    *)
      warn "TLS trust store could not be read: ${TLS_STATUS#*|*|} (continuing; certifi is verified after install)" ;;
  esac
fi

# --- project files --------------------------------------------------------
for f in requirements.txt run.sh; do
  if [ -f "$f" ]; then ok "found $f"; else PREFLIGHT_FAILED=1; warn "missing $f — are you in the project root?"; fi
done

# --- disk space -----------------------------------------------------------
FREE_MB="$(df -Pm . 2>/dev/null | awk 'NR==2 {print $4}')"
if [ -n "${FREE_MB:-}" ]; then
  if [ "$FREE_MB" -lt 1200 ]; then
    warn "only ${FREE_MB}MB free here; the large model needs ~${LARGE_MB}MB plus deps"
  else
    ok "disk free: ${FREE_MB}MB"
  fi
fi

# --- existing models -------------------------------------------------------
# Probe ONLY the interpreter this project will actually use. Falling back to
# system python3 here would report "en_core_web_lg already installed" for a
# model that lives in an interpreter run.sh never touches — the precise
# confusion this script exists to eliminate.
PRESENT_LARGE=0
PRESENT_SMALL=0
if [ -x "$VENV_DIR/bin/python3" ]; then
  "$VENV_DIR/bin/python3" -c "import importlib.util as u,sys; sys.exit(0 if u.find_spec('$LARGE_MODEL') else 1)" 2>/dev/null && PRESENT_LARGE=1
  "$VENV_DIR/bin/python3" -c "import importlib.util as u,sys; sys.exit(0 if u.find_spec('$SMALL_MODEL') else 1)" 2>/dev/null && PRESENT_SMALL=1
  [ "$PRESENT_LARGE" -eq 1 ] && ok "$LARGE_MODEL present in $VENV_DIR"
  [ "$PRESENT_SMALL" -eq 1 ] && ok "$SMALL_MODEL present in $VENV_DIR"
  [ "$PRESENT_LARGE" -eq 0 ] && [ "$PRESENT_SMALL" -eq 0 ] && ok "no spaCy model in $VENV_DIR yet"
else
  ok "no virtualenv at ./$VENV_DIR yet; it will be created below"
  # Stale activation: $VIRTUAL_ENV survives `rm -rf .venv` and keeps the
  # (.venv) prompt, while python3 silently falls back to the base interpreter.
  # This exact state cost a full debugging session; name it explicitly.
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    warn "your shell says a virtualenv is ACTIVE at:"
    warn "    $VIRTUAL_ENV"
    warn "  but there is no interpreter there. That is a stale activation:"
    warn "  the (.venv) prompt and \$VIRTUAL_ENV outlive a deleted or replaced"
    warn "  venv, and python3 quietly falls back to the system interpreter."
    warn "  Run 'deactivate' (or open a new terminal) before continuing."
  fi
fi

if [ "$PREFLIGHT_FAILED" -eq 1 ]; then
  rule
  die "Preflight failed. Nothing was installed or modified. Fix the items above and re-run."
fi
say "Preflight passed"

if [ "$CHECK_ONLY" -eq 1 ]; then
  say "--check requested; stopping here. Nothing was installed."
  exit 0
fi

# ===========================================================================
# MODE SELECTION
# ===========================================================================
if [ -z "$MODE" ]; then
  # Prompt whenever stdin can supply an answer — a terminal, or a pipe/heredoc
  # from a script. Only --yes, or stdin closed entirely, skips the question.
  # Silently ignoring a piped answer and installing 400MB instead would be its
  # own kind of rude.
  if [ "$ASSUME_YES" -eq 1 ]; then
    if   [ "$PRESENT_LARGE" -eq 1 ]; then MODE="large"
    elif [ "$PRESENT_SMALL" -eq 1 ]; then MODE="small"
    else MODE="large"; fi
    say "--yes given; selecting: $MODE"
  else
    rule
    printf 'Detection mode:\n\n' >&2
    printf '  1) Regex only        email, phone, URL. No download, no model.\n' >&2
    printf '                       Names and locations are NOT detected.\n\n' >&2
    printf '  2) NER + small       adds names and locations.\n' >&2
    printf '                       %s, ~%sMB download, ~200MB RAM. Weaker on names.\n\n' "$SMALL_MODEL" "$SMALL_MB" >&2
    printf '  3) NER + large       adds names and locations.\n' >&2
    printf '                       %s, ~%sMB download, ~1GB RAM. Best recall. [default]\n\n' "$LARGE_MODEL" "$LARGE_MB" >&2
    rule
    printf 'Choose 1, 2 or 3 [3]: ' >&2
    if read -r REPLY; then
      printf '\n' >&2
    else
      REPLY=""
      printf '\nno input available; taking the default\n' >&2
    fi
    case "${REPLY:-3}" in
      1)    MODE="regex" ;;
      2)    MODE="small" ;;
      3|"") MODE="large" ;;
      *)    die "Invalid choice: '$REPLY'. Re-run and choose 1, 2 or 3, or pass --small / --large / --no-ner." ;;
    esac
  fi
fi

case "$MODE" in
  regex) MODEL=""            ; say "Mode: regex only" ;;
  small) MODEL="$SMALL_MODEL"; say "Mode: NER with $SMALL_MODEL" ;;
  large) MODEL="$LARGE_MODEL"; say "Mode: NER with $LARGE_MODEL" ;;
  *)     die "Internal error: unknown mode '$MODE'" ;;
esac

# ===========================================================================
# INSTALL
# ===========================================================================
if [ ! -x "$VENV_DIR/bin/python3" ]; then
  say "Creating virtualenv in $VENV_DIR"
  "$BASE_PY" -m venv "$VENV_DIR" || die "venv creation failed in $PWD"
else
  say "Reusing existing virtualenv in $VENV_DIR"
fi

PY="$PWD/$VENV_DIR/bin/python3"
[ -x "$PY" ] || die "Expected interpreter not found at $PY"
say "All installs use: $PY"

say "Upgrading pip"
"$PY" -m pip install --quiet --upgrade pip || warn "pip self-upgrade failed; continuing"

say "Installing requirements.txt"
"$PY" -m pip install --quiet -r requirements.txt || die "Dependency install failed. Re-run verbosely to see why:
    $PY -m pip install -r requirements.txt"

MODEL_INSTALLED=0
if [ -n "$MODEL" ]; then
  if "$PY" -c "import importlib.util as u,sys; sys.exit(0 if u.find_spec('$MODEL') else 1)" 2>/dev/null; then
    say "$MODEL already present in this virtualenv — not downloading"
    MODEL_INSTALLED=1
  else
    say "Downloading $MODEL"
    if "$PY" -m spacy download "$MODEL"; then
      MODEL_INSTALLED=1
    else
      warn "spaCy model download failed."
      DOWNLOAD_FAILED=1
      "$PY" - <<'PYEOF' >&2 || true
import ssl, sys
print("")
print("  Diagnostics:")
print("    interpreter    : %s" % sys.executable)
try:
    import certifi
    print("    certifi        : present at %s" % certifi.where())
except ImportError:
    print("    certifi        : NOT importable in this interpreter")
p = ssl.get_default_verify_paths()
print("    openssl cafile : %s" % (p.cafile or "none"))
print("    openssl capath : %s" % (p.capath or "none"))
try:
    n = len(ssl.create_default_context().get_ca_certs())
    print("    trusted roots  : %d" % n)
    if n == 0:
        print("")
        print("    Zero trusted roots. On macOS python.org builds, run once:")
        print("      /Applications/Python 3.x/Install Certificates.command")
except Exception as exc:
    print("    trust store    : unreadable: %s" % exc)
print("")
PYEOF
      warn "Falling back to regex-only so the app still starts."
      MODE="regex"; MODEL=""
      if [ "$MODE_WAS_EXPLICIT" -eq 1 ]; then
        # The user named a model on the command line. Quietly delivering a
        # weaker configuration than the one asked for is how this project
        # arrived at "tested and working" claims that were not true.
        rule
        warn "You asked for a model explicitly and it could not be installed."
        warn "The environment is usable in REGEX-ONLY mode (.env has"
        warn "BLACKJET_USE_NER=0), but names and locations will NOT be"
        warn "detected. Re-run ./setup.sh once the download works."
        rule
      fi
    fi
  fi
fi

# ===========================================================================
# .env — must agree with what was installed, or Presidio downloads behind us
# ===========================================================================
if [ ! -f .env ]; then
  if [ -f .env.example ]; then cp .env.example .env; say "Created .env from .env.example"
  else : > .env; say "Created empty .env"; fi
else
  say ".env exists; updating only the keys this script owns"
fi

set_env_key() {
  key="$1"; val="$2"; tmp=".env.setup.$$"
  grep -v "^[[:space:]]*#\{0,1\}[[:space:]]*${key}=" .env > "$tmp" 2>/dev/null || : > "$tmp"
  printf '%s=%s\n' "$key" "$val" >> "$tmp"
  mv "$tmp" .env
  ok "$key=$val"
}

if [ "$MODE" = "regex" ]; then
  set_env_key BLACKJET_USE_NER 0
else
  set_env_key BLACKJET_USE_NER 1
  set_env_key BLACKJET_SPACY_MODEL "$MODEL"
fi

# ===========================================================================
# VERIFY — in the interpreter run.sh will actually use
# ===========================================================================
say "Verifying"
"$PY" - "$MODE" "$MODEL" <<'PYEOF' || die "Verification failed; the environment is not usable."
import importlib.util, sys
mode, model = sys.argv[1], sys.argv[2]
missing = []
for mod, label in (("presidio_analyzer","presidio-analyzer"),("spacy","spacy"),
                   ("pypdf","pypdf"),("certifi","certifi")):
    present = importlib.util.find_spec(mod) is not None
    print("   %s  %s" % ("ok  " if present else "MISS", label))
    if not present:
        missing.append(label)

# The TLS check that actually matters: the app builds its HTTPS context from
# certifi (model_client._ssl_context), so verify THAT loads roots — not the
# base interpreter's default store, which is legitimately empty on macOS
# python.org builds.
try:
    import ssl, certifi
    n = len(ssl.create_default_context(cafile=certifi.where()).get_ca_certs())
    if n:
        print("   ok    TLS via certifi: %d root certificate(s)" % n)
    else:
        print("   MISS  TLS via certifi: bundle loaded but contains 0 roots")
        missing.append("certifi roots")
except Exception as exc:
    print("   MISS  TLS via certifi: %s" % exc)
    missing.append("certifi TLS context")
if mode == "regex":
    print("   ok    regex-only mode: no spaCy model needed")
elif importlib.util.find_spec(model):
    print("   ok    %s" % model)
else:
    print("   MISS  %s" % model)
    missing.append(model)
if missing:
    print("\n   Missing: %s" % ", ".join(missing))
    sys.exit(1)
PYEOF

chmod +x run.sh 2>/dev/null || true

rule
if [ "$DOWNLOAD_FAILED" -eq 1 ]; then
  warn "Setup finished with the model download UNRESOLVED."
else
  say "Setup complete."
fi
if [ "$MODE" = "regex" ]; then
  say "Mode: regex only. Names and locations will NOT be detected."
else
  say "Mode: NER with $MODEL."
fi
say "Model backend defaults to a local Ollama (AI_PROVIDER=ollama, no key needed)."
say "Override it in .env with AI_PROVIDER / AI_PROVIDER_URL / AI_PROVIDER_MODEL,"
say "then start with:  ./run.sh"
say "(With no reachable endpoint the app still runs detection and tokenization;"
say " the model round trip is skipped and reported as skipped.)"

[ "$DOWNLOAD_FAILED" -eq 1 ] && exit 1
exit 0
