# blackjet

> Stealth Airframes for ferrying PII

**Send a resume through an LLM API without the LLM ever seeing a name, email,
phone number or address.**

blackjet is a small, self-contained demonstration of PII tokenization around a
model API call: personal data is detected and replaced with placeholder tokens
*before* anything leaves your machine, and restored *after* the response comes
back. The map between tokens and real values never crosses the network.

```
Your machine                                          Model endpoint
─────────────────────────────────────────────         ──────────────
Real resume ──► Detect + tokenize ──► <<PERSON_01>> ──►  model
                      │                                    │
                      ▼                                    │
              Rehydration map                              │
              (never leaves)                               │
                      │                                    │
                      ▼                                    ▼
Final output ◄── Rehydrate ◄─────── tokens return ◄────────┘
```

## Quick start

Requires Python 3.10+. No Node, no Docker, no database, no build step.

```bash
git clone <this-repo> blackjet && cd blackjet
chmod +x setup.sh run.sh     # if you unpacked from a zip; git preserves this
./setup.sh                   # venv + dependencies + en_core_web_lg (~600MB)
#   ./setup.sh --small       # ...or en_core_web_sm (~12MB, weaker on names)
#   ./setup.sh --no-model    # ...or dependencies only, regex-only detection
# put your API key in the .env that setup.sh created (AI_PROVIDER_APIKEY,
# or point AI_PROVIDER=ollama at a local/remote instance — see below)
./run.sh
```

`setup.sh` resolves **one** interpreter — `.venv/bin/python3` — and performs
every install with it, dependencies and spaCy model alike. This matters: a bare
`python3 -m spacy download` run from a shell that is no longer inside the
activated venv installs the model into system Python, where `run.sh` cannot see
it. The result is a 600MB model on disk and a startup message insisting no model
is installed. `setup.sh` exists to make that failure impossible; if you install
by hand instead, use `.venv/bin/python3` explicitly and never a bare `python3`.

`run.sh` never downloads anything. Without a spaCy model it starts immediately
and runs regex detection (email, phone, URL); NER — names and locations —
requires the model.

Model load adds several seconds to startup (once, not per request) and roughly
100-300ms per document on CPU. To stay regex-only, set `BLACKJET_USE_NER=0`.

Open <http://127.0.0.1:8080>, pick a sample resume, press **Run**, and compare
the Original / Anonymized / Rehydrated tabs.

Without a configured model backend the app still runs detection →
tokenization → tripwire and states plainly that the round trip was skipped.
Nothing is simulated.

## Configuration

All configuration is environment variables; `run.sh` sources `.env` into the
environment before starting the server — that is how the API key reaches the
app. Defaults in brackets.

| Variable | Purpose |
|---|---|
| `AI_PROVIDER` | preset: `ollama`, `vllm`, `llamacpp`, `lmstudio` (local, keyless); `openai`, `groq`, `together`, `openrouter` (hosted, keyed) [ollama] |
| `AI_PROVIDER_URL` | endpoint URL; bare host, `/v1` base or full path all work, `?token=...` preserved |
| `AI_PROVIDER_APIKEY` | bearer token (required by the hosted presets only) |
| `AI_PROVIDER_MODEL` | model id or tag [mistral:7b for ollama] |
| `BLACKJET_PORT` | listen port [8080] |
| `BLACKJET_HOST` | bind address [127.0.0.1] |
| `BLACKJET_USE_NER` | set `0` to disable the NER tier [1] |
| `BLACKJET_SPACY_MODEL` | spaCy model for NER [en_core_web_lg] |
| `BLACKJET_LOG_LEVEL` | DEBUG/INFO/WARNING/ERROR [INFO] |
| `BLACKJET_MODEL_TIMEOUT` | seconds [600 local / 120 hosted] |

Legacy names (`HOSTED_OLLAMA_URL`, `HOSTED_OLLAMA_AUTH_TOKEN`, `OLLAMA_URL`,
`OLLAMA_HOST`/`OLLAMA_PORT`, `OLLAMA_MODEL`) are honored as fallbacks, so an
existing `.env` keeps working.

## Using a local model (Ollama)

The round trip goes through any endpoint that serves the OpenAI Chat
Completions API, and has been tested with Ollama (mistral, llama, qwen). The
provider may be a local install, a Docker instance, a remote/rented GPU box, or
a hosted endpoint. Ollama is the default and needs no key. Set:

```bash
AI_PROVIDER=ollama
AI_PROVIDER_URL=http://127.0.0.1:11434          # or http://ip:port?token=xxxx
AI_PROVIDER_APIKEY=                              # bearer token if the instance needs one
AI_PROVIDER_MODEL=mistral:7b
```

The app talks to Ollama's OpenAI-compatible `/v1/chat/completions` endpoint
(appended automatically if your URL is just a host). Auth, when your instance
requires it, is sent as `Authorization: Bearer <AI_PROVIDER_APIKEY>`; a
`?token=...` query parameter already in your URL is preserved too, and its
value is redacted from all logs.

**Will your machine cope?** Rough guidance for judging before you pull:

| Model | Disk (Q4) | RAM/VRAM needed |
|---|---|---|
| `mistral:7b` / `llama3.1:8b` | ~4.4–4.7 GB | ~6–8 GB |
| `qwen2.5:32b` (or other ~32B at Q4) | ~20 GB | ~24–32 GB |

Local inference on modest hardware is far slower than an API call, which is
why `BLACKJET_MODEL_TIMEOUT` defaults to 600 s for this backend. Note that the
~cents an API charges per call is the provider's *price*, not a measure of the
compute involved — a local model is not automatically cheap or fast on your
machine.

**Placeholder-token caveat.** The whole design assumes the model reproduces
`<<PERSON_01>>` tokens verbatim. Small local models mangle placeholders far
more often than frontier models; any altered token is surfaced as unmatched
rather than silently dropped. Measure before trusting:
`scripts/measure_token_preservation.py` runs the whole corpus through your
configured backend and reports exact-preservation rates per entity type.

**Security note.** Because the egress URL is configurable, it is by
construction a place a payload could be redirected. The app logs the URL in
effect (token redacted) at every send — check the log pane if in doubt, and
treat `AI_PROVIDER_URL` in a shared `.env` with the same care as a key. Plain
`http://` to a remote instance sends the (tokenized) payload and any URL token
unencrypted; that is fine on a trusted LAN, not across the open internet.

### Troubleshooting

**`CERTIFICATE_VERIFY_FAILED` on the model call.** Python installed from
python.org (common on macOS) ships without CA certificates and does not use the
system keychain. Fix with `pip3 install certifi`, or run
`Install Certificates.command` in your `/Applications/Python 3.x/` folder.

## How detection works

Two tiers, both local:

1. **Deterministic** — regex/validators for emails, phone numbers (including
   non-US formats) and URLs.
2. **NER** — [Microsoft Presidio](https://github.com/microsoft/presidio) with
   a spaCy model, for names and locations in prose. Free, MIT-licensed, runs
   entirely on your machine, CPU-only (no GPU is used). Costs a ~600MB model
   download and ~1GB RAM for `en_core_web_lg`; `en_core_web_sm` is far smaller
   with measurably worse name recall. Not installed by default — see above.

Overlapping findings are resolved longest-match-first so text is never
double-tokenized. Each value becomes a token like `<<PERSON_01>>`; repeated
values reuse their token so the document stays coherent for the model.

## What the code guarantees

- **Fail closed.** If detection errors, nothing is sent — the request aborts.
- **Tripwire.** The outbound payload is re-scanned for every vaulted value
  immediately before the network call; any hit blocks the send.
- **One egress point.** `model_client.py` is the only module that touches the
  network. Auditing what leaves the machine means reading one file.
- **The map cannot leak via the model.** The token↔value map is never part of
  any request, so prompt injection has nothing to exfiltrate.
- **Honest failure.** A token the model alters or invents is reported and left
  visible in the output — never silently dropped or guessed at.
- **Logs are scrubbed.** Every detected value is registered with a redaction
  filter the moment it is found; it cannot appear in any log line afterward.

## What this does not solve

- **Detection is probabilistic.** A missed name goes to the API in the clear.
  Recall must be measured, not assumed — the sample corpus ships with
  hand-labelled ground truth in `data/labels/` for exactly that purpose (a
  measurement harness is deliberately out of scope for this demo).
- **Context re-identifies.** "Sole female VP at a 40-person Reykjavík fintech"
  identifies a person with every name removed. Tokenization narrows exposure;
  it is not anonymity.
- **Extraction artifacts defeat patterns.** `alfred_pennyworth_pm.pdf` in the
  sample corpus has icon glyphs fused onto its phone number by PDF text
  extraction — a deliberate trap showing why real-world recall is lower than
  clean-text benchmarks suggest.

## Layout

```
blackjet/            the application (stdlib HTTP server, no framework)
  loader.py          PDF (text layer + metadata sweep), TXT, JSON Resume → text
  detect.py          tier 1 regex + tier 2 Presidio, overlap resolution
  vault.py           in-memory token↔value map, session-scoped
  pipeline.py        anonymize / tripwire / rehydrate
  model_client.py    the single network egress point
  logging_setup.py   PII redaction filter for all log output
  server.py          HTTP API + static UI
static/index.html    the four-tab UI
data/resumes/        sample corpus (PDF, TXT, JSON Resume)
data/labels/         hand-labelled ground truth for the corpus
```

## Live Demo

A live demo of this project, using the code from this repo is available at:
 [2KSTech's BlackJet Demo Site](https://blackjet.2kstech.fun) 

## Sample corpus

The resume corpus includes the real-life resume of Thomas Alwyn Davis as
`thomasdavis.json`, through his generous permission.  Thomas is the founder of
[JSON Resume](https://jsonresume.org), which is a JSON schema supported by WorkInPilot. 
All other resumes are synthetic; any resemblance to real persons is
coincidental.

## License

Apache-2.0. See [LICENSE](LICENSE).
