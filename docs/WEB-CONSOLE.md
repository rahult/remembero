# Local personal knowledge web console

Remembero 0.53 adds a modern local web interface for testing the complete personal knowledge
workflow against the real store, rule engine, proofs, search, health report, and graph.
It is a product surface over Remembero—not a mocked dashboard.

## Start

```bash
npm run web:dev
```

Open `http://127.0.0.1:4318`. The development server uses one origin for the API and Vite
client, so mutations do not require cross-origin access.

For the production build:

```bash
npm run web
```

The installed `remembero-web` binary opens your **real memory root** (`REMBERO_HOME`,
default `~/.rembero/memory`, namespace `default`) — the same store the CLI and MCP server
use. Demo mode is an explicit opt-in: pass `--demo` (or set `REMBERO_WEB_DEMO=true`) to
use the `.rembero-web/` sandbox instead, where a sourced Personal demo workspace is seeded
only when that sandbox is empty. The repo's `npm run web` and `npm run web:dev` scripts
run in demo mode so evaluating the project never touches real memory. Set
`REMBERO_WEB_SEED_DEMO=false` for an empty sandbox, point `REMBERO_WEB_ROOT` at another
explicit directory, or pick a namespace with `REMBERO_WEB_NAMESPACE`.

## Real use case

The supplied workspace models an Atlas project briefing:

- Rahul owns Atlas and Maya contributes;
- vendor security review blocks Atlas;
- Rahul promised Maya an update;
- three reviewed rules derive collaboration, follow-up, and project risk.

Guided questions such as “Who is collaborating on Atlas?” execute exact deterministic
Datalog locally and return real bindings, sourced rule proofs, and a proof graph. “What
gift does Maya want?” intentionally returns a non-answer with clearly separated related
knowledge.

The **Documents** workspace adds four real public PDFs: English and Spanish IRS W-9 forms,
the MathBridge arXiv paper, and the UN multilingualism publication. Each original PDF and
selected rendered page has a recorded SHA-256 digest, source URL, page count, retrieval time,
and rights note. The actual page image sits beneath selectable extraction coordinates.

PDF text-layer regions are human reviewed and materialized into separately sourced accepted
facts or non-authoritative proposed claims. Guided questions then execute through the ordinary
deterministic query and proof engine. Selecting a proof source highlights the exact region on
the real page. Each PDF uses an isolated namespace, so switching documents cannot carry stale
page, claim, or proof state across the boundary.

The official Unlimited-OCR adapter is wired and its real-page attempt is frozen, but the
provider currently rejects inference because the available ZeroGPU quota is zero. The UI
labels that operational block and does not display a model-quality score. Raw regions and
proposed claims remain evidence only; only explicit accepted facts and reviewed rules
participate in proof.

The workspace also exposes a **Memorg memory** export. It packages the four source documents,
20 page regions, accepted/proposed claims, reviewed rules, and question contracts into a
parent-first, content-addressed import plan while preserving the same authority boundary. See
[DOCUMENT-MEMORG-EXPORT.md](DOCUMENT-MEMORG-EXPORT.md).

Custom questions use the ordinary model-assisted recall pipeline when `LLM_API_KEY` is
configured. The UI labels this boundary; guided questions, exact evidence, local search,
health, structured capture, and explicit graph browse remain model-free.

## Product surfaces

- **Ask** — guided or custom recall, canonical query, supported answer, proof claims,
  authored rules, durable sources, and related discovery for non-answers.
- **Documents** — real rendered pages, provenance hashes, evidence regions, reviewed claims,
  guided recall, deterministic rule proof, and page-region lineage across four public PDFs.
  See the [executable evaluation](DOCUMENT-SHOWCASE-EVALUATION.md).
- **Knowledge** — deterministic local search across fact, rule, and policy text with exact
  score reasons and provenance.
- **Graph** — bounded explicit stored relationships on an accessible SVG canvas and an
  equivalent ordered relationship list. It never presents graph proximity as proof.
- **Rules** — current authored rules and their exact canonical definitions.
- **Add memory** — a structured ground-fact drawer with an explicit relationship and
  durable source statement. It does not silently infer or apply model output.

## Local security boundary

The server binds to `127.0.0.1` by default and refuses every non-loopback host. Mutating
browser requests must be same-origin, API inputs retain the 64 KiB bound, API responses
retain the 16 MiB bound, and production responses set restrictive content, framing,
referrer, and permissions headers. Remote access is not an option in this release because
the personal workspace has no network authentication layer.

Source statements are stored through the existing journal path, including credential
redaction and retry-safe provenance. The browser never receives `LLM_API_KEY`.

The document endpoints retain the same boundary:

- `GET /api/document` reads the source-attributed document and default proof;
- `POST /api/document/parse` validates and idempotently rematerializes accepted claims; and
- `POST /api/document/ask` accepts only a fixed guided question ID.

The two POST routes require the same origin. Unknown document questions, incomplete source
lineage, invalid geometry, and source-manifest reference errors fail closed.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `REMBERO_WEB_ROOT` | Dedicated memory directory | `.rembero-web` |
| `REMBERO_WEB_NAMESPACE` | Selected namespace | `personal` |
| `REMBERO_WEB_HOST` | Bind host | `127.0.0.1` |
| `REMBERO_WEB_PORT` | Bind port | `4318` |
| `REMBERO_WEB_SEED_DEMO` | Seed the demo when empty | `true` |
| `LLM_API_KEY` | Enable custom natural-language questions | unset |

## Visual contract

The desktop and mobile references live in
`docs/assets/rembero-web-concept-desktop.png` and
`docs/assets/rembero-web-concept-mobile.png`. The implementation uses a true-white
editorial evidence desk, deep ink navigation, cobalt actions, amber provenance markers,
one primary proof frame, and code-native controls and graph text.

The document workspace reference is
`docs/assets/rembero-document-intelligence-concept.png`. It extends the same evidence desk
with a document preview, parsed-evidence rail, and recall/proof rail without making the
reference image part of the runtime interface.
