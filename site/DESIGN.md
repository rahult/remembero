# Ledger — the Remembero site design system

Status: 2026-09-18. Ledger replaces the previous true-white/cobalt system. It exists
to walk a **first-time visitor with zero context** through what Remembero is —
problem → idea → try → evidence — under one hard boundary: **no trained model
weights are ever hosted, served, or required by this site.** The system's job is
to make determinism, provenance, and honest measurement visible on every surface,
and to keep internal vocabulary (reader generations, benchmark jargon) off
visitor-facing pages; the measured detail lives at `/research`, written in plain
language.

## Surfaces

Two worlds, never blended within a component:

- **Paper** — warm off-white (`--paper #f8f5ee`) with a faint ultramarine graph grid
  (`--grid-line`) on hero surfaces. The evidence canvas: anything a visitor reads
  and verifies. Cards lift to `--paper-raised`; insets recess into `--soft`.
- **Ink** — deep navy (`--ink #131c2e`) with raised panels (`--ink-soft`,
  `--ink-raised`) and rules (`--line-dark`). The machinery chrome: labs, the IDE,
  sections about engines and models.

Sections alternate paper → ink to pace the page. The playground IDE sits on paper
with ink chrome, inheriting the same tokens.

## Color is semantic, never decorative

| Role | Token (paper) | On ink | Meaning |
| --- | --- | --- | --- |
| Execution | `--blue #2440d2` | `--blue-on-ink #93a8ff` | links, queries, primary actions, interactive states |
| Provenance | `--amber #b96f00` | `--amber-on-ink #ffb84d` | source chains, citations, stamps of record, rules |
| Verdict | `--green #157a4b` | same hue, soft washes | proven answers, passing checks, deltas gained |
| Refusal | `--red #b33131` | same | wrong arms, refused writes, the failure list |

Washes (`--blue-wash`, `--amber-soft`, `--green-soft`, `--red-soft`) tint paper
panels; never place hue-on-hue. Text on paper is `--text`/`--muted`; on ink
`--on-ink`/`--on-ink-muted`.

## Typography

- **Fraunces** (`--serif`, normal + italic) — display headlines, section titles,
  lab titles, the brand wordmark, and every *answer*. Answers are italic; `em`
  accents inside headlines are italic and take the section's accent (amber on
  paper, amber-on-ink on ink; ultramarine inside the hero).
- **Geist** (`--sans`) — body copy and controls.
- **Geist Mono** (`--mono`) — data of every kind: scores, dates, deltas, queries,
  notes blocks, eyebrows, footers, stamps. Numbers are always mono and
  tabular (`font-variant-numeric: tabular-nums`).

Type pairs one serif voice (claims and answers) against one mono voice (evidence).
If a string is a measurement, it is mono — no exceptions.

## Signature components

- **Stamp** (`.stamp`) — a slightly rotated, double-bordered mono label that marks
  execution provenance: `Proof-carrying`, `Replay`, `Not served here`,
  `Deterministic · live`, `Live · 0 models`. Green = verified here and now; amber =
  recorded elsewhere, honestly labeled; ink = policy boundary. Stamps are the system's
  honesty device: anything a model produced wears one.
- **Knowledge card** — every extracted fact renders English-first: a plain-language
  headline ("Your Chicago flight is Saturday morning"), the source sentence in italic,
  and the formal atom demoted to a small muted `machine form:` line. Predicate
  syntax never leads; it appears only as a footnote that teaches the mapping.
- **Proof card** (`.hero-proof`) — the site's opening artifact and recurring motif,
  built on a changed-flight story because everyone has felt this failure:
  *You said* (the raw sentence, serif italic) → *Remembero extracted* (knowledge
  cards) → *You asked* (question, with the query as "the machine sees") → the serif
  answer with its because-chain, where the stale fact appears **struck through** and
  marked superseded → a red "Why this is hard" strip naming the failure a
  similarity-based assistant would make → a source row stamped "Text in · knowledge
  out · proof attached." Rows reveal in a staggered flow on load (collapsed under
  reduced motion). The sixty-second demo carries the derivation half of the thesis:
  its rule derives "1:1" from "meeting with your manager" — a word that never
  appeared in the source text.
- **Paragraph ladder** (`paragraph-ladder.tsx` + module CSS) — the motif at rising
  complexity: one realistic team-update paragraph taken apart in four layers —
  facts (replayed writer reading), dates (resolved live by code), logic (arithmetic
  and a conditional rule, live), and change (validity timelines, live). Each layer
  stamps its execution boundary and keeps English first with machine forms demoted.
  The same text→knowledge shape repeats as the `.extract-strip` on `/research` and
  as the extraction stages in the demo widget and both pipeline labs.
- **First-proof demo** (`first-proof-demo.tsx` + module CSS) — the sixty-second loop
  as a stepped widget: say it → store it → rule it → ask it, each stage revealing on
  click, the final stage a real engine call with the proof chain and a measured
  duration. The newcomer's first taste of the product, with zero models.
- **Writer–reader lab** (`/labs/writer-reader`) — the whole loop on messy history:
  replayed writer claims, then live deterministic resolution into validity
  timelines (current / superseded / ended rows with interval rules), then four
  hard questions whose paired arms show raw-chat failure against structured
  success — including an honest "not in memory" verdict chip.
- **Note block** (`.note-block`) — the computed-notes artifact: mono, amber left
  rule, header stating `0 model calls`, sentences quoted per line. Used on
  `/research` and generated live in the reading-recall lab.
- **Delta ledger** (`.delta-ledger`) — research results as ruled rows: mechanism in
  plain language, mono delta, green chip. Every table names its grader and pairing;
  rows in one table share one grader.
- **Miss list** (`.miss-list`, `.miss-list-ink`) — failures stated with red ✗
  markers before the fix is argued. Negative results stay visible.
- **Metric row** (`.metric-row`) — three big mono numbers with plain captions, the
  homepage's evidence teaser.
- **Lab card** (`.lab-card`) — workbench cards; paper-raised with amber keyline on
  paper, ink-raised on dark sections; ultramarine (`data-kind="model"`) where an
  optional model can be loaded.
## Motion

Transitions are 160–220ms `--ease-out` on color, border, and small translations
(≤4px) only. Nothing animates data. `prefers-reduced-motion` collapses all
transitions site-wide.

## The no-served-models rule

1. No route ships, fetches, or executes Remembero's fine-tuned weights. Claims
   about trained models live on `/research`, framed as research results
   ("measured, never served"), and link to the runs in `docs/research/`.
2. Where model output appears it is either (a) a clearly stamped replay of a
   recorded run, or (b) the optional third-party Hermes 7B the visitor's own
   browser loads via WebLLM behind an explicit action.
3. Every lab states its execution boundary on screen ("no model executes on this
   page", "Live · 0 models", "Model calls use a labeled simulator").
4. Visitor-facing copy carries no internal version jargon — no reader or writer
   generations, no unexplained benchmark names. `/research` explains each term
   in plain language before using it.
5. The tests enforce this: lab and demo sources are swept for `fetch`/storage/
   API-key use, the export assertions require the boundary copy on every lab
   page, and the homepage and lab renders are checked for version jargon.

## Voice

Direct, technically literate, comfortable publishing negative results. Numbers
always carry their grader and protocol, stated in plain words ("one automated
grader, named"; "run-to-run noise of about ±7"). "Unknown" and "replayed" are
stated, never smoothed over.
