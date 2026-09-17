# Ledger — the Remembero site design system

Status: 2026-09-18. Ledger replaces the previous true-white/cobalt system. It exists
to showcase proof-carrying memory and the research that built it, under one hard
boundary: **no trained model weights are ever hosted, served, or required by this
site.** The system's job is to make determinism, provenance, and honest measurement
visible on every surface.

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
  `Deterministic · live`. Green = verified here and now; amber = recorded
  elsewhere, honestly labeled; ink = policy boundary. Stamps are the system's
  honesty device: anything a model produced wears one.
- **Proof card** (`.hero-proof`) — question → query → answer → because-chain →
  source row, with an amber top rule and the stamp overhanging the border.
- **Note block** (`.note-block`) — the computed-notes artifact: mono, amber left
  rule, header stating `0 model calls`, sentences quoted per line. Used on the
  homepage and generated live in the reading-recall lab.
- **Delta ledger** (`.delta-ledger`) — research results as ruled rows: mechanism,
  plain-language detail, mono delta, green chip. Every table names its judge and
  pairing; rows in one table share one judge.
- **Miss list** (`.miss-list`) — failures stated with red ✗ markers before the
  fix is argued. Negative results stay visible.
- **Lab card** (`.lab-card`) — paper-raised workbench cards; amber keyline for
  deterministic labs, ultramarine (`data-kind="model"`) where an optional model
  can be loaded.

## Motion

Transitions are 160–220ms `--ease-out` on color, border, and small translations
(≤4px) only. Nothing animates data. `prefers-reduced-motion` collapses all
transitions site-wide.

## The no-served-models rule

1. No route ships, fetches, or executes Remembero's fine-tuned weights. Claims
   about trained models link to measured runs in `docs/research/` and are framed
   as research results ("Measured, never served"), never as site functionality.
2. Where model output appears it is either (a) a clearly stamped replay of a
   recorded run, or (b) the optional third-party Hermes 7B the visitor's own
   browser loads via WebLLM behind an explicit action.
3. Every lab states its execution boundary on screen ("no model executes on this
   page", "Model calls use a labeled simulator").
4. The tests enforce this: lab sources are swept for `fetch`/storage/API-key use,
   and the export assertions require the boundary copy on every lab page.

## Voice

Direct, technically literate, comfortable publishing negative results. Numbers
always carry their judge and protocol. "Unknown" and "replayed" are stated, never
smoothed over.
