#!/usr/bin/env node
// Print a Remembero CLI JSON result as a compact, readable transcript: bindings as lines,
// proofs as trees, and the supporting graph and rule listing left out. Anything it does
// not recognise is printed as indented key/value pairs, so nothing is hidden by mistake.
// Usage: remembero explain '...' | node examples/lib/show.mjs
import { readFileSync } from "node:fs";

const SKIP = new Set(["graph", "rules", "id", "constraintId", "evaluatedQuery", "bindingOrder", "checkpointId", "nodeIds", "segments", "selection", "opId"]);
const text = readFileSync(0, "utf8").trim();
let data;
try { data = JSON.parse(text); } catch { process.stdout.write(text + "\n"); process.exit(0); }

const pad = (n) => "  ".repeat(n);
const term = (v) => (v === null ? "_" : typeof v === "string" ? v : JSON.stringify(v));
const atom = (p) => `${p.negated ? "not " : ""}${p.predicate}(${(p.values ?? p.pattern ?? []).map(term).join(", ")})`;
const day = (ts) => (typeof ts === "string" && ts.startsWith("1970") ? "assumed" : ts?.slice(0, 10) ?? "stored");
const when = (sources) => (Array.isArray(sources) && sources.length ? `  [${sources.length === 1 ? day(sources[0].ts) : `${sources.length} sources`}]` : "");

function proof(p, depth) {
  const out = [`${pad(depth)}${atom(p)}${p.rule ? `  <- rule ${p.rule}` : when(p.sources)}`];
  for (const b of p.because ?? []) out.push(...proof(b, depth + 1));
  return out;
}

function row(r, depth) {
  const out = [];
  if (r.bindings) out.push(`${pad(depth)}${Object.entries(r.bindings).map(([k, v]) => `${k} = ${term(v)}`).join(", ") || "yes"}`);
  for (const p of r.proofs ?? []) { out.push(`${pad(depth + 1)}because`); out.push(...proof(p, depth + 2)); }
  return out;
}

function render(v, depth) {
  const out = [];
  if (Array.isArray(v)) {
    if (v.length === 0) return [`${pad(depth)}(none)`];
    for (const item of v) {
      if (item && typeof item === "object" && "reason" in item && "goal" in item) {
        out.push(`${pad(depth)}${item.reason}: ${item.goal}`);
        const rest = Object.fromEntries(Object.entries(item).filter(([k]) => !["reason", "goal", "bindings", "id"].includes(k)));
        out.push(...render(rest, depth + 1));
      } else if (item && typeof item === "object" && ("bindings" in item || "proofs" in item)) out.push(...row(item, depth));
      else if (item && typeof item === "object" && !Array.isArray(item) && Object.values(item).every((x) => x === null || typeof x !== "object"))
        out.push(`${pad(depth)}${Object.entries(item).map(([k, x]) => `${k} = ${term(x)}`).join(", ")}`);
      else if (item && typeof item === "object" && "predicate" in item) out.push(...proof(item, depth));
      else if (item && typeof item === "object") { out.push(`${pad(depth)}-`); out.push(...render(item, depth + 1)); }
      else out.push(`${pad(depth)}${term(item)}`);
    }
    return out;
  }
  if (v && typeof v === "object") {
    for (const [k, val] of Object.entries(v)) {
      if (SKIP.has(k)) continue;
      if (k === "sources" && Array.isArray(val)) { out.push(`${pad(depth)}sources: ${val.length}`); continue; }
      if (val === null || typeof val !== "object") out.push(`${pad(depth)}${k}: ${term(val)}`);
      else if (Array.isArray(val) && val.length === 0) out.push(`${pad(depth)}${k}: (none)`);
      else if (Array.isArray(val) && val.every((x) => typeof x !== "object")) out.push(`${pad(depth)}${k}: ${val.map(term).join("; ")}`);
      else { out.push(`${pad(depth)}${k}:`); out.push(...render(val, depth + 1)); }
    }
    return out;
  }
  return [`${pad(depth)}${term(v)}`];
}

process.stdout.write(render(data, 0).join("\n") + "\n");
