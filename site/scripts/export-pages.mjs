import { copyFile, cp, mkdir, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const siteRoot = resolve(import.meta.dirname, "..");
const clientRoot = resolve(siteRoot, "dist/client");
const pagesRoot = resolve(siteRoot, "dist/pages");
const workerUrl = pathToFileURL(resolve(siteRoot, "dist/server/index.js"));
workerUrl.searchParams.set("static-export", `${process.pid}-${Date.now()}`);

const origin = (process.env.SITE_ORIGIN ?? "https://remembero.rahultrikha.com").replace(
  /\/$/,
  "",
);
const originUrl = new URL(origin);

const directionComment = `<!--
THESIS: A first-time visitor with zero context walks problem → idea → try → evidence; the sixty-second live demo proves the mechanism before any vocabulary is needed, labs deepen it, /research carries the measured story, and no trained weights are ever served.
OWN-WORLD: Ledger system — warm paper evidence canvas with a faint graph grid, deep ink chrome, ultramarine execution, amber provenance, green verdicts; Geist controls, mono data, Fraunces display and answers.
STORY: Hero promise with one proof card, then the forgets/misremembers problem, the store-rule-ask idea beside a real in-page engine demo, four workbenches on ink with the IDE showcase, a three-metric evidence teaser, and the models-translate-rules-decide boundary; /research holds the full de-jargonized measurement story.
FIRST VIEWPORT: Editorial hero on graph paper with one proof-carrying answer card, a stamped provenance seal, and the no-weights-served boundary line.
FORM: Progressive editorial product site with a live demo widget, a try section of four workbenches, a separate evidence page at /research, three labs and the SQLite + Datalog IDE at /playground/.
BOUNDARY: The trained reader/writer models are never hosted, served, or required here; claims about them link to measured runs in docs/research/ and use no internal version jargon on visitor surfaces.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md
-->`;

const { default: worker } = await import(workerUrl.href);

async function render(pathname) {
  const response = await worker.fetch(
    new Request(new URL(pathname, origin), {
      headers: {
        accept: "text/html",
        host: originUrl.host,
        "x-forwarded-proto": originUrl.protocol.replace(":", ""),
      },
    }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
  if (!response.ok) {
    throw new Error(`static render for ${pathname} failed with HTTP ${response.status}`);
  }
  return (await response.text()).replace(/(<body\b[^>]*>)/i, `$1${directionComment}`);
}

const [homeHtml, playgroundHtml, chatMemoryHtml, groundedAgentHtml, readingRecallHtml, researchHtml, agentHarnessHtml] = await Promise.all([
  render("/"),
  render("/playground"),
  render("/labs/chat-memory"),
  render("/labs/grounded-agent"),
  render("/labs/reading-recall"),
  render("/research"),
  render("/guides/agent-harness"),
]);
if (
  !homeHtml.includes("Memory you") ||
  !homeHtml.includes('href="/playground"') ||
  !homeHtml.includes('href="/labs/chat-memory"') ||
  !homeHtml.includes('href="/labs/grounded-agent"') ||
  !homeHtml.includes('href="/labs/reading-recall"') ||
  !homeHtml.includes('href="/research"') ||
  !homeHtml.includes('href="/guides/agent-harness"')
) {
  throw new Error("static homepage is missing the product story, labs, or playground navigation");
}
if (
  !homeHtml.includes("AI that forgets") ||
  !homeHtml.includes("Write it down.") ||
  !homeHtml.includes("Four workbenches.") ||
  !homeHtml.includes("We measured") ||
  !homeHtml.includes("ran with zero models")
) {
  throw new Error("static homepage is missing the progressive narrative sections");
}
if (!researchHtml.includes("Every claim on this site") || !researchHtml.includes("Not served here")) {
  throw new Error("static research page is missing the evidence story or the no-served-models boundary");
}
if (!playgroundHtml.includes("SQLite + Datalog IDE") || !playgroundHtml.includes('id="playground"')) {
  throw new Error("static playground is missing the SQLite IDE bundle");
}
if (!chatMemoryHtml.includes("Same database. Same model. Different powers.") || !chatMemoryHtml.includes("Shared SQLite")) {
  throw new Error("static chat memory lab is missing its comparison experience");
}
if (!groundedAgentHtml.includes("Let the model propose") || !groundedAgentHtml.includes("Grounded agent")) {
  throw new Error("static grounded agent lab is missing its decision experience");
}
if (
  !readingRecallHtml.includes("no model executes on this page") ||
  !readingRecallHtml.includes("COMPUTED NOTES — written by code, not a model") ||
  !readingRecallHtml.includes("How long after moving to the Marina did I run my first 10K?")
) {
  throw new Error("static reading recall lab is missing its deterministic pipeline");
}
if (!agentHarnessHtml.includes("Add proof-carrying memory") || !agentHarnessHtml.includes("Wire one narrow tool loop")) {
  throw new Error("static agent harness guide is missing its integration contract");
}

await rm(pagesRoot, { recursive: true, force: true });
await mkdir(pagesRoot, { recursive: true });
await cp(clientRoot, pagesRoot, { recursive: true });
await writeFile(resolve(pagesRoot, "index.html"), homeHtml);
await copyFile(resolve(pagesRoot, "index.html"), resolve(pagesRoot, "404.html"));
await mkdir(resolve(pagesRoot, "playground"), { recursive: true });
await writeFile(resolve(pagesRoot, "playground", "index.html"), playgroundHtml);
await mkdir(resolve(pagesRoot, "labs", "chat-memory"), { recursive: true });
await writeFile(resolve(pagesRoot, "labs", "chat-memory", "index.html"), chatMemoryHtml);
await mkdir(resolve(pagesRoot, "labs", "grounded-agent"), { recursive: true });
await writeFile(resolve(pagesRoot, "labs", "grounded-agent", "index.html"), groundedAgentHtml);
await mkdir(resolve(pagesRoot, "labs", "reading-recall"), { recursive: true });
await writeFile(resolve(pagesRoot, "labs", "reading-recall", "index.html"), readingRecallHtml);
await mkdir(resolve(pagesRoot, "research"), { recursive: true });
await writeFile(resolve(pagesRoot, "research", "index.html"), researchHtml);
await mkdir(resolve(pagesRoot, "guides", "agent-harness"), { recursive: true });
await writeFile(resolve(pagesRoot, "guides", "agent-harness", "index.html"), agentHarnessHtml);
await writeFile(resolve(pagesRoot, ".nojekyll"), "");

console.log(`Exported GitHub Pages site for ${origin}`);
