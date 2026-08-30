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
THESIS: The main site sells proof-carrying memory; two labs show the real-life effect; the playground exposes the mechanism.
OWN-WORLD: True white evidence canvas, navy structural chrome, cobalt execution, amber provenance, compact sans controls, mono data, serif answers.
STORY: A visitor compares chat recall and grounded agent behavior in /labs/, then opens /playground/ to inspect the machinery.
FIRST VIEWPORT: Marketing hero with one proof-carrying answer, product promise, and direct Playground and GitHub actions.
FORM: Editorial product site plus two comparison labs and a separate Guided Query Canvas at /playground/.
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

const [homeHtml, playgroundHtml, chatMemoryHtml, groundedAgentHtml, agentHarnessHtml] = await Promise.all([
  render("/"),
  render("/playground"),
  render("/labs/chat-memory"),
  render("/labs/grounded-agent"),
  render("/guides/agent-harness"),
]);
if (
  !homeHtml.includes("Memory you") ||
  !homeHtml.includes('href="/playground"') ||
  !homeHtml.includes('href="/labs/chat-memory"') ||
  !homeHtml.includes('href="/labs/grounded-agent"') ||
  !homeHtml.includes('href="/guides/agent-harness"')
) {
  throw new Error("static homepage is missing the product story, labs, or playground navigation");
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
await mkdir(resolve(pagesRoot, "guides", "agent-harness"), { recursive: true });
await writeFile(resolve(pagesRoot, "guides", "agent-harness", "index.html"), agentHarnessHtml);
await writeFile(resolve(pagesRoot, ".nojekyll"), "");

console.log(`Exported GitHub Pages site for ${origin}`);
