import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render(path = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request(`http://localhost${path}`, { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the Remembero marketing homepage with lab and playground navigation", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Remembero — Memory you can reason with<\/title>/i);
  assert.match(html, /Memory you(?:<br\/>|\s)+can <em>reason<\/em> with\./);
  assert.match(html, /Remembero extracted/);
  assert.match(html, /Your Chicago flight is Saturday morning/);
  assert.match(html, /Friday 6pm/);
  assert.match(html, /Why this is hard:/);
  assert.match(html, /similarity-based assistant hands it back/);
  assert.match(html, /Do I have a 1:1 this week\?/);
  assert.match(html, /a meeting with your manager is a one-on-one/);
  assert.match(html, /machine form:/);
  assert.match(html, /the machine sees/);
  assert.match(html, /Text in · knowledge out · proof attached/);
  assert.match(html, /One paragraph,(?:<br\/>|\s)+taken <em>apart\.<\/em>/);
  assert.match(html, /The facts it states/);
  assert.match(html, /Nadia takes over the rollout/);
  assert.match(html, /AI that forgets/);
  assert.match(html, /Write it down\./);
  assert.match(html, /Four workbenches\./);
  assert.match(html, /We measured <em>everything\.<\/em>/);
  assert.match(html, /318 → 425/);
  assert.match(html, /ran with zero models/);
  assert.match(html, /Models translate\./);
  assert.match(html, /See it work — 60 seconds/);
  assert.match(html, /Build agents that can/);
  assert.match(html, /href="#problem"/);
  assert.match(html, /href="#idea"/);
  assert.match(html, /href="#try"/);
  assert.match(html, /href="#evidence"/);
  assert.match(html, /href="#demo"/);
  assert.match(html, /href="\/research"/);
  assert.match(html, /href="\/playground"/);
  assert.match(html, /href="\/labs\/chat-memory"/);
  assert.match(html, /href="\/labs\/grounded-agent"/);
  assert.match(html, /href="\/labs\/reading-recall"/);
  assert.match(html, /href="\/labs\/writer-reader"/);
  assert.match(html, /Raw text in\./);
  assert.match(html, /href="\/guides\/agent-harness"/);
  assert.match(html, /http:\/\/localhost(?::3000)?\/og\.png/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
  assert.doesNotMatch(html, /Reader v7|reader v4|Gemma 4 E4B, distilled/);
});

test("server-renders the de-jargonized research page with the measured evidence", async () => {
  const response = await render("/research");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /<title>Remembero Research — The evidence, measured<\/title>/i);
  assert.match(html, /Every claim on this site/);
  assert.match(html, /the writer extracts/);
  assert.match(html, /Norsk Dental became a client/);
  assert.match(html, /client_of\(norsk_dental, us\) · from 2026-01-11/);
  assert.match(html, /LongMemEval-S/);
  assert.match(html, /318 <i>→ 359<\/i>/);
  assert.match(html, /425<i>\/500<\/i>/);
  assert.match(html, /teacher 440/);
  assert.match(html, /Not served here/);
  assert.match(html, /ships no model weights/);
  assert.match(html, /The translator/);
  assert.match(html, /The answerer/);
  assert.match(html, /one automated grader \(GPT-4o\)/);
  assert.match(html, /one grader \(DeepSeek\)/);
  assert.match(html, /remembro-eval\/README\.md/);
  assert.match(html, /docs\/research\/READER-STRUCTURE\.md/);
  assert.match(html, /docs\/research\/EXTRACTION-BENCH\.md/);
  assert.match(html, /href="\/labs\/reading-recall"/);
  assert.match(html, /href="\/playground"/);
  assert.doesNotMatch(html, /Reader v7|reader v4/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("server-renders the writer-reader lab over messy history", async () => {
  const response = await render("/labs/writer-reader");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /<title>Writer–Reader Lab — Raw text in, proven answers out<\/title>/i);
  assert.match(html, /Raw text in\. Proven answers out\./);
  assert.match(html, /no model executes on this page/);
  assert.match(html, /Writer &amp; reader replayed/);
  assert.match(html, /No weights served/);
  assert.match(html, /Who is my dentist now — and who did I see immediately before them\?/);
  assert.match(html, /What are we paying for the dashboard tool now, and what did we pay before the switch back\?/);
  assert.match(html, /Who leads the Norsk Dental account today — and is the account still active\?/);
  assert.match(html, /How many people are on Priya/);
  assert.match(html, /STRUCTURED-EVIDENCE NOTES — computed by code/);
  assert.match(html, /hard shape: (?:<!-- -->)?a three-deep update chain/);
  assert.match(html, /hard shape: (?:<!-- -->)?a question never answered anywhere/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("server-renders the deterministic reading recall lab", async () => {
  const response = await render("/labs/reading-recall");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /<title>Reading Recall Lab — The reader contract, model taken out<\/title>/i);
  assert.match(html, /Watch the reading pipeline assemble/);
  assert.match(html, /no model executes on this page/);
  assert.match(html, /Replayed answers/);
  assert.match(html, /No weights served/);
  assert.match(html, /How long after moving to the Marina did I run my first 10K\?/);
  assert.match(html, /How much would I lose if I sell the road bike at my asking price\?/);
  assert.match(html, /When exactly did I move to the Marina\?/);
  assert.match(html, /COMPUTED NOTES — written by code, not a model/);
  assert.match(html, /full text/);
  assert.match(html, /Small reader, notes \+ working — replay/);
  assert.match(html, /docs\/research\/READER-STRUCTURE\.md/);
  assert.doesNotMatch(html, /Reader v7|reader v4/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("server-renders both real-life browser labs", async () => {
  const [chatResponse, agentResponse] = await Promise.all([
    render("/labs/chat-memory"),
    render("/labs/grounded-agent"),
  ]);
  assert.equal(chatResponse.status, 200);
  assert.equal(agentResponse.status, 200);

  const [chatHtml, agentHtml] = await Promise.all([
    chatResponse.text(),
    agentResponse.text(),
  ]);
  assert.match(chatHtml, /<title>Remembero Lab — Same database, different powers<\/title>/i);
  assert.match(chatHtml, /Same database\. Same model\. Different powers\./);
  assert.match(chatHtml, /Shared SQLite/);
  assert.match(chatHtml, /Model calls/);
  assert.match(chatHtml, /Data only/);
  assert.match(chatHtml, /With Remembero/);
  assert.match(chatHtml, /href="\/guides\/agent-harness"/);
  assert.match(chatHtml, /Model calls use a labeled simulator/);

  assert.match(agentHtml, /<title>Grounded Agent Lab — Let the gate show its work<\/title>/i);
  assert.match(agentHtml, /Let the model propose\. Let the gate show its work\./);
  assert.match(agentHtml, /Prompt-only agent/);
  assert.match(agentHtml, /Grounded agent/);
  assert.match(agentHtml, /The model never gets mutation authority\./);
  assert.match(agentHtml, /Model proposals use the deterministic simulator/);
});

test("server-renders the agent harness integration guide", async () => {
  const response = await render("/guides/agent-harness");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(
    html,
    /<title>Remembero Guide — Add proof-carrying memory to an agent harness<\/title>/i,
  );
  assert.match(html, /Add proof-carrying memory without dumping memory into the prompt\./);
  assert.match(html, /Wire one narrow tool loop\./);
  assert.match(html, /recall_explain/);
  assert.match(html, /explain_query/);
  assert.match(html, /propose_memory/);
  assert.match(html, /apply_memory_proposal/);
  assert.match(html, /prepare_semantic_search/);
  assert.match(html, /bench:agent-db:check/);
  assert.match(html, /Gate the database, not the marketing claim\./);
  assert.match(html, /100k facts/);
  assert.match(html, /Million-fact gate/);
  assert.match(html, /2\.15 GiB/);
  assert.match(html, /bench:agent-db:scale/);
  assert.match(html, /bench:agent-db:install:check/);
  assert.match(html, /bench:agent-db:million/);
  assert.match(html, /bench:agent-db:cost/);
  assert.match(html, /bench:memory:external/);
  assert.match(html, /Top retrieval group/);
  assert.match(html, /4 stacks · 100% R@k/);
  assert.match(html, /100% P@k/);
  assert.match(html, /88\.6% P@k/);
  assert.match(html, /\$0\.044 · 118 calls/);
  assert.match(html, /bench:memory:mem0/);
  assert.match(html, /Graphiti retrieval/);
  assert.match(html, /44\.0% R@k/);
  assert.match(html, /\$0\.0379 · 275 calls/);
  assert.match(html, /bench:memory:graphiti/);
  assert.match(html, /LongMemEval-S/);
  assert.match(html, /500 questions/);
  assert.match(html, /83\.27%/);
  assert.match(html, /80\.96%/);
  assert.match(html, /bench:longmemeval/);
  assert.match(html, /bench:longmemeval:semantic/);
  assert.match(html, /Preference Recall@5/);
  assert.match(html, /43\.3 → 73\.3%/);
  assert.match(html, /47\.2%/);
  assert.match(html, /\$0\.000412 avg/);
  assert.match(html, /Restart cache/);
  assert.match(html, /32 → 9 tokens/);
  assert.match(html, /Prewarmed first query/);
  assert.match(html, /412 ms · 9 tokens/);
  assert.match(html, /\$0\.000644 avg/);
  assert.match(html, /href="\/labs\/chat-memory"/);
});

test("labs run real browser-safe tool and policy loops without remote model APIs or persistence surfaces", async () => {
  const [chatClient, chatEngine, chatTools, agentClient, agentEngine, browserModel, recallClient, recallFixture, demoLib, demoClient, writerReaderClient, writerReaderFixture] = await Promise.all([
    readFile(new URL("../app/labs/chat-memory/chat-memory-lab.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/chat-memory-lab.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/chat-memory-tools.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/labs/grounded-agent/grounded-agent-lab.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/grounded-agent-lab.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/browser-language-model.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/labs/reading-recall/reading-recall-lab.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/reading-recall-fixture.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/first-proof-demo.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/first-proof-demo.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/labs/writer-reader/writer-reader-lab.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/writer-reader-fixture.ts", import.meta.url), "utf8"),
  ]);

  assert.match(chatEngine, /from "\.\/engine"/);
  assert.match(chatEngine, /evaluateQuerySpecWithProof/);
  assert.match(chatEngine, /parseProgram/);
  assert.doesNotMatch(chatClient, /FINAL_ANSWER/);
  assert.match(chatClient, /openBrowserDatalogDatabase/);
  assert.match(chatClient, /verifyChatMemorySeed/);
  assert.match(chatClient, /Startup probe:/);
  assert.match(chatClient, /rows verified/);
  assert.match(chatClient, /requestLoadedWebLlmToolCall/);
  assert.match(chatClient, /finalAnswerPrompt/);
  assert.match(chatClient, /includesAll\("procurement", "freeze"\)/);
  assert.match(chatClient, /vendor security review/);
  assert.match(chatClient, /failing premise/);
  assert.match(chatClient, /parseChatToolCall/);
  assert.match(chatClient, /executeChatTool/);
  assert.match(chatClient, /Tool execution failed closed/);
  assert.match(chatClient, /tables · \$\{seed\.rowCount\} rows verified/);
  assert.match(chatTools, /CHAT_MEMORY_SQLITE_VERIFY/);
  assert.match(chatTools, /expected 6 tables and 9 rows/);
  assert.match(chatClient, /Seed data loaded · run the tool loop/);
  assert.match(chatClient, /if \(value === null\) return "not run"/);
  assert.doesNotMatch(chatClient, /sharedSourceData|SOURCE_DATA:/);
  assert.match(chatTools, /CREATE TABLE waits_on/);
  assert.match(chatTools, /CREATE TABLE review_slot/);
  assert.match(chatTools, /root_blocker\(Root\)/);
  assert.match(chatTools, /WITH RECURSIVE chain/);
  assert.match(chatTools, /ROLLBACK TO remembero_gate/);
  assert.match(chatTools, /WHY_NOT_PREMISES/);
  assert.match(chatTools, /name: "Query"/);
  assert.match(chatTools, /lane === "data"/);
  assert.match(chatTools, /normalizedName/);
  assert.match(chatClient, /canonicalized to the sole allowlisted Query tool/);
  assert.match(chatTools, /database\.datalogQuery/);
  assert.match(chatTools, /database\.datalogExplain/);
  assert.match(agentEngine, /from "\.\/engine"/);
  assert.match(agentEngine, /evaluateQuerySpecWithProof/);
  assert.match(agentEngine, /requires_human/);
  assert.match(agentEngine, /proposed_action/);
  assert.match(agentEngine, /action_allowed/);
  assert.match(agentEngine, /action_blocked/);
  assert.match(agentClient, /promptProposalStatus/);
  assert.match(agentClient, /failed closed/);
  assert.match(agentClient, /promptModel\(\s*AGENT_PROMPT,\s*promptOnlyRequest/);
  assert.match(agentClient, /promptModel\(\s*AGENT_PROMPT,\s*groundedRequest/);

  assert.match(browserModel, /LanguageModel/);
  assert.match(browserModel, /availability\(\)/);
  assert.match(browserModel, /!== "available"/);
  assert.match(browserModel, /reason: "unsupported"/);
  assert.match(browserModel, /reason: "not_ready"/);
  assert.match(browserModel, /reason: "runtime_error"/);
  assert.match(browserModel, /Hermes-2-Pro-Mistral-7B-q4f16_1-MLC/);
  assert.match(browserModel, /CreateMLCEngine/);
  assert.match(browserModel, /promptLoadedWebLlm/);
  assert.match(browserModel, /requestLoadedWebLlmToolCall/);
  assert.match(browserModel, /tool_choice/);
  assert.match(browserModel, /tool_calls/);
  assert.match(browserModel, /role: "system",\s*content: systemPrompt/);
  assert.match(browserModel, /role: "user",\s*content: userPrompt/);

  assert.match(recallClient, /buildComputedNotes/);
  assert.match(recallClient, /runPipeline/);
  assert.match(recallFixture, /No model weights are served, fetched, or/);
  assert.match(recallFixture, /previousSaturday/);
  assert.match(recallFixture, /84% to 95%/);
  assert.match(demoLib, /evaluateQuerySpecWithProof/);
  assert.match(demoLib, /parseProgram/);
  assert.match(demoLib, /No model is/);
  assert.match(demoLib, /nothing is sent anywhere/);
  assert.match(demoClient, /runFirstProofDemo/);
  assert.match(demoClient, /Live · 0 models/);
  assert.match(writerReaderFixture, /buildTimelines/);
  assert.match(writerReaderFixture, /WRITER_CLAIMS/);
  assert.match(writerReaderFixture, /retrieveForQuestion/);
  assert.match(writerReaderFixture, /No model weights are served, fetched, or/);
  assert.match(writerReaderFixture, /update chains, flip-backs, effective-dated roles/);
  assert.match(writerReaderClient, /buildTimelines/);
  assert.match(writerReaderClient, /buildReaderNotes/);
  const ladderLib = await readFile(new URL("../lib/paragraph-ladder.ts", import.meta.url), "utf8");
  const ladderClient = await readFile(new URL("../app/paragraph-ladder.tsx", import.meta.url), "utf8");
  assert.match(ladderLib, /resolveDates/);
  assert.match(ladderLib, /computeLogic/);
  assert.match(ladderLib, /buildChange/);
  assert.match(ladderLib, /No model weights are served/);
  assert.match(ladderClient, /ParagraphLadder/);
  assert.match(ladderClient, /machine form:/);

  const hostedLabSource = `${chatClient}\n${chatEngine}\n${chatTools}\n${agentClient}\n${agentEngine}\n${browserModel}\n${recallClient}\n${recallFixture}\n${demoLib}\n${demoClient}\n${writerReaderClient}\n${writerReaderFixture}\n${ladderLib}\n${ladderClient}`;
  assert.doesNotMatch(hostedLabSource, /fetch\s*\(|XMLHttpRequest|WebSocket|EventSource/);
  assert.doesNotMatch(hostedLabSource, /localStorage|sessionStorage|document\.cookie|indexedDB/);
  assert.doesNotMatch(hostedLabSource, /OPENAI_API_KEY|LLM_API_KEY/);
});

test("server-renders the full SQLite IDE on the playground route", async () => {
  const response = await render("/playground");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /<title>Remembero Playground — SQLite \+ Datalog IDE<\/title>/i);
  assert.match(html, /SQLite \+ Datalog IDE/);
  assert.match(html, /Get to the proof/);
  assert.match(html, /Insert SQLite row/);
  assert.match(html, /Who needs a follow-up\?/);
  assert.match(html, /Which tickets need escalation\?/);
  assert.match(html, /Which service is ready to ship\?/);
  assert.match(html, /Who can read the launch plan\?/);
  assert.match(html, /Why this is true/);
  assert.match(html, /Query graph/);
  assert.match(html, /href="\/"[^>]*>Home/);
});

test("playground uses the statically linked SQLite extension and remains browser-contained", async () => {
  const [playground, ide, globalCss, demo, adapter, page, playgroundPage, packageJson, og] = await Promise.all([
    readFile(new URL("../app/playground.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/sqlite-ide.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../lib/ide-demo.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/sqlite-wasm.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/playground/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../public/og.png", import.meta.url)),
  ]);

  assert.match(playground, /^"use client";/);
  assert.match(playground, /<SqliteIde/);
  assert.match(ide, /openBrowserDatalogDatabase/);
  assert.match(ide, /datalogSql/);
  assert.match(ide, /datalogQuery/);
  assert.match(ide, /datalogExplain/);
  assert.match(ide, /Insert SQLite row/);
  assert.match(ide, /Why this is true/);
  assert.match(ide, /Query graph/);
  assert.match(ide, /onClick=\{\(\) => void insertRow\(\)\}/);
  assert.match(ide, /Current browser performance/);
  assert.match(ide, /performance\.now\(\)/);
  assert.doesNotMatch(ide, /className="ide-activity"/);
  assert.match(globalCss, /grid-template-rows: minmax\(140px, auto\)/);
  assert.doesNotMatch(ide, /localStorage|sessionStorage|document\.cookie/);
  assert.match(adapter, /new Worker\(/);
  assert.match(adapter, /sqlite3-worker1-promiser\.mjs/);
  assert.match(adapter, /SELECT datalog_query\(\?\)/);
  assert.match(adapter, /OMIT_LOAD_EXTENSION/);
  assert.match(demo, /CREATE TABLE project_owner/);
  assert.match(demo, /needs_follow_up/);
  assert.match(demo, /reachable/);
  assert.match(demo, /support_ticket/);
  assert.match(demo, /release_candidate/);
  assert.match(demo, /team_member/);
  assert.match(page, /href=\{playground\}/);
  assert.match(playgroundPage, /<Playground \/>/);
  assert.doesNotMatch(`${page}\n${playgroundPage}\n${ide}`, /from "next\/link"/);
  assert.doesNotMatch(page, /_sites-preview|SkeletonPreview|codex-preview/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.match(packageJson, /"@mlc-ai\/web-llm"/);
  assert.equal(og.subarray(0, 8).toString("hex"), "89504e470d0a1a0a");
});

test("GitHub Pages export is self-contained when present", async () => {
  const exported = new URL("../dist/pages/index.html", import.meta.url);
  const exportedPlayground = new URL("../dist/pages/playground/index.html", import.meta.url);
  const exportedChatLab = new URL("../dist/pages/labs/chat-memory/index.html", import.meta.url);
  const exportedAgentLab = new URL("../dist/pages/labs/grounded-agent/index.html", import.meta.url);
  const exportedRecallLab = new URL("../dist/pages/labs/reading-recall/index.html", import.meta.url);
  const exportedWriterReaderLab = new URL("../dist/pages/labs/writer-reader/index.html", import.meta.url);
  const exportedResearch = new URL("../dist/pages/research/index.html", import.meta.url);
  const exportedAgentGuide = new URL("../dist/pages/guides/agent-harness/index.html", import.meta.url);
  await access(exported);
  await access(exportedPlayground);
  await access(exportedChatLab);
  await access(exportedAgentLab);
  await access(exportedRecallLab);
  await access(exportedWriterReaderLab);
  await access(exportedResearch);
  await access(exportedAgentGuide);
  const [html, playgroundHtml, chatLabHtml, agentLabHtml, recallLabHtml, writerReaderLabHtml, researchHtml, agentGuideHtml] = await Promise.all([
    readFile(exported, "utf8"),
    readFile(exportedPlayground, "utf8"),
    readFile(exportedChatLab, "utf8"),
    readFile(exportedAgentLab, "utf8"),
    readFile(exportedRecallLab, "utf8"),
    readFile(exportedWriterReaderLab, "utf8"),
    readFile(exportedResearch, "utf8"),
    readFile(exportedAgentGuide, "utf8"),
  ]);
  assert.match(html, /Memory you(?:<br\/>|\s)+can <em>reason<\/em> with\./);
  assert.match(html, /href="\/playground"/);
  assert.match(html, /href="\/research"/);
  assert.match(html, /href="\/labs\/writer-reader"/);
  assert.match(playgroundHtml, /SQLite \+ Datalog IDE/);
  assert.match(chatLabHtml, /Same database\. Same model\. Different powers\./);
  assert.match(agentLabHtml, /Let the model propose\. Let the gate show its work\./);
  assert.match(recallLabHtml, /no model executes on this page/);
  assert.match(writerReaderLabHtml, /no model executes on this page/);
  assert.match(researchHtml, /Every claim on this site/);
  assert.match(agentGuideHtml, /Wire one narrow tool loop\./);
  for (const labHtml of [chatLabHtml, agentLabHtml, recallLabHtml, writerReaderLabHtml]) {
    assert.doesNotMatch(
      labHtml,
      /lib-[A-Za-z0-9_-]+\.js/,
      "the multi-megabyte WebLLM chunk must stay behind the explicit load action",
    );
  }
  for (const rendered of [html, playgroundHtml, chatLabHtml, agentLabHtml, recallLabHtml, writerReaderLabHtml, researchHtml, agentGuideHtml]) {
    assert.match(rendered, /href="\/_next\/static\/css\//);
    assert.match(rendered, /src="\/_next\/static\/chunks\//);
    assert.match(rendered, /https:\/\/remembero\.rahultrikha\.com\/og\.png/);
  }
  await access(new URL("../dist/pages/.nojekyll", import.meta.url));
  await access(new URL("../dist/pages/og.png", import.meta.url));
  await access(new URL("../dist/pages/sqlite-wasm/sqlite3.wasm", import.meta.url));
});

test("SQLite Wasm contains the pinned Remembero extension build", async () => {
  const assetRoot = new URL("../public/sqlite-wasm/", import.meta.url);
  const manifest = JSON.parse(
    await readFile(new URL("manifest.json", assetRoot), "utf8"),
  );
  assert.equal(manifest.format, "rembero.sqlite-wasm.v1");
  assert.equal(manifest.sqlite.version, "3.53.4");
  assert.equal(manifest.extension.dynamicallyLoaded, false);
  assert.equal(
    manifest.extension.registration,
    "sqlite3_auto_extension(sqlite3_rembero_init)",
  );

  for (const [name, expected] of Object.entries(manifest.artifacts)) {
    const contents = await readFile(new URL(name, assetRoot));
    assert.equal(contents.byteLength, expected.bytes, `${name} byte length`);
    assert.equal(
      createHash("sha256").update(contents).digest("hex"),
      expected.sha256,
      `${name} digest`,
    );
  }

  const sourceRoot = new URL("../../native/", import.meta.url);
  for (const [name, expected] of Object.entries(manifest.extension.sources)) {
    const contents = await readFile(new URL(name, sourceRoot));
    assert.equal(
      createHash("sha256").update(contents).digest("hex"),
      expected.sha256,
      `${name} source digest`,
    );
  }

  const wasm = await readFile(new URL("sqlite3.wasm", assetRoot));
  assert.equal(wasm.subarray(0, 4).toString("hex"), "0061736d");
  for (const functionName of ["datalog_sql", "datalog_query", "datalog_explain"]) {
    assert.ok(wasm.includes(Buffer.from(functionName)), `${functionName} is linked`);
  }
});
