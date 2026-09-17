import Image from "next/image";

const github = "https://github.com/rahult/remembero";
const playground = "/playground";
const chatMemoryLab = "/labs/chat-memory";
const groundedAgentLab = "/labs/grounded-agent";
const readingRecallLab = "/labs/reading-recall";
const agentHarnessGuide = "/guides/agent-harness";
const readerDoc = `${github}/blob/main/docs/research/READER-STRUCTURE.md`;
const modelComparison = `${github}/blob/main/docs/research/MODEL-COMPARISON.md`;
const decisionsReadme = `${github}/blob/main/remembro-eval/README.md`;
const exercisesDir = `${github}/tree/main/remembro-eval/exercises`;
const extractionBench = `${github}/blob/main/docs/research/EXTRACTION-BENCH.md`;

function HeroProof() {
  return (
    <div className="hero-proof" aria-label="Example proof-carrying answer">
      <span className="stamp stamp-proven hero-proof-stamp" aria-hidden="true">Proof-carrying</span>
      <div className="hero-proof-row"><span>Question</span><p>Who is collaborating on Atlas?</p></div>
      <div className="hero-proof-row"><span>Query</span><code>collaborator(Person, atlas)</code></div>
      <div className="hero-proof-row hero-answer"><span>Answer</span><p>Maya is collaborating on Atlas.</p></div>
      <div className="hero-proof-row hero-because">
        <span>Because</span>
        <ol>
          <li><b>1</b><code>project_owner(atlas, rahul)</code></li>
          <li><b>2</b><code>project_contributor(atlas, maya)</code></li>
        </ol>
      </div>
      <div className="hero-proof-source"><span>Atlas planning session · 17 Aug</span><a href={playground}>Open in playground</a></div>
    </div>
  );
}

const computedNotesSample = `COMPUTED NOTES — written by code, not a model

Dated events (the user's own words, resolved)
  "two weeks ago I moved to the Marina"
      said 2026-03-03 → 2026-02-17 · 153 days before the question
  "ran my first 10K yesterday"
      said 2026-04-02 → 2026-04-01 · 110 days before the question

Gaps
  Marina move → first 10K: 43 days (~6 weeks)

Quantities (each line quotes its sentence)
  $1,100  "bought the road bike in March"
  $700    "selling it for $700"
  exactly two figures → difference: $400`;

const deltas = [
  {
    label: "Computed notes",
    detail: "Dates resolved against the day they were said, distances and gaps stated, totals summed — a deterministic block placed in the prompt before the reader reads. No training, no added model call.",
    value: "+29",
  },
  {
    label: "The thinking step, trained",
    detail: "Reader v7, distilled to write its own working: the dated items it relies on and the arithmetic first, then one answer line. Only the final line is judged.",
    value: "+10",
  },
  {
    label: "Turn-level retrieval, routed by question text",
    detail: "The retrieval unit is chosen from the question itself — whole sessions for some types, single turns for others — with dated-notes rules rebuilt to match.",
    value: "+17",
  },
  {
    label: "TypeSafe re-rank of the shortlist",
    detail: "A typed re-ranker reorders the lexical shortlist so the sessions that carry the answer land in context. Answer turns in context rose from 84% to 95% for $0.28 of calls.",
    value: "+13",
  },
];

export default function Home() {
  return (
    <main className="marketing-home">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Remembero home">remembero</a>
        <nav className="desktop-nav" aria-label="Main navigation">
          <a href="#product">Product</a><a href="#research">Research</a><a href="#labs">Labs</a><a href="#models">Models</a><a href="#examples">Examples</a><a href={playground}>Playground</a><a href={github}>GitHub</a>
        </nav>
        <div className="header-actions">
          <a className="button primary header-try" href={playground}>Try the playground</a>
          <a className="button secondary desktop-source" href={github}>View on GitHub</a>
          <details className="mobile-menu">
            <summary aria-label="Open menu"><i /><i /><i /></summary>
            <nav aria-label="Mobile navigation"><a href="#product">Product</a><a href="#research">Research</a><a href="#labs">Labs</a><a href="#models">Models</a><a href="#examples">Examples</a><a href={agentHarnessGuide}>Agent guide</a><a href={playground}>Playground</a><a href={github}>GitHub</a></nav>
          </details>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="hero-eyebrow">Proof-carrying memory · deterministic core</p>
          <h1>Memory you<br />can <em>reason</em> with.</h1>
          <p>Store facts and rules as readable knowledge. Ask useful questions. Get deterministic answers with the proof attached — and a research trail behind every number we claim.</p>
          <div className="hero-actions"><a className="button primary" href={playground}>Try the playground</a><a className="button secondary" href={github}>View on GitHub</a></div>
          <span className="hero-boundary">Every demo on this site runs in your browser. No weights served, nothing stored, nothing leaves the tab.</span>
        </div>
        <HeroProof />
      </section>

      <section className="difference section-dark" id="product">
        <div className="section-shell">
          <p className="section-tag">01 · Product</p>
          <h2>Not another vector store.</h2>
          <p className="section-lede">Similarity finds nearby text. Remembero proves what follows.</p>
          <div className="difference-grid">
            <article><h3>Readable memory</h3><p>Plain-text facts, rules, and constraints.</p><pre><code>{`project_owner(atlas, rahul).
status(atlas, blocked).`}</code></pre></article>
            <article><h3>Deterministic rules</h3><p>Same knowledge. Same query. Same answer.</p><pre><code>{`needs_follow_up(Person, Project) :-
  promised_update(rahul, Person, Project),
  status(Project, blocked).`}</code></pre></article>
            <article><h3>Proof, not vibes</h3><p>Every derived answer carries its supporting claims.</p><pre><code>{`collaborator(maya, atlas)
├─ project_owner(atlas, rahul)
└─ project_contributor(atlas, maya)`}</code></pre></article>
          </div>
        </div>
      </section>

      <section className="research section" id="research" aria-labelledby="research-title">
        <div className="section-shell">
          <p className="section-tag">02 · Research</p>
          <h2 id="research-title">Structure first.<br />Then <em>reading.</em></h2>
          <div className="research-grid">
            <div>
              <p className="section-lede" style={{ marginTop: 0 }}>We kept asking why a small reader loses on long chat histories when the evidence is right there. The answer shaped everything we have built since.</p>
              <p className="section-lede">It reads the right sentences and computes wrong:</p>
              <ul className="miss-list">
                <li><span><b>Counts drift across sessions</b> — three festivals counted for four.</span></li>
                <li><span><b>Relative dates stay unanchored</b> — “just got back,” said July 15, asked Aug 5, answered without the three weeks.</span></li>
                <li><span><b>Gaps get mis-subtracted</b> — “4:22 minus 4:10” answered as 17 minutes.</span></li>
                <li><span><b>Sums drop an item</b> — two of three road-trip legs; 50 lb of feed for 70.</span></li>
              </ul>
              <p className="section-lede">A small model can read a sentence and cannot be trusted with the arithmetic. So the structure moved into code, and the model was trained on what the structure hands it.</p>
            </div>
            <div>
              <div className="note-block">
                <div className="note-block-header"><span>Inserted before the reader reads</span><span>0 model calls</span></div>
                <pre>{computedNotesSample}</pre>
              </div>
              <div className="pair-callout">
                <strong>318 <i>→ 359</i></strong>
                <span>LongMemEval, 500 questions, reader v4 — the same reader, the same retrieval, before and after the computed-notes block. One paired run, one judge (gpt-4o), and the block is identical on every run.</span>
              </div>
            </div>
          </div>
          <div className="delta-ledger" aria-labelledby="delta-title">
            <div className="delta-heading">
              <h3 id="delta-title">The delta ledger: how the reader closed on its teacher</h3>
              <span>paired runs · one judge (DeepSeek) · noise ±7</span>
            </div>
            {deltas.map((delta) => (
              <div className="delta-item" key={delta.label}>
                <div><strong>{delta.label}</strong><small>{delta.detail}</small></div>
                <code>{delta.value}</code>
                <span className="delta-chip" aria-label="questions gained">▲</span>
              </div>
            ))}
            <div className="delta-item">
              <div><strong>Reader v7, full stack — against its teacher</strong><small>Gemma 4 E4B distilled from GLM 5.3 Flash, the same model that taught it. Fifteen questions short of the teacher on the 500.</small></div>
              <code>425<i>/500</i></code>
              <span className="delta-chip teacher">teacher 440</span>
            </div>
            <p className="delta-footnote">Each row is a measured paired run — two arms differing in exactly one setting with identical retrieval — from <a href={readerDoc}>the method and every run</a>. Nothing on this page required serving a model: the deltas were measured in the repository, and the lab below shows you the machinery instead.</p>
          </div>
        </div>
      </section>

      <section className="product-showcase section-dark" aria-labelledby="showcase-title">
        <div className="section-shell">
          <p className="section-tag">03 · The database</p>
          <div className="showcase-heading">
            <div><h2 id="showcase-title">The database is the demo.</h2><p>Insert a SQLite row, run Datalog, then inspect the exact facts and rule behind the answer—all inside your browser.</p></div>
            <a className="button primary" href={playground}>Open the full playground</a>
          </div>
          <a className="showcase-frame" href={playground} aria-label="Open the Remembero SQLite and Datalog playground">
            <Image src="/og.png" alt="Remembero SQLite and Datalog IDE showing tables, a query, proof, and graph" width={1731} height={909} unoptimized priority />
          </a>
          <div className="showcase-ledger">
            <span><strong>SQLite owns the rows.</strong> Ordinary tables remain the storage authority.</span>
            <span><strong>Rules own the query.</strong> The C extension executes inside SQLite WebAssembly.</span>
            <span><strong>Proof owns the answer.</strong> Every result can show its complete support chain.</span>
          </div>
        </div>
      </section>

      <section className="labs-showcase section" id="labs" aria-labelledby="labs-title">
        <div className="section-shell">
          <div className="labs-heading">
            <p className="section-tag">04 · Labs</p>
            <h2 id="labs-title">Four workbenches.<br />Zero served <em>weights.</em></h2>
            <p>Everything below runs browser-local: deterministic engines, SQLite compiled to WebAssembly with our C extension linked in, and model output only where it is clearly labeled — as an optional third-party WebLLM load or a replay of a recorded run.</p>
          </div>
          <div className="lab-grid">
            <a className="lab-card" href={readingRecallLab}>
              <span className="lab-kind">Deterministic · New</span>
              <h3>The reader contract,<br /><em>model taken out.</em></h3>
              <p>Pick a question and watch the reading pipeline assemble: the shortlist retrieved and re-ranked, sessions tiered into abstracts and full text, computed notes written by code in front of you — then both arms of a recorded paired run, the same reader with and without the notes.</p>
              <div className="lab-foot"><span>live deterministic code · replayed answers</span><b>Open lab →</b></div>
            </a>
            <a className="lab-card" data-kind="model" href={chatMemoryLab}>
              <span className="lab-kind">Optional model</span>
              <h3>Same small model.<br /><em>Better tool.</em></h3>
              <p>A model issues tool calls against one shared SQLite database: raw SQL rows in one lane, Remembero bindings and proof in the other, across four questions where the lanes structurally diverge.</p>
              <div className="lab-foot"><span>SQLite + Wasm · optional Hermes 7B</span><b>Open lab →</b></div>
            </a>
            <a className="lab-card" data-kind="model" href={groundedAgentLab}>
              <span className="lab-kind">Optional model</span>
              <h3>Let the model propose.<br /><em>Let rules decide.</em></h3>
              <p>The same model with and without memory, proposing an action it must never own. Request facts, packet swap, gate query, rule, and decision proof stay on screen while the action resolves.</p>
              <div className="lab-foot"><span>gate rule · decision proof</span><b>Open lab →</b></div>
            </a>
            <a className="lab-card" href={playground}>
              <span className="lab-kind">Deterministic</span>
              <h3>Mutate SQLite.<br /><em>Measure the proof.</em></h3>
              <p>Insert real rows, execute the Remembero extension inside SQLite WebAssembly, and inspect the browser-local tables, compiled rule, proof graph, and current-browser timings.</p>
              <div className="lab-foot"><span>SQLite + Wasm · C extension</span><b>Open playground →</b></div>
            </a>
          </div>
        </div>
      </section>

      <section className="models section-dark" id="models" aria-labelledby="models-title">
        <div className="section-shell">
          <div className="models-heading">
            <div>
              <p className="section-tag">05 · Models</p>
              <h2 id="models-title">Our own small models. <em>Measured, never served.</em></h2>
              <p>The writer translates and the reader answers; both are fine-tunes we trained, measured against their teacher, and published as paired runs. The frontier model is the teacher and the yardstick — not a runtime dependency, and not a download on this site.</p>
            </div>
            <a className="button secondary" href={modelComparison}>Model comparison</a>
          </div>
          <div className="not-served">
            <span className="stamp stamp-ink">Not served here</span>
            <p><b>This site ships no model weights and calls no model API.</b> The playground and labs run deterministic code in your browser. Where model output appears, it is either a clearly labeled replay of a recorded run or the optional third-party Hermes 7B your own browser loads via WebLLM. Our fine-tuned readers and writers stay in the repository as research artifacts — <a href={readerDoc}>with every run that measured them</a>.</p>
          </div>
          <div className="model-grid">
            <article>
              <span>Writer</span>
              <h3>Gemma 4 E2B, fine-tuned</h3>
              <p>Turns chat and documents into facts, authors the Datalog query, extracts policy claims. Served as a 4.6 GiB Q8_0 GGUF under llama.cpp; a laptop scores what the bf16 copy scores on a cloud GPU.</p>
              <dl>
                <div><dt>Extraction benchmark</dt><dd>87<i>/103</i></dd></div>
                <div><dt>Query authoring</dt><dd>27<i>/31</i></dd></div>
                <div><dt>Policy decisions, unseen fixture</dt><dd>20<i>/20</i></dd></div>
                <div><dt>Unjustified ALLOW</dt><dd>0</dd></div>
              </dl>
            </article>
            <article>
              <span>Reader</span>
              <h3>Gemma 4 E4B, distilled</h3>
              <p>Answers from retrieved history. Distilled from GLM 5.3 Flash over real sessions, handed computed notes before it reads, and trained to write its own working: the dated items and the arithmetic first, then one answer line.</p>
              <dl>
                <div><dt>LongMemEval, 500 questions</dt><dd>425<i>/500</i></dd></div>
                <div><dt>Teacher, same judge</dt><dd>440<i>/500</i></dd></div>
                <div><dt>Computed notes, no training</dt><dd>+29</dd></div>
                <div><dt>Writing its working, trained</dt><dd>+10</dd></div>
                <div><dt>Turn-level retrieval and dated notes</dt><dd>+17</dd></div>
                <div><dt>Relevance re-ranking (TypeSafe)</dt><dd>+13</dd></div>
              </dl>
            </article>
          </div>
          <div className="model-ledger">
            <span><strong>Computed notes.</strong> Every temporal expression resolved against when it was said, every line quoting its sentence.</span>
            <span><strong>Structured evidence.</strong> The memory&apos;s own facts, dated, grounded and deduplicated, placed before the chats.</span>
            <span><strong>Evidence mode by default.</strong> One model call to translate, none to phrase; recalled facts never leave the process.</span>
            <span><strong>Empty results explain themselves.</strong> The engine reports which goal matched nothing and which swap would return rows.</span>
          </div>
          <p className="models-note">Numbers under one judge (DeepSeek) and one protocol; the reader is 15 behind its teacher on the 500 and the gap is the work. Local embeddings (nomic-embed-text) tie the hosted model on the semantic route. <a href={readerDoc}>Method and every run</a>.</p>
        </div>
      </section>

      <section className="examples section" id="examples" aria-labelledby="examples-title">
        <div className="section-shell">
          <p className="section-tag">06 · Examples</p>
          <h2 id="examples-title">Worked examples, with the <em>numbers attached.</em></h2>
          <p className="examples-lede">Each one is executable from the repository and reports what it refused as carefully as what it answered.</p>
          <div className="examples-grid">
            <a href={decisionsReadme}>
              <span>Decisions</span>
              <strong>Evidence → claims → beliefs → decisions</strong>
              <p>A five-page delegation policy, its amendment, a Slack message that revokes authority. &ldquo;May X approve Y on this date&rdquo; returns ALLOW, DENY or UNKNOWN with the quotes it rests on. Uncertainty never silently becomes permission.</p>
              <code>17/17 and 20/20 decisions · 0 unjustified ALLOW · eight MCP tools</code>
            </a>
            <a href={exercisesDir}>
              <span>Exercises</span>
              <strong>Scripts of what happens over time</strong>
              <p>A YAML script ingests, registers an alias, then decides at each date, and states what the answer must be and why. Runs against the real local writer in a fresh workspace.</p>
              <code>leave-week 10/10 · amendment 10/10</code>
            </a>
            <a href={readerDoc}>
              <span>LongMemEval</span>
              <strong>500 questions over long chat histories</strong>
              <p>Multi-session, temporal, knowledge-update, abstention. Every stored run re-judged under one judge; paired runs, identical retrieval, so a change is measured against the reader&apos;s own noise.</p>
              <code>GLM 5.3 Flash 440 · our reader 425 · noise band ±7</code>
            </a>
            <a href={extractionBench}>
              <span>Benchmarks</span>
              <strong>Extraction and the agent boundary</strong>
              <p>103 schema-conditioned extraction cases across twelve phenomena, and a 31-case boundary benchmark where the model proposes and a frozen rule set decides. Both publish their negative results.</p>
              <code>writer 87/103 · query 27/31 · frontier 96 and 30</code>
            </a>
          </div>
        </div>
      </section>

      <section className="boundary section-dark">
        <div className="section-shell boundary-grid">
          <article className="model-boundary">
            <p className="section-tag">07 · The boundary</p>
            <h2>Models translate.<br />Rules <em>decide.</em></h2>
            <p>A writer translates a question into a query. Remembero evaluates the accepted query against explicit knowledge, adds the computed notes, and returns the evidence locally. No model phrases the answer unless you ask for one.</p>
            <ol className="boundary-flow"><li>Question <span>natural language</span></li><li>Translate <span>small fine-tuned writer</span></li><li>Query <span>accepted</span></li><li>Evaluate <span>rules + facts</span></li><li>Notes <span>dates, distances, totals</span></li><li>Answer + evidence</li></ol>
          </article>
          <article className="integrations">
            <p className="section-tag">08 · Integration</p>
            <h2>One memory layer.<br />Three ways <em>in.</em></h2>
            <div className="integration-list"><div><strong>MCP</strong><span>An eight-tool core profile for agents; <code>remembero init</code> installs the Claude Code hooks and a session brief.</span></div><div><strong>TypeScript</strong><span>Use the typed library API inside your applications.</span></div><div><strong>CLI</strong><code>npx -y remembero</code></div></div>
          </article>
        </div>
      </section>

      <section className="final-cta section">
        <div className="section-shell final-cta-grid">
          <div><h2>Build agents that can <em>show their work.</em></h2><p>Work a lab first — the reading pipeline without a model, or a gated agent — then open the IDE when you want to inspect the machinery.</p></div>
          <div className="final-actions"><a className="button primary" href={readingRecallLab}>Open the reading lab</a><a className="button link-button" href={playground}>Open the playground <span aria-hidden="true">→</span></a></div>
        </div>
      </section>

      <footer className="site-footer">
        <strong>remembero</strong>
        <nav aria-label="Footer navigation"><a href="#research">Research</a><a href="#models">Models</a><a href="#examples">Examples</a><a href={readingRecallLab}>Reading lab</a><a href={chatMemoryLab}>Chat lab</a><a href={groundedAgentLab}>Agent lab</a><a href={playground}>Playground</a><a href={github}>GitHub</a><a href={`${github}#readme`}>Docs</a><a href="https://www.npmjs.com/package/remembero">npm</a><span>MIT licensed</span></nav>
      </footer>
    </main>
  );
}
