import Image from "next/image";

const github = "https://github.com/rahult/remembero";
const playground = "/playground";
const chatMemoryLab = "/labs/chat-memory";
const groundedAgentLab = "/labs/grounded-agent";
const agentHarnessGuide = "/guides/agent-harness";
const readerDoc = `${github}/blob/main/docs/research/READER-STRUCTURE.md`;
const modelComparison = `${github}/blob/main/docs/research/MODEL-COMPARISON.md`;
const decisionsReadme = `${github}/blob/main/remembro-eval/README.md`;
const exercisesDir = `${github}/tree/main/remembro-eval/exercises`;
const extractionBench = `${github}/blob/main/docs/research/EXTRACTION-BENCH.md`;

function HeroProof() {
  return (
    <div className="hero-proof" aria-label="Example proof-carrying answer">
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

export default function Home() {
  return (
    <main className="marketing-home">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Remembero home">remembero</a>
        <nav className="desktop-nav" aria-label="Main navigation">
          <a href="#product">Product</a><a href="#models">Models</a><a href="#labs">Labs</a><a href="#examples">Examples</a><a href={agentHarnessGuide}>Agent guide</a><a href={playground}>Playground</a><a href={github}>GitHub</a>
        </nav>
        <div className="header-actions">
          <a className="button primary header-try" href={playground}>Try the playground</a>
          <a className="button secondary desktop-source" href={github}>View on GitHub</a>
          <details className="mobile-menu">
            <summary aria-label="Open menu"><i /><i /><i /></summary>
            <nav aria-label="Mobile navigation"><a href="#product">Product</a><a href="#models">Models</a><a href="#labs">Labs</a><a href="#examples">Examples</a><a href={agentHarnessGuide}>Agent guide</a><a href={playground}>Playground</a><a href={github}>GitHub</a></nav>
          </details>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <h1>Memory you<br />can reason with.</h1>
          <p>Store facts and rules as readable knowledge. Ask useful questions. Get deterministic answers with the proof attached.</p>
          <div className="hero-actions"><a className="button primary" href={playground}>Try the playground</a><a className="button secondary" href={github}>View on GitHub</a></div>
          <span className="hero-boundary">Local-first by default. Runs on our own 2.3B model. Logic owns the answer.</span>
        </div>
        <HeroProof />
      </section>

      <section className="difference section-dark" id="product">
        <div className="section-shell">
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

      <section className="product-showcase section" aria-labelledby="showcase-title">
        <div className="section-shell">
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

      <section className="labs-showcase section-dark" id="labs" aria-labelledby="labs-title">
        <div className="section-shell">
          <div className="labs-heading">
            <h2 id="labs-title">See what better tools do for a small model.</h2>
            <p>Three browser workbenches expose the full chain: optional Hermes 7B WebLLM inference with native tool calls, deterministic memory and policy, then SQLite + Wasm execution with the call, result, proof, and timing evidence on screen.</p>
          </div>
          <div className="lab-links">
            <a href={chatMemoryLab}>
              <span>Chat recall lab</span>
              <strong>Same small model.<br /><em>Better tool.</em></strong>
              <p>Watch Hermes 7B issue native WebLLM tool calls against one shared SQLite database: raw SQL rows in one lane, Remembero bindings and proof in the other.</p>
              <b aria-hidden="true">Open lab →</b>
            </a>
            <a href={groundedAgentLab}>
              <span>Grounded agent lab</span>
              <strong>Let the model propose.<br /><em>Let rules decide.</em></strong>
              <p>Run the same Hermes 7B model with and without memory, then watch the request facts, packet swap, gate query, rule, and proof chain stay visible while the action resolves.</p>
              <b aria-hidden="true">Open lab →</b>
            </a>
            <a href={playground}>
              <span>SQLite + Datalog playground</span>
              <strong>Mutate SQLite.<br /><em>Measure the proof.</em></strong>
              <p>Insert real rows, execute the Remembero extension inside SQLite WebAssembly, and inspect the browser-local tables, compiled rule, proof graph, and current-browser timings.</p>
              <b aria-hidden="true">Open playground →</b>
            </a>
          </div>
        </div>
      </section>

      <section className="how section" id="how-it-works">
        <div className="section-shell">
          <h2>An answer is only useful if you can inspect <em>why.</em></h2>
          <div className="steps">
            <article><div className="step-title"><b>1</b><h3>Store evidence</h3></div><p>Capture a fact with the statement it came from.</p><code>project_owner(atlas, rahul).</code></article>
            <article><div className="step-title"><b>2</b><h3>Apply reviewed rules</h3></div><p>Derive useful knowledge without storing invented conclusions.</p><code>collaborator(Person, Project) :- …</code></article>
            <article><div className="step-title"><b>3</b><h3>Return the support chain</h3></div><p>Inspect the exact claims and rules behind every answer.</p><code>answer → rule → sourced facts</code></article>
          </div>
        </div>
      </section>

      <section className="models section-dark" id="models" aria-labelledby="models-title">
        <div className="section-shell">
          <div className="models-heading">
            <div><h2 id="models-title">Runs on our own <em>small models.</em></h2><p>The model that translates is a 2.3B fine-tune that runs on a laptop. The frontier model is the teacher and the yardstick, not a dependency at runtime.</p></div>
            <a className="button secondary" href={modelComparison}>Model comparison</a>
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
              <p>Answers from retrieved history. Distilled from GLM 5.3 Flash over real sessions, handed computed notes before it reads (dates resolved, distances stated, totals summed, deterministically), and trained to write its own working: the dated items and the arithmetic first, then one answer line.</p>
              <dl>
                <div><dt>LongMemEval, 500 questions</dt><dd>414<i>/500</i></dd></div>
                <div><dt>Teacher, same judge</dt><dd>440<i>/500</i></dd></div>
                <div><dt>Computed notes, no training</dt><dd>+29</dd></div>
                <div><dt>Writing its working, trained</dt><dd>+10</dd></div>
                <div><dt>Turn-level retrieval and dated notes</dt><dd>+17</dd></div>
              </dl>
            </article>
          </div>
          <div className="model-ledger">
            <span><strong>Computed notes.</strong> Every temporal expression resolved against when it was said, every line quoting its sentence.</span>
            <span><strong>Structured evidence.</strong> The memory&apos;s own facts, dated, grounded and deduplicated, placed before the chats.</span>
            <span><strong>Evidence mode by default.</strong> One model call to translate, none to phrase; recalled facts never leave the process.</span>
            <span><strong>Empty results explain themselves.</strong> The engine reports which goal matched nothing and which swap would return rows.</span>
          </div>
          <p className="models-note">Numbers under one judge (DeepSeek) and one protocol; the reader is 57 behind its teacher on the 500 and the gap is the work. Local embeddings (nomic-embed-text) tie the hosted model on the semantic route. <a href={readerDoc}>Method and every run</a>.</p>
        </div>
      </section>

      <section className="examples section" id="examples" aria-labelledby="examples-title">
        <div className="section-shell">
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
              <code>GLM 5.3 Flash 440 · our reader 414 · noise band ±7</code>
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
            <h2>Models translate.<br />Rules <em>decide.</em></h2>
            <p>Our own writer translates a question into a query. Remembero evaluates the accepted query against explicit knowledge, adds the computed notes, and returns the evidence locally. No model phrases the answer unless you ask for one.</p>
            <ol className="boundary-flow"><li>Question <span>natural language</span></li><li>Translate <span>our 2.3B writer</span></li><li>Query <span>accepted</span></li><li>Evaluate <span>rules + facts</span></li><li>Notes <span>dates, distances, totals</span></li><li>Answer + evidence</li></ol>
          </article>
          <article className="integrations">
            <h2>One memory layer.<br />Three ways <em>in.</em></h2>
            <div className="integration-list"><div><strong>MCP</strong><span>An eight-tool core profile for agents; <code>remembero init</code> installs the Claude Code hooks and a session brief.</span></div><div><strong>TypeScript</strong><span>Use the typed library API inside your applications.</span></div><div><strong>CLI</strong><code>npx -y remembero</code></div></div>
          </article>
        </div>
      </section>

      <section className="final-cta section">
        <div className="section-shell final-cta-grid">
          <div><h2>Build agents that can <em>show their work.</em></h2><p>Try a real-life lab first, then open the IDE when you want to inspect the machinery.</p></div>
          <div className="final-actions"><a className="button primary" href={chatMemoryLab}>Open a lab</a><a className="button link-button" href={playground}>Open the playground <span aria-hidden="true">→</span></a></div>
        </div>
      </section>

      <footer className="site-footer">
        <strong>remembero</strong>
        <nav aria-label="Footer navigation"><a href="#models">Models</a><a href="#examples">Examples</a><a href={chatMemoryLab}>Chat lab</a><a href={groundedAgentLab}>Agent lab</a><a href={playground}>Playground</a><a href={github}>GitHub</a><a href={`${github}#readme`}>Docs</a><a href="https://www.npmjs.com/package/remembero">npm</a><span>MIT licensed</span></nav>
      </footer>
    </main>
  );
}
