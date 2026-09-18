import Image from "next/image";
import { FirstProofDemo } from "./first-proof-demo";

const github = "https://github.com/rahult/remembero";
const playground = "/playground";
const chatMemoryLab = "/labs/chat-memory";
const groundedAgentLab = "/labs/grounded-agent";
const readingRecallLab = "/labs/reading-recall";
const writerReaderLab = "/labs/writer-reader";
const researchPage = "/research";
const agentHarnessGuide = "/guides/agent-harness";

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

const problems = [
  {
    title: "Chats forget.",
    body: "A model's memory is a whiteboard wiped at the end of the session. Decisions, preferences, facts — gone by tomorrow, unless someone rebuilds them by hand.",
  },
  {
    title: "Retrieval guesses.",
    body: "The usual fix searches old text for passages similar to the question. Similar is not relevant: it grabs what is nearby, cannot combine facts across sessions, and cannot say what it missed.",
  },
  {
    title: "Confidently wrong.",
    body: "When memory half-remembers, agents answer anyway. The failure is silent — no missing-piece alarm, no audit trail, no way to check the working after the fact.",
  },
];

const ideaSteps = [
  {
    title: "Store what was said",
    body: "Facts are short, plain sentences with their source attached — readable by you, and exact enough for a machine.",
    hint: "project_contributor(atlas, maya)",
  },
  {
    title: "Add rules once",
    body: "Rules are plain if-then knowledge: anyone contributing to a project is a collaborator on it. Written once, applied forever, same result every time.",
    hint: "collaborator(P, Proj) :- project_contributor(Proj, P)",
  },
  {
    title: "Ask, then check the working",
    body: "Questions become queries over facts and rules. Every answer carries the chain of evidence behind it — or an honest “not in memory.”",
    hint: "answer → rule → sourced facts",
  },
];

export default function Home() {
  return (
    <main className="marketing-home">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Remembero home">remembero</a>
        <nav className="desktop-nav" aria-label="Main navigation">
          <a href="#problem">The problem</a><a href="#idea">The idea</a><a href="#try">Try it</a><a href="#evidence">Evidence</a><a href={researchPage}>Research</a><a href={playground}>Playground</a><a href={github}>GitHub</a>
        </nav>
        <div className="header-actions">
          <a className="button primary header-try" href="#demo">See it work</a>
          <a className="button secondary desktop-source" href={github}>View on GitHub</a>
          <details className="mobile-menu">
            <summary aria-label="Open menu"><i /><i /><i /></summary>
            <nav aria-label="Mobile navigation"><a href="#problem">The problem</a><a href="#idea">The idea</a><a href="#try">Try it</a><a href="#evidence">Evidence</a><a href={researchPage}>Research</a><a href={agentHarnessGuide}>Agent guide</a><a href={playground}>Playground</a><a href={github}>GitHub</a></nav>
          </details>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="hero-eyebrow">Durable memory for AI agents</p>
          <h1>Memory you<br />can <em>reason</em> with.</h1>
          <p>Remembero gives AI agents memory as plain, readable facts and rules — and proves every answer it gives. Nothing fuzzy, nothing hidden: you can check the working yourself, starting sixty seconds from now.</p>
          <div className="hero-actions"><a className="button primary" href="#demo">See it work — 60 seconds</a><a className="button secondary" href={github}>View on GitHub</a></div>
          <span className="hero-boundary">This whole site runs in your browser. No accounts, no installs — and none of our model weights anywhere.</span>
        </div>
        <HeroProof />
      </section>

      <section className="difference section-dark" id="problem">
        <div className="section-shell">
          <p className="section-tag">01 · The problem</p>
          <h2>AI that forgets —<br />or worse, <em>misremembers.</em></h2>
          <p className="section-lede">If you have ever asked an assistant about something from last month and gotten a confident invention, you already know all three of these.</p>
          <div className="problem-grid">
            {problems.map((problem) => (
              <article key={problem.title}>
                <h3>{problem.title}</h3>
                <p>{problem.body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="research section" id="idea" aria-labelledby="idea-title">
        <div className="section-shell">
          <p className="section-tag">02 · The idea</p>
          <h2 id="idea-title">Write it down.<br />Prove what <em>follows.</em></h2>
          <div className="idea-grid">
            <div>
              <ol className="idea-steps">
                {ideaSteps.map((step, index) => (
                  <li key={step.title}>
                    <div className="idea-step-head"><b>{index + 1}</b><h3>{step.title}</h3></div>
                    <p>{step.body}</p>
                    <code>{step.hint}</code>
                  </li>
                ))}
              </ol>
              <p className="idea-kicker">No model did any of this. That is the point — and the demo on the right is doing it for real, in your browser, as you click.</p>
            </div>
            <FirstProofDemo />
          </div>
        </div>
      </section>

      <section className="labs-showcase section-dark" id="try" aria-labelledby="try-title">
        <div className="section-shell">
          <p className="section-tag">03 · Try it</p>
          <div className="try-head">
            <h2 id="try-title">Four workbenches.<br />Zero <em>downloads.</em></h2>
            <p>Work deeper at each step. Everything runs in this tab over fictional data — and wherever a model appears, it is either an open model your own browser loads on demand, or a clearly labeled replay of a recorded run.</p>
          </div>
          <div className="lab-grid">
            <a className="lab-card" href={writerReaderLab}>
              <span className="lab-kind">Writer + reader · replays</span>
              <h3>Raw text in.<br /><em>Proven answers out.</em></h3>
              <p>Six months of messy chat — corrections, a switch-back at a new price, a handover, a contract that ends, and one question never answered anywhere. Watch the writer turn it into claims, watch code build validity timelines, then ask what breaks naive memory.</p>
              <div className="lab-foot"><span>live structure · replayed writer & reader</span><b>Open lab →</b></div>
            </a>
            <a className="lab-card" data-kind="model" href={chatMemoryLab}>
              <span className="lab-kind">Optional model</span>
              <h3>Watch a model<br /><em>use it as a tool.</em></h3>
              <p>A language model answers questions by calling Remembero as a tool over one shared database — against raw SQL in the other lane, so you can see exactly what the structure buys.</p>
              <div className="lab-foot"><span>SQLite + Wasm · bring an open model</span><b>Open lab →</b></div>
            </a>
            <a className="lab-card" data-kind="model" href={groundedAgentLab}>
              <span className="lab-kind">Optional model</span>
              <h3>Watch a bad write<br /><em>get refused.</em></h3>
              <p>A model proposes an action it must never own — and a deterministic gate approves or blocks it, with the complete decision proof on screen.</p>
              <div className="lab-foot"><span>gate rule · decision proof</span><b>Open lab →</b></div>
            </a>
            <a className="lab-card" href={readingRecallLab}>
              <span className="lab-kind">Deterministic</span>
              <h3>Watch long chats become<br /><em>reliable answers.</em></h3>
              <p>The reading pipeline, assembled step by step: which chats get picked, which get condensed, and the date-and-arithmetic notes code writes before any model reads — then both sides of a recorded before/after run.</p>
              <div className="lab-foot"><span>live code · replayed answers</span><b>Open lab →</b></div>
            </a>
          </div>
          <div className="try-showcase">
            <a className="showcase-frame" href={playground} aria-label="Open the Remembero SQLite and Datalog playground">
              <Image src="/og.png" alt="Remembero SQLite and Datalog IDE showing tables, a query, proof, and graph" width={1731} height={909} unoptimized priority />
            </a>
            <div className="showcase-ledger">
              <span><strong>SQLite owns the rows.</strong> Ordinary tables remain the storage authority.</span>
              <span><strong>Rules own the query.</strong> The C extension executes inside SQLite WebAssembly.</span>
              <span><strong>Proof owns the answer.</strong> Every result can show its complete support chain.</span>
            </div>
            <div className="try-cta"><a className="button primary" href={playground}>Open the playground — the real database, in your tab</a></div>
          </div>
        </div>
      </section>

      <section className="evidence section" id="evidence" aria-labelledby="evidence-title">
        <div className="section-shell">
          <p className="section-tag">04 · The evidence</p>
          <h2 id="evidence-title">We measured <em>everything.</em></h2>
          <p className="section-lede">Remembero's claim — that structure beats scale for agent memory — is tested, not asserted. On a public benchmark of 500 questions about long chat histories, a small open model kept getting more right as deterministic code took over the parts models are bad at. It now sits fifteen questions behind the frontier model that trained it.</p>
          <div className="metric-row">
            <div className="metric"><strong>500</strong><span>questions on the public long-memory benchmark (LongMemEval-S) — multi-session, temporal, updates, abstention</span></div>
            <div className="metric"><strong>318 → 425</strong><span>correct answers as code-written structure was added to the same small model — dates resolved, arithmetic done, chats re-ranked</span></div>
            <div className="metric"><strong>15 short</strong><span>of the frontier model that trained it, under the same automated grader — with run-to-run noise of about ±7</span></div>
          </div>
          <div className="evidence-actions"><a className="button primary" href={researchPage}>Read the evidence</a><a className="button link-button" href={github}>Browse the raw logs <span aria-hidden="true">→</span></a></div>
        </div>
      </section>

      <section className="boundary section-dark">
        <div className="section-shell boundary-grid">
          <article className="model-boundary">
            <p className="section-tag">05 · Where models fit</p>
            <h2>Models translate.<br />Rules <em>decide.</em></h2>
            <p>Everything you tried above ran with zero models. When you want to speak plain English to your memory, a model does the translating — an open model loaded by your own browser in the labs, or any API model in the real product. The model never decides what is true. The rules do, and they show their work.</p>
            <ol className="boundary-flow"><li>Question <span>plain English</span></li><li>Translate <span>optional model</span></li><li>Query <span>checked, accepted</span></li><li>Evaluate <span>rules + facts</span></li><li>Notes <span>dates, sums — by code</span></li><li>Answer + proof</li></ol>
          </article>
          <article className="integrations">
            <p className="section-tag">06 · Start building</p>
            <h2>One memory layer.<br />Three ways <em>in.</em></h2>
            <div className="integration-list"><div><strong>MCP</strong><span>An eight-tool core profile for agents; <code>remembero init</code> installs the Claude Code hooks and a session brief.</span></div><div><strong>TypeScript</strong><span>Use the typed library API inside your applications.</span></div><div><strong>CLI</strong><code>npx -y remembero</code></div></div>
          </article>
        </div>
      </section>

      <section className="final-cta section">
        <div className="section-shell final-cta-grid">
          <div><h2>Build agents that can <em>show their work.</em></h2><p>Start with the sixty-second demo, work a lab, then open the database itself — the whole ladder is one click away.</p></div>
          <div className="final-actions"><a className="button primary" href="#demo">Back to the demo</a><a className="button link-button" href={playground}>Open the playground <span aria-hidden="true">→</span></a></div>
        </div>
      </section>

      <footer className="site-footer">
        <strong>remembero</strong>
        <nav aria-label="Footer navigation"><a href="#problem">Problem</a><a href="#idea">Idea</a><a href="#try">Try it</a><a href="#evidence">Evidence</a><a href={researchPage}>Research</a><a href={writerReaderLab}>Writer–reader lab</a><a href={readingRecallLab}>Reading lab</a><a href={chatMemoryLab}>Chat lab</a><a href={groundedAgentLab}>Agent lab</a><a href={playground}>Playground</a><a href={github}>GitHub</a><a href={`${github}#readme`}>Docs</a><a href="https://www.npmjs.com/package/remembero">npm</a><span>MIT licensed</span></nav>
      </footer>
    </main>
  );
}
