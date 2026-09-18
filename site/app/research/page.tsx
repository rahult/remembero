import type { Metadata } from "next";

const github = "https://github.com/rahult/remembero";
const readerDoc = `${github}/blob/main/docs/research/READER-STRUCTURE.md`;
const modelComparison = `${github}/blob/main/docs/research/MODEL-COMPARISON.md`;
const decisionsReadme = `${github}/blob/main/remembro-eval/README.md`;
const exercisesDir = `${github}/tree/main/remembro-eval/exercises`;
const extractionBench = `${github}/blob/main/docs/research/EXTRACTION-BENCH.md`;

export const metadata: Metadata = {
  title: "Remembero Research — The evidence, measured",
  description:
    "Every claim on the Remembero site, with its measurement: a public 500-question long-memory benchmark, controlled before/after runs, automated graders named, and small open models that close most of the gap to their frontier teacher — measured, never served.",
};

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

const failures = [
  "Counts drift across sessions — three festivals counted for four.",
  "Relative dates stay unanchored — “just got back,” said July 15, asked Aug 5, answered without the three weeks.",
  "Gaps get mis-subtracted — “4:22 minus 4:10” answered as 17 minutes.",
  "Sums drop an item — two of three road-trip legs; 50 lb of feed for 70.",
];

const deltas = [
  {
    label: "Dates and arithmetic, computed by code",
    detail: "Before the model reads, a code-written block resolves every relative date against the day it was said, states the gaps, and lists the quantities the question needs — each line quoting its sentence. No training, no extra model call.",
    value: "+29",
  },
  {
    label: "Trained to show its working",
    detail: "The reader learned to write the dated facts it relies on and the arithmetic before answering — then one final answer line. Only that line is graded.",
    value: "+10",
  },
  {
    label: "Smarter selection of which chats to include",
    detail: "The unit of retrieved chat is chosen from the question itself — whole sessions for some question types, single turns for others — with the date notes rebuilt to match.",
    value: "+17",
  },
  {
    label: "Re-ranked shortlist",
    detail: "A typed re-ranker reorders the candidate chats so the ones carrying the answer actually reach the model. Answer-bearing chats in context rose from 84% to 95%, for $0.28 of calls.",
    value: "+13",
  },
];

export default function ResearchPage() {
  return (
    <main className="marketing-home">
      <header className="site-header">
        <a className="brand" href="/" aria-label="Back to the Remembero site">remembero</a>
        <nav className="desktop-nav" aria-label="Research navigation">
          <a href="#the-gap">The gap</a><a href="#the-fix">The fix</a><a href="#the-ledger">The ledger</a><a href="#the-models">The models</a><a href="#the-examples">Examples</a>
        </nav>
        <div className="header-actions">
          <a className="button secondary desktop-source" href={github}>View on GitHub</a>
        </div>
      </header>

      <section className="research section" aria-labelledby="research-intro-title">
        <div className="section-shell">
          <p className="section-tag">The evidence</p>
          <h2 id="research-intro-title">Every claim on this site<br />comes with its <em>measurement.</em></h2>
          <p className="section-lede">
            The benchmark is LongMemEval-S: 500 questions asked against long, simulated chat histories —
            questions that span sessions, depend on dates, track updates, and punish guessing. Answers are
            graded by an automated grader, and every grader is named with its score, because graders differ
            in leniency. Each improvement below was measured as a controlled pair: the same model, the same
            retrieved chats, exactly one change. Run the same setup twice and scores move by about ±7 —
            any claim smaller than that is noise, and we do not make it.
          </p>
          <div className="extract-strip" aria-label="Knowledge extracted from raw text">
            <div className="extract-raw">
              <span>Raw text · chat, 12 January</span>
              <p>“We signed Norsk Dental as a client yesterday. I&apos;m the account lead. Their contract runs twelve months and is worth $84,000.”</p>
            </div>
            <div className="extract-arrow" aria-hidden="true">the writer extracts</div>
            <ul className="extract-facts">
              <li>
                <strong>Norsk Dental became a client</strong>
                <small className="extract-machine">client_of(norsk_dental, us) · from 2026-01-11</small>
                <em>“signed … as a client yesterday”</em>
              </li>
              <li>
                <strong>You led the account — superseded 1 May</strong>
                <small className="extract-machine">account_lead(norsk_dental, you) · 2026-01-12 → 2026-05-01</small>
                <em>“I&apos;m the account lead.”</em>
              </li>
              <li>
                <strong>The contract is worth $84,000 over twelve months</strong>
                <small className="extract-machine">contract_value(norsk_dental, $84,000 / 12 mo)</small>
                <em>“worth $84,000”</em>
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section className="difference section-dark" id="the-gap">
        <div className="section-shell">
          <p className="section-tag">01 · The gap</p>
          <h2>A small model reads fine.<br />It computes <em>wrong.</em></h2>
          <p className="section-lede">
            We handed a small open model every piece of evidence for each question — perfect retrieval — and
            it still lost half the questions that span sessions or dates. Reading the logs, the failures had
            one shape: the right sentences, the wrong arithmetic. This is what it got wrong, in its own answers:
          </p>
          <ul className="miss-list miss-list-ink">
            {failures.map((failure) => (
              <li key={failure}><span>{failure}</span></li>
            ))}
          </ul>
          <p className="section-lede">
            A model can read a sentence; it cannot be trusted to subtract two timestamps or anchor
            “two weeks ago” to the day it was said. So we stopped asking it to.
          </p>
        </div>
      </section>

      <section className="research section" id="the-fix" aria-labelledby="the-fix-title">
        <div className="section-shell">
          <p className="section-tag">02 · The fix</p>
          <h2 id="the-fix-title">Let code do the <em>arithmetic.</em></h2>
          <div className="research-grid">
            <div>
              <p className="section-lede" style={{ marginTop: 0 }}>
                Before the model reads, deterministic code scans the retrieved chats and writes a block of
                notes: every relative date resolved against the day it was said, gaps between the events the
                question asks about, and the quantities that belong to the question — each line quoting the
                sentence it came from.
              </p>
              <p className="section-lede">
                The block is identical on every run, costs no model call, and never guesses. The model's job
                shrinks to what it is good at: reading.
              </p>
              <div className="pair-callout">
                <strong>318 <i>→ 359</i></strong>
                <span>The same small model, the same retrieved chats, before and after adding the code-written notes — 41 more questions right out of 500. One controlled pair, one automated grader (GPT-4o), no training involved.</span>
              </div>
            </div>
            <div>
              <div className="note-block">
                <div className="note-block-header"><span>Inserted before the model reads</span><span>0 model calls</span></div>
                <pre>{computedNotesSample}</pre>
              </div>
              <p className="delta-footnote">A real block from the reading lab's fixture — <a href="/labs/reading-recall">watch it being written live in your browser</a>.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="research section section-tint" id="the-ledger" aria-labelledby="the-ledger-title">
        <div className="section-shell">
          <p className="section-tag">03 · The ledger</p>
          <h2 id="the-ledger-title">Then train on the <em>structure.</em></h2>
          <p className="section-lede">
            With the notes proven, we distilled a small reader from its frontier teacher — trained on real
            sessions with the notes present, and taught to write its own working before answering. Four
            measured steps, each a controlled pair under one grader (DeepSeek), took it to within fifteen
            questions of the model that trained it.
          </p>
          <div className="delta-ledger" aria-labelledby="delta-title">
            <div className="delta-heading">
              <h3 id="delta-title">How the small reader closed on its teacher</h3>
              <span>controlled pairs · one grader (DeepSeek) · noise ±7</span>
            </div>
            {deltas.map((delta) => (
              <div className="delta-item" key={delta.label}>
                <div><strong>{delta.label}</strong><small>{delta.detail}</small></div>
                <code>{delta.value}</code>
                <span className="delta-chip" aria-label="questions gained">▲</span>
              </div>
            ))}
            <div className="delta-item">
              <div><strong>The small reader, full system — against its teacher</strong><small>An open 4-billion-parameter model, distilled from GLM 5.3 Flash — the same frontier model that taught it, and the yardstick. Fifteen questions short on the 500.</small></div>
              <code>425<i>/500</i></code>
              <span className="delta-chip teacher">teacher 440</span>
            </div>
            <p className="delta-footnote">Each row is a measured pair from <a href={readerDoc}>the method and every run</a> — including the runs that failed: every negative result in the log is published with its configuration.</p>
          </div>
        </div>
      </section>

      <section className="models section-dark" id="the-models" aria-labelledby="the-models-title">
        <div className="section-shell">
          <div className="models-heading">
            <div>
              <p className="section-tag">04 · The models</p>
              <h2 id="the-models-title">Two small open models. <em>Measured, never served.</em></h2>
              <p>Both are fine-tunes of open-weights models, small enough to run on a laptop. Neither is hosted, downloadable, or required by this site — they live in the repository as research artifacts, with every run that measured them.</p>
            </div>
            <a className="button secondary" href={modelComparison}>Model comparison</a>
          </div>
          <div className="not-served">
            <span className="stamp stamp-ink">Not served here</span>
            <p><b>This site ships no model weights and calls no model API.</b> The playground and labs run deterministic code in your browser. Where model output appears, it is either a clearly labeled replay of a recorded run or the optional third-party open model your own browser loads via WebLLM.</p>
          </div>
          <div className="model-grid">
            <article>
              <span>Writer · 2.3B</span>
              <h3>The translator</h3>
              <p>Turns chat and documents into facts, authors the query, extracts policy claims. A fine-tune of Gemma 4 E2B, served as a 4.6 GiB quantized file under llama.cpp — a laptop scores what a cloud GPU scores.</p>
              <dl>
                <div><dt>Extraction benchmark</dt><dd>87<i>/103</i></dd></div>
                <div><dt>Query authoring</dt><dd>27<i>/31</i></dd></div>
                <div><dt>Policy decisions, unseen fixture</dt><dd>20<i>/20</i></dd></div>
                <div><dt>Unjustified approvals</dt><dd>0</dd></div>
              </dl>
            </article>
            <article>
              <span>Reader · 4B</span>
              <h3>The answerer</h3>
              <p>Reads retrieved chat history and answers, handed the code-written notes before it starts. A fine-tune of Gemma 4 E4B, distilled from GLM 5.3 Flash over real sessions — the teacher is the yardstick it is measured against.</p>
              <dl>
                <div><dt>Long-memory benchmark, 500 questions</dt><dd>425<i>/500</i></dd></div>
                <div><dt>Teacher, same grader</dt><dd>440<i>/500</i></dd></div>
                <div><dt>Code-written notes, no training</dt><dd>+29</dd></div>
                <div><dt>Trained to show its working</dt><dd>+10</dd></div>
                <div><dt>Smarter turn selection</dt><dd>+17</dd></div>
                <div><dt>Re-ranked shortlist</dt><dd>+13</dd></div>
              </dl>
            </article>
          </div>
          <div className="model-ledger">
            <span><strong>Notes before reading.</strong> Every relative date resolved against when it was said, every line quoting its sentence.</span>
            <span><strong>Structure before chats.</strong> The memory's own facts, dated and deduplicated, placed before the raw conversations.</span>
            <span><strong>One call to translate.</strong> None to phrase — recalled facts never leave the process.</span>
            <span><strong>Empty results explain themselves.</strong> The engine names the goal that matched nothing and the swap that would return rows.</span>
          </div>
          <p className="models-note">Local embeddings (nomic-embed-text) tie the hosted model on the semantic route. <a href={readerDoc}>Method and every run</a>.</p>
        </div>
      </section>

      <section className="examples section" id="the-examples" aria-labelledby="the-examples-title">
        <div className="section-shell">
          <p className="section-tag">05 · The examples</p>
          <h2 id="the-examples-title">Worked examples, with the <em>numbers attached.</em></h2>
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
              <span>Long-memory benchmark</span>
              <strong>500 questions over long chat histories</strong>
              <p>Multi-session, temporal, knowledge-update, abstention. Every stored run re-graded under one grader; controlled pairs, identical retrieval, so a change is measured against the reader&apos;s own noise.</p>
              <code>teacher 440 · our reader 425 · noise band ±7</code>
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

      <section className="final-cta section">
        <div className="section-shell final-cta-grid">
          <div><h2>Don't take our word for <em>any of it.</em></h2><p>Run the sixty-second demo, work a lab, or reproduce a benchmark from the repository — the claim and the check ship together.</p></div>
          <div className="final-actions"><a className="button primary" href="/">Back to the site</a><a className="button link-button" href="/playground">Open the playground <span aria-hidden="true">→</span></a></div>
        </div>
      </section>

      <footer className="site-footer">
        <strong>remembero</strong>
        <nav aria-label="Footer navigation"><a href="/">Site</a><a href="/labs/reading-recall">Reading lab</a><a href="/labs/chat-memory">Chat lab</a><a href="/labs/grounded-agent">Agent lab</a><a href="/playground">Playground</a><a href={github}>GitHub</a><a href={`${github}#readme`}>Docs</a><span>MIT licensed</span></nav>
      </footer>
    </main>
  );
}
