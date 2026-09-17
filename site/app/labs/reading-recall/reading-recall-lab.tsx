"use client";

import { useMemo, useState } from "react";
import {
  QUESTION_DATE,
  QUESTIONS,
  REPLAY_NOTE,
  RERANK_NOTE,
  SCOPE_NOTE,
  buildComputedNotes,
  runPipeline,
} from "../../../lib/reading-recall-fixture";
import styles from "./reading-recall-lab.module.css";

const readerDoc =
  "https://github.com/rahult/remembero/blob/main/docs/research/READER-STRUCTURE.md";

const kindLabel: Record<string, string> = {
  temporal: "temporal reasoning",
  quantity: "arithmetic over quantities",
};

function Verdict({ hit }: { hit: boolean }) {
  return (
    <span className={hit ? styles.verdictHit : styles.verdictMiss}>
      {hit ? "✓ matches gold" : "✗ judged wrong"}
    </span>
  );
}

export function ReadingRecallLab() {
  const [questionId, setQuestionId] = useState(QUESTIONS[0].id);
  const [rerank, setRerank] = useState(true);

  const question = QUESTIONS.find((q) => q.id === questionId) ?? QUESTIONS[0];
  const pipeline = useMemo(() => runPipeline(question, rerank), [question, rerank]);
  const notes = useMemo(() => buildComputedNotes(question.question), [question]);
  const lexicalById = new Map(
    pipeline.lexicalOrder.map((item, index) => [item.session.id, index + 1]),
  );

  return (
    <main className={styles.lab}>
      <header className={styles.topbar}>
        <a href="/" aria-label="Back to the Remembero site">remembero<span aria-hidden="true">.</span></a>
        <span className={styles.topbarTitle}>Reading recall lab</span>
        <span className={styles.runtimeLine}>deterministic pipeline · no model executes on this page</span>
      </header>

      <section className={styles.intro}>
        <div>
          <p className={styles.eyebrow}>The reader contract · paired-run replay</p>
          <h1>Watch the reading pipeline assemble — <em>then see what the notes bought.</em></h1>
          <p className={styles.lede}>
            Retrieval, re-ranking, context tiering, and the computed-notes block below are real code
            running in your browser over a fixed fictional history, asked on {QUESTION_DATE}. The two
            reader answers are labeled replays of a recorded paired run: same reader, same evidence,
            with and without the notes. That pair is the unit the research scores.
          </p>
        </div>
        <div className={styles.stampRow} aria-label="Execution boundary">
          <span className={`${styles.stamp} ${styles.stampGreen}`}>Deterministic · live</span>
          <span className={`${styles.stamp} ${styles.stampAmber}`}>Replayed answers</span>
          <span className={`${styles.stamp} ${styles.stampInk}`}>No weights served</span>
        </div>
      </section>

      <div className={styles.shell}>
        <div className={styles.questionBar} role="tablist" aria-label="Questions">
          {QUESTIONS.map((q) => (
            <button
              key={q.id}
              role="tab"
              aria-selected={q.id === question.id}
              className={q.id === question.id ? styles.questionTabActive : styles.questionTab}
              onClick={() => setQuestionId(q.id)}
            >
              <span className={styles.questionKind}>{kindLabel[q.kind]}</span>
              <span className={styles.questionText}>{q.question}</span>
            </button>
          ))}
        </div>

        <section className={styles.stage} aria-labelledby="stage-retrieve">
          <div className={styles.stageHead}>
            <span className={styles.stageNumber}>1</span>
            <div>
              <h2 id="stage-retrieve">Retrieve the shortlist</h2>
              <p>
                A lexical score counts the question&apos;s content words in each session; ties break
                toward the more recent session.
                <label className={styles.rerankToggle}>
                  <input
                    type="checkbox"
                    checked={rerank}
                    onChange={(event) => setRerank(event.target.checked)}
                  />
                  apply the re-rank stand-in
                </label>
              </p>
            </div>
          </div>
          <div className={styles.sessionTable} aria-live="polite">
            {pipeline.finalOrder.map((item, rank) => (
              <div className={styles.sessionRow} key={item.session.id}>
                <span className={styles.sessionRank}>{rank + 1}</span>
                <div className={styles.sessionMeta}>
                  <code>{item.session.id} · {item.session.date}</code>
                  <span>
                    lexical score {item.lexicalScore}
                    {item.hasDatedExpression ? " · carries a dated expression" : ""}
                  </span>
                </div>
                <span
                  className={pipeline.rerankChangedOrder && lexicalById.get(item.session.id) !== rank + 1
                    ? styles.orderMoved
                    : styles.orderHeld}
                >
                  {lexicalById.get(item.session.id) === rank + 1
                    ? "lexical order"
                    : `moved from #${lexicalById.get(item.session.id)}`}
                </span>
              </div>
            ))}
          </div>
          <p className={styles.stageNote}>{RERANK_NOTE}</p>
        </section>

        <section className={styles.stage} aria-labelledby="stage-tier">
          <div className={styles.stageHead}>
            <span className={styles.stageNumber}>2</span>
            <div>
              <h2 id="stage-tier">Tier the context</h2>
              <p>
                Every retrieved session gets a code-built abstract; only the highest-ranked two
                keep their full text. Tiering replaces cutting every session to the same sliver.
              </p>
            </div>
          </div>
          <div className={styles.tierGrid}>
            {pipeline.finalOrder.map((item) => (
              <article
                key={item.session.id}
                className={item.tier === "full" ? styles.tierFull : styles.tierAbstract}
              >
                <header>
                  <span>{item.tier === "full" ? "full text" : "abstract"}</span>
                  <code>{item.session.id} · {item.session.date}</code>
                </header>
                <p>{item.tier === "full" ? item.session.text : item.abstract}</p>
              </article>
            ))}
          </div>
        </section>

        <section className={styles.stage} aria-labelledby="stage-notes">
          <div className={styles.stageHead}>
            <span className={styles.stageNumber}>3</span>
            <div>
              <h2 id="stage-notes">Write the computed notes</h2>
              <p>
                Rebuilt for this question in front of you: relative dates resolved against the day
                they were said, the gap between the best-matching events, and the quantities that
                belong to the question.
              </p>
            </div>
          </div>
          <div className={styles.notesBlock}>
            <div className={styles.notesHeader}>
              <span>inserted before the reader reads</span>
              <span className={styles.notesZero}>0 model calls · identical on every run</span>
            </div>
            <pre>{notes}</pre>
          </div>
          <p className={styles.stageNote}>{SCOPE_NOTE}</p>
        </section>

        <section className={styles.stage} aria-labelledby="stage-reader">
          <div className={styles.stageHead}>
            <span className={styles.stageNumber}>4</span>
            <div>
              <h2 id="stage-reader">The reader, paired</h2>
              <p>
                Gold for this question: <code>{question.gold}</code>. {REPLAY_NOTE}
              </p>
            </div>
          </div>
          <div className={styles.armGrid}>
            <article className={styles.armMiss}>
              <header>
                <span>{question.withoutNotes.label}</span>
                <span className={`${styles.stamp} ${styles.stampAmber}`}>Replay</span>
              </header>
              <p className={styles.armAnswer}>{question.withoutNotes.answer}</p>
              <Verdict hit={false} />
              <p className={styles.armNote}>{question.withoutNotes.verdictNote}</p>
            </article>
            <article className={styles.armHit}>
              <header>
                <span>{question.withNotes.label}</span>
                <span className={`${styles.stamp} ${styles.stampAmber}`}>Replay</span>
              </header>
              {question.withNotes.thinking ? (
                <pre className={styles.thinking}>{question.withNotes.thinking}</pre>
              ) : null}
              <p className={styles.armAnswer}>{question.withNotes.answer}</p>
              <Verdict hit />
              <p className={styles.armNote}>{question.withNotes.verdictNote}</p>
            </article>
          </div>
        </section>

        <footer className={styles.labFooter}>
          <p>
            The measured version of this pipeline — with a real small reader model, the full parser, and the
            typed re-ranker — is documented run by run in{" "}
            <a href={readerDoc}>the research log</a>: adding the code-written notes alone took the small
            reader from 318 to 359 of 500, and the full system reaches 425 — fifteen short of the frontier
            model that trained it. None of it is served from this site.
          </p>
          <p className={styles.resetNote}>Nothing is stored and nothing leaves this tab — refresh to reset.</p>
        </footer>
      </div>
    </main>
  );
}
