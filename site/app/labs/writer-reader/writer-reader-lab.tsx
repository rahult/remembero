"use client";

import { useMemo, useState } from "react";
import {
  QUESTIONS,
  QUESTION_DATE,
  READER_REPLAY_NOTE,
  SCOPE_NOTE,
  SESSIONS,
  WRITER_CLAIMS,
  WRITER_REPLAY_NOTE,
  buildReaderNotes,
  buildTimelines,
  retrieveForQuestion,
} from "../../../lib/writer-reader-fixture";
import styles from "./writer-reader-lab.module.css";

function StatusChip({ status }: { status: "current" | "superseded" | "ended" }) {
  if (status === "current") return <span className={`${styles.chip} ${styles.chipCurrent}`}>current</span>;
  if (status === "ended") return <span className={`${styles.chip} ${styles.chipEnded}`}>ended</span>;
  return <span className={`${styles.chip} ${styles.chipSuperseded}`}>superseded</span>;
}

function Verdict({ arm }: { arm: { verdict: "miss" | "hit" | "unknown-honest" } }) {
  if (arm.verdict === "hit") return <span className={`${styles.verdict} ${styles.verdictHit}`}>✓ matches gold</span>;
  if (arm.verdict === "unknown-honest") return <span className={`${styles.verdict} ${styles.verdictHit}`}>✓ refuses to invent</span>;
  return <span className={`${styles.verdict} ${styles.verdictMiss}`}>✗ judged wrong</span>;
}

export function WriterReaderLab() {
  const [writerStep, setWriterStep] = useState(0);
  const [questionId, setQuestionId] = useState(QUESTIONS[0].id);

  const question = QUESTIONS.find((q) => q.id === questionId) ?? QUESTIONS[0];
  const timelines = useMemo(() => buildTimelines(WRITER_CLAIMS), []);
  const retrieval = useMemo(
    () => retrieveForQuestion(question.question, timelines),
    [question, timelines],
  );
  const notes = useMemo(
    () => buildReaderNotes(question.question, timelines),
    [question, timelines],
  );

  return (
    <main className={styles.lab}>
      <header className={styles.topbar}>
        <a href="/" aria-label="Back to the Remembero site">remembero<span aria-hidden="true">.</span></a>
        <span className={styles.topbarTitle}>Writer–reader lab</span>
        <span className={styles.runtimeLine}>deterministic structure · replays labeled · no model executes on this page</span>
      </header>

      <section className={styles.intro}>
        <div>
          <p className={styles.eyebrow}>The whole loop · on genuinely messy history</p>
          <h1>Raw text in. Proven answers out.<br /><em>Even when the truth keeps moving.</em></h1>
          <p className={styles.lede}>
            Six months of one person's chat — {SESSIONS.length} sessions, asked on {QUESTION_DATE}. People
            correct themselves, switch providers and switch back at a different price, hand work over on
            future dates, and end contracts. Watch the writer turn that text into claims, watch code give
            every claim a validity interval, then ask the questions that punish naive retrieval.
          </p>
        </div>
        <div className={styles.stampRow} aria-label="Execution boundary">
          <span className={`${styles.stamp} ${styles.stampGreen}`}>Structure · live</span>
          <span className={`${styles.stamp} ${styles.stampAmber}`}>Writer & reader replayed</span>
          <span className={`${styles.stamp} ${styles.stampInk}`}>No weights served</span>
        </div>
      </section>

      <div className={styles.shell}>
        <section className={styles.act} aria-labelledby="act-writer">
          <div className={styles.actHead}>
            <span className={styles.actNumber}>1</span>
            <div>
              <h2 id="act-writer">The writer: text becomes claims</h2>
              <p>
                A small writer model reads each session and returns claims — subject, predicate, object,
                the sentence it came from, and whether the sentence states, corrects, or ends a value.
                The claim rows below are a <b>replay</b>; what happens to them next is code, live.
              </p>
            </div>
          </div>
          <div className={styles.writerGrid}>
            <div className={styles.rawPanel} aria-label="Raw sessions">
              <h3>Raw text · {SESSIONS.length} sessions</h3>
              <ol>
                {SESSIONS.map((session) => (
                  <li key={session.id}>
                    <code>{session.date} · {session.label}</code>
                    <p>{session.text}</p>
                  </li>
                ))}
              </ol>
            </div>
            <div className={styles.claimPanel} aria-label="Writer output and resolution">
              {writerStep === 0 ? (
                <div className={styles.stepWait}>
                  <p>The writer's output for these six sessions is ready to replay.</p>
                  <button type="button" className={styles.advance} onClick={() => setWriterStep(1)}>
                    Replay the writer's claims
                  </button>
                </div>
              ) : null}
              {writerStep >= 1 ? (
                <div className={styles.claimBlock}>
                  <div className={styles.blockHeader}>
                    <span>Claims · replay of a recorded writer run</span>
                    <span className={`${styles.stamp} ${styles.stampAmber}`}>Replay</span>
                  </div>
                  <ul>
                    {WRITER_CLAIMS.map((claim) => (
                      <li key={claim.id}>
                        <code>{claim.subject} · {claim.predicate}</code>
                        <strong>{claim.object}</strong>
                        <span>“{claim.quote}”</span>
                        <em data-kind={claim.kind}>{claim.kind}</em>
                      </li>
                    ))}
                  </ul>
                  {writerStep === 1 ? (
                    <div className={styles.resolveWait}>
                      <p>Now the deterministic part: group by subject and predicate, order in time, and give every value a validity interval.</p>
                      <button type="button" className={styles.advance} onClick={() => setWriterStep(2)}>
                        Resolve · date · supersede
                      </button>
                    </div>
                  ) : null}
                </div>
              ) : null}
              {writerStep >= 2 ? (
                <div className={styles.timelineBlock} aria-live="polite">
                  <div className={styles.blockHeader}>
                    <span>Validity timelines · computed in your browser</span>
                    <span className={`${styles.stamp} ${styles.stampGreen}`}>Live · 0 models</span>
                  </div>
                  <table className={styles.timelineTable}>
                    <thead>
                      <tr><th>fact</th><th>value</th><th>valid</th><th>state</th></tr>
                    </thead>
                    <tbody>
                      {timelines.map((row) => (
                        <tr key={row.key} data-status={row.status}>
                          <td><code>{row.subject} · {row.predicate}</code></td>
                          <td>{row.value}</td>
                          <td><span>{row.from}</span> → <span>{row.to ?? "open"}</span></td>
                          <td><StatusChip status={row.status} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className={styles.blockNote}>{WRITER_REPLAY_NOTE}</p>
                </div>
              ) : null}
            </div>
          </div>
        </section>

        <section className={styles.act} aria-labelledby="act-reader">
          <div className={styles.actHead}>
            <span className={styles.actNumber}>2</span>
            <div>
              <h2 id="act-reader">The reader: questions over moving truth</h2>
              <p>
                Four questions, each built around a shape that breaks naive memory. Retrieval scoring and
                the notes are code in your browser; both reader answers are labeled replays — same history,
                raw chats versus the structured evidence above.
              </p>
            </div>
          </div>
          <div className={styles.questionBar} role="tablist" aria-label="Questions">
            {QUESTIONS.map((q) => (
              <button
                key={q.id}
                role="tab"
                aria-selected={q.id === question.id}
                className={q.id === question.id ? styles.questionTabActive : styles.questionTab}
                onClick={() => setQuestionId(q.id)}
              >
                <span className={styles.questionKind}>hard shape: {q.kindLabel}</span>
                <span className={styles.questionText}>{q.question}</span>
              </button>
            ))}
          </div>

          <p className={styles.hardBecause}>
            <b>Why this is hard:</b> {question.hardBecause} <b>Gold:</b> {question.gold}.
          </p>

          <div className={styles.readerGrid}>
            <section className={styles.retrievalPanel} aria-label="What retrieval returns">
              <h3>Retrieved evidence <span>scored by code</span></h3>
              <ul>
                {retrieval.map(({ row, score, reason }) => (
                  <li key={row.key} data-status={row.status}>
                    <div>
                      <code>{row.subject} · {row.predicate}</code>
                      <strong>{row.value}</strong>
                      <span>{reason}</span>
                    </div>
                    <StatusChip status={row.status} />
                  </li>
                ))}
              </ul>
              <p className={styles.panelNote}>
                Superseded and ended rows still surface — that is the point. Raw retrieval cannot tell a
                stale row from a live one; the interval column can.
              </p>
            </section>
            <section className={styles.notesPanel} aria-label="Structured evidence notes">
              <h3>Notes before reading <span>written by code</span></h3>
              <div className={styles.notesBlock}>
                <div className={styles.notesHeader}>
                  <span>as of {QUESTION_DATE}</span>
                  <span>0 model calls</span>
                </div>
                <pre>{notes}</pre>
              </div>
            </section>
          </div>

          <div className={styles.armGrid}>
            <article className={styles.armMiss}>
              <header>
                <span>{question.rawSessionsArm.label}</span>
                <span className={`${styles.stamp} ${styles.stampAmber}`}>Replay</span>
              </header>
              <p className={styles.armAnswer}>{question.rawSessionsArm.answer}</p>
              <Verdict arm={question.rawSessionsArm} />
              <p className={styles.armNote}>{question.rawSessionsArm.verdictNote}</p>
            </article>
            <article className={styles.armHit}>
              <header>
                <span>{question.structuredArm.label}</span>
                <span className={`${styles.stamp} ${styles.stampAmber}`}>Replay</span>
              </header>
              <p className={styles.armAnswer}>{question.structuredArm.answer}</p>
              <Verdict arm={question.structuredArm} />
              <p className={styles.armNote}>{question.structuredArm.verdictNote}</p>
            </article>
          </div>
          <p className={styles.replayNote}>{READER_REPLAY_NOTE}</p>
        </section>

        <footer className={styles.labFooter}>
          <p>{SCOPE_NOTE}</p>
          <p className={styles.resetNote}>Nothing is stored and nothing leaves this tab — refresh to reset.</p>
        </footer>
      </div>
    </main>
  );
}
