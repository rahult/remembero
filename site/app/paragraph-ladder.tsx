"use client";

import { useState } from "react";
import {
  CONDITIONAL_RULE,
  FACTS,
  LADDER_REPLAY_NOTE,
  LADDER_SCOPE_NOTE,
  PARAGRAPH,
  PARAGRAPH_SAID_ON,
  buildChange,
  computeLogic,
  resolveDates,
} from "../lib/paragraph-ladder";
import styles from "./paragraph-ladder.module.css";

const LAYERS = [
  { id: 0, label: "Facts", hint: "who and what" },
  { id: 1, label: "Dates", hint: "when, resolved" },
  { id: 2, label: "Logic", hint: "arithmetic and rules" },
  { id: 3, label: "Change", hint: "truth over time" },
] as const;

function MachineForm({ form }: { form: string }) {
  return <small className={styles.machine}>machine form: <code>{form}</code></small>;
}

export function ParagraphLadder() {
  const [revealed, setRevealed] = useState(1);
  const dates = resolveDates();
  const logic = computeLogic();
  const change = buildChange();

  return (
    <div className={styles.ladder} aria-label="One paragraph, taken apart">
      <blockquote className={styles.paragraph}>
        <p>{PARAGRAPH}</p>
        <cite>one chat message · said {PARAGRAPH_SAID_ON}</cite>
      </blockquote>

      <ol className={styles.rail} aria-label="Extraction layers">
        {LAYERS.map((layer) => (
          <li key={layer.id} data-state={layer.id < revealed ? "done" : layer.id === revealed - 1 ? "next" : "todo"}>
            <b>{layer.id + 1}</b>
            <div>
              <span>{layer.label}</span>
              <em>{layer.hint}</em>
            </div>
          </li>
        ))}
      </ol>

      <div className={styles.layers}>
        {revealed >= 1 ? (
          <section className={styles.layer} aria-labelledby="ladder-facts">
            <header>
              <h3 id="ladder-facts">1 · The facts it states</h3>
              <span className={`${styles.stamp} ${styles.stampAmber}`}>Writer · replay</span>
            </header>
            <ul className={styles.factGrid}>
              {FACTS.map((fact) => (
                <li key={fact.machine}>
                  <strong>{fact.headline}</strong>
                  <MachineForm form={fact.machine} />
                  <em>“{fact.quote}”</em>
                </li>
              ))}
              <li className={styles.ruleCard}>
                <strong>{CONDITIONAL_RULE.headline}</strong>
                <MachineForm form={CONDITIONAL_RULE.machine} />
                <em>“{CONDITIONAL_RULE.quote}”</em>
              </li>
            </ul>
            <p className={styles.layerNote}>The writer reads English so the rest of the system never has to. A sentence becomes a claim that carries its own source.</p>
          </section>
        ) : null}

        {revealed >= 2 ? (
          <section className={styles.layer} aria-labelledby="ladder-dates">
            <header>
              <h3 id="ladder-dates">2 · The dates it hides</h3>
              <span className={`${styles.stamp} ${styles.stampGreen}`}>Code · live</span>
            </header>
            <ul className={styles.dateList}>
              {dates.map((date) => (
                <li key={date.expression}>
                  <code>“{date.expression}”</code>
                  <span>→</span>
                  <strong>{date.resolved}</strong>
                  <em>{date.daysAfterSaid} days after it was said</em>
                  <small>“{date.sentence}”</small>
                </li>
              ))}
            </ul>
            <p className={styles.layerNote}>Relative dates are where readers — human and model — slip most. Code resolves each one against the day it was said, and the resolution carries its sentence.</p>
          </section>
        ) : null}

        {revealed >= 3 ? (
          <section className={styles.layer} aria-labelledby="ladder-logic">
            <header>
              <h3 id="ladder-logic">3 · The logic it implies</h3>
              <span className={`${styles.stamp} ${styles.stampGreen}`}>Code · live</span>
            </header>
            <ul className={styles.logicList}>
              {logic.map((line) => (
                <li key={line.machine}>
                  <strong>{line.label}</strong>
                  <p>{line.detail}</p>
                  <MachineForm form={line.machine} />
                </li>
              ))}
            </ul>
            <p className={styles.layerNote}>The arithmetic is checked, and the conditional stays conditional: an unmet “if” never becomes a fact.</p>
          </section>
        ) : null}

        {revealed >= 4 ? (
          <section className={styles.layer} aria-labelledby="ladder-change">
            <header>
              <h3 id="ladder-change">4 · The change it announces</h3>
              <span className={`${styles.stamp} ${styles.stampGreen}`}>Code · live</span>
            </header>
            <div className={styles.bands}>
              {change.map((band) => (
                <article key={band.machine}>
                  <header>
                    <strong>{band.fact}</strong>
                    <MachineForm form={band.machine} />
                  </header>
                  <ul>
                    {band.spans.map((span) => (
                      <li key={span.value} data-status={span.status}>
                        <strong>{span.value}</strong>
                        <span>{span.from} → {span.to ?? "now"}</span>
                        <em>{span.status}</em>
                      </li>
                    ))}
                  </ul>
                </article>
              ))}
            </div>
            <p className={styles.layerNote}>One paragraph, and the lead has already changed hands and the launch already moved once. Every value gets an interval, so “now” is a lookup — and history is still there when you ask about March.</p>
          </section>
        ) : null}
      </div>

      <footer className={styles.ladderFooter}>
        {revealed < 4 ? (
          <button type="button" className={styles.advance} onClick={() => setRevealed((n) => Math.min(n + 1, 4))}>
            {revealed === 1 ? "Resolve the dates" : revealed === 2 ? "Check the logic" : "Track the change"}
          </button>
        ) : (
          <button type="button" className={styles.reset} onClick={() => setRevealed(1)}>
            Take it apart again
          </button>
        )}
        <p>{LADDER_REPLAY_NOTE}</p>
      </footer>
      <p className={styles.scope}>{LADDER_SCOPE_NOTE}</p>
    </div>
  );
}
