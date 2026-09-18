"use client";

import { useState } from "react";
import {
  DEMO_MAGIC,
  DEMO_QUESTION,
  DEMO_QUERY,
  DEMO_RULE,
  DEMO_RULE_PLAIN,
  DEMO_UTTERANCES,
  runFirstProofDemo,
  type DemoResult,
} from "../lib/first-proof-demo";
import styles from "./first-proof-demo.module.css";

const STEPS = ["Say it", "Store it", "Rule it", "Ask it"] as const;

export function FirstProofDemo() {
  const [step, setStep] = useState(0);
  const [result, setResult] = useState<DemoResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  function advance() {
    setError(null);
    if (step === 2) {
      try {
        setResult(runFirstProofDemo());
      } catch {
        setError("The engine returned no rows — unexpected, and worth a refresh.");
        return;
      }
    }
    setStep((current) => Math.min(current + 1, 3));
  }

  function reset() {
    setStep(0);
    setResult(null);
    setError(null);
  }

  return (
    <div className={styles.demo} id="demo" aria-label="Sixty-second live demo">
      <header className={styles.demoHeader}>
        <span className={styles.demoTitle}>Try the core loop — right here, right now</span>
        <span className={`${styles.stamp} ${styles.stampGreen}`}>Live · 0 models</span>
      </header>

      <ol className={styles.stepRail} aria-label="Demo steps">
        {STEPS.map((label, index) => (
          <li
            key={label}
            data-state={index < step ? "done" : index === step ? "current" : "todo"}
            aria-current={index === step ? "step" : undefined}
          >
            <b>{index + 1}</b>
            {label}
            {index < step ? " ✓" : ""}
          </li>
        ))}
      </ol>

      <div className={styles.demoBody}>
        <section className={styles.stageChat} aria-label="What you said">
          <h3>1 · You say</h3>
          {DEMO_UTTERANCES.map((utterance) => (
            <blockquote key={utterance.fact} className={styles.bubble}>
              <p>“{utterance.said}”</p>
              <cite>{utterance.when}</cite>
            </blockquote>
          ))}
        </section>

        <section className={styles.stageMemory} aria-label="What Remembero stores" data-revealed={step >= 1}>
          <h3>2 · Remembero stores</h3>
          {step < 1 ? (
            <p className={styles.placeholder}>Plain facts, each with the sentence it came from.</p>
          ) : (
            <ul className={styles.factList}>
              {DEMO_UTTERANCES.map((utterance) => (
                <li key={utterance.fact}>
                  <strong>{utterance.plain}</strong>
                  <small className={styles.factMachine}>machine form: <code>{utterance.fact}</code></small>
                  <span>“{utterance.said}”</span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className={styles.stageRule} aria-label="The rule" data-revealed={step >= 2}>
          <h3>3 · You add one rule</h3>
          {step < 2 ? (
            <p className={styles.placeholder}>Reads as: {DEMO_RULE_PLAIN}.</p>
          ) : (
            <div className={styles.ruleCard}>
              <p className={styles.rulePlain}>{DEMO_RULE_PLAIN}</p>
              <pre className={styles.rule}>{DEMO_RULE}</pre>
            </div>
          )}
        </section>

        <section className={styles.stageAsk} aria-label="The question and answer" data-revealed={step >= 3}>
          <h3>4 · You ask</h3>
          {step < 3 ? (
            <p className={styles.placeholder}>“{DEMO_QUESTION}”</p>
          ) : result ? (
            <div className={styles.answerBlock}>
              <p className={styles.question}>“{DEMO_QUESTION}”</p>
              <code className={styles.query}>{result.query}</code>
              <p className={styles.answer}>{result.answer}</p>
              <div className={styles.proof}>
                <span className={styles.proofLabel}>because</span>
                <ol>
                  {result.proofChain.map((line) => (
                    <li key={line}><code>{line}</code></li>
                  ))}
                </ol>
                <span className={styles.magic}>{DEMO_MAGIC}</span>
              </div>
              <span className={styles.duration}>
                evaluated in {result.durationMs.toFixed(1)} ms, in this tab
              </span>
            </div>
          ) : (
            <p className={styles.placeholder}>{error}</p>
          )}
        </section>
      </div>

      <footer className={styles.demoFooter}>
        {step < 3 ? (
          <button type="button" className={styles.advance} onClick={advance}>
            {step === 0 ? "Store these as facts" : step === 1 ? "Compile the rule" : "Run the query"}
          </button>
        ) : (
          <button type="button" className={styles.reset} onClick={reset}>
            Run it again
          </button>
        )}
        <p>
          Deterministic engine executing in your browser — same question, same answer, every time.
          No model, no server, nothing stored. The full SQLite version waits in the{" "}
          <a href="/playground">playground</a>.
        </p>
      </footer>
    </div>
  );
}
