# Model comparison and current preference

One page on which models were tried in each role of the memory system, how they compare on
the same benchmarks, and which one is preferred today. Numbers come from the result files in
`results/`; the full per-run table is [RUN-MATRIX.md](RUN-MATRIX.md), and the plan for moving
every remaining role onto our own models is [SELF-HOSTED-ROADMAP.md](SELF-HOSTED-ROADMAP.md).
Last updated 2026-09-11.

The system has four model roles. A **writer** turns text into Datalog facts and turns
questions into Datalog queries (one fine-tuned adapter does both). A **reader** answers a
question from retrieved sessions. A **time-range extractor** reads a date range off temporal
questions. A **labeller** produces training data from real transcripts. The judge (GPT-4o) is
fixed by the LongMemEval protocol and is not a choice.

## Writer: extraction and query authoring

Same prompts and write-side guards for every row. Extraction is the 103-case schema-conditioned
benchmark (closed vocabulary); query is query-correct out of 31 on the agent-boundary benchmark
(closure condition). LongMemEval is hybrid formation with the model as extractor, all 500
questions, Luna reading, lexical plus semantic retrieval, multi-session k 15 and temporal k 10
with a time range (the composed policy); it is only available for the models that were served.

| model                              | tuned? | size (effective) | extraction /103 | query /31 |                                 LongMemEval /500 | serve cost                                |
| ---------------------------------- | ------ | ---------------- | --------------: | --------: | -----------------------------------------------: | ----------------------------------------- |
| Llama 3.2 3B instruct              | no     | 3B               |              35 |        19 |                                                – | local Ollama                              |
| Qwen2.5-coder 7B                   | no     | 7B               |              84 |        26 |                                                – | local Ollama                              |
| Llama 3.2 3B, r4–r6 fine-tunes     | yes    | 3B               |           61–73 |     26–27 |                                                – | Tinker sampler                            |
| Qwen3.5-4B, r7–r15 fine-tunes      | yes    | 4B               |     85–94 (r13) |     26–29 | 416 (r16 policy, r14 extractor 414 lexical-only) | Modal L4 ≈$0.80/h                         |
| Gemma 4 E4B, r16                   | yes    | 4.5B             |              84 |        24 |                                                – | needs A100/L40S                           |
| **Gemma 4 E2B, r16 (served)**      | yes    | 2.3B             |           85–86 |     27–28 |                                          **416** | Modal L4 ≈$0.80/h                         |
| Gemma 4 E2B, r17 (+ real sessions) | yes    | 2.3B             |              81 |        23 |                      201 (dev-261 scale, shared) | Modal L4                                  |
| openai/gpt-5.6-luna (frontier)     | no     | –                |              96 |        29 |                  33/48 on the slice as extractor | OpenRouter, ≈$2.35 per 48 q of extraction |
| z-ai/glm-5.3 (frontier)            | no     | –                |               – |        30 |                                                – | Ollama Cloud                              |

Reading the table:

- Fine-tuning is what closed the gap. Untuned 3B scores 35 on extraction; the same size class
  tuned scores 85 to 94, two to eleven cases behind Luna. The data rounds, not the base model,
  did that (see [EXTRACTION-BENCH.md](EXTRACTION-BENCH.md)).
- Four Qwen3.5-4B runs on identical data spanned 85 to 91 extraction cases and 26 to 27
  queries, so differences inside that band between any two rows are noise.
- Gemma 4 E2B ties Qwen3.5-4B on both benchmarks at half the size, with a plain chat template
  and no special attention kernels, and repeated the result on a second run (27, 85). It is
  the served writer. Gemma 4 E4B was a query below the band on one run and needs a bigger GPU
  to serve; not pursued.
- r17, trained on Luna-labelled real sessions, nearly tripled loose fact recall on real
  transcripts (10% to 29%) but writes three times as many facts, which crowds raw sessions
  out of shared retrieval; it loses on LongMemEval under the current retrieval. Kept as an
  adapter, not served.
- As an extractor on the LongMemEval slice, Luna was not better than the fine-tune (33 vs 37
  of 48, inside noise) and cost about a hundred times more per session.

**Current preference: Gemma 4 E2B r16**, served on Modal (`SERVE_RUN=r16-gemma4-e2b`), with
Qwen3.5-4B r14 kept as the fallback checkpoint.

## Reader

All rows use the same retrieval (hybrid formation with the E2B extractor, semantic route,
multi-session k 15, temporal k 10 with a Luna time range) and the same extraction cache, so the
reader is the only difference. "Aggregation types" are multi-session, temporal-reasoning and
knowledge-update (344 of 500 questions); the other 156 stayed with Luna where a split is shown.

| reader                                               | where        | LongMemEval /500 | multi /133 | temporal /133 | preference /30 | reader cost for 500 |
| ---------------------------------------------------- | ------------ | ---------------: | ---------: | ------------: | -------------: | ------------------- |
| openai/gpt-5.6-luna, all types                       | OpenRouter   |              416 |        103 |           110 |             23 | ≈$0.20              |
| glm-5.3, aggregation types                           | Ollama Cloud |              433 |        109 |           116 |             26 | subscription        |
| glm-5.3-flash, aggregation types                     | Ollama Cloud |              435 |        108 |           115 |             27 | subscription        |
| **glm-5.3-flash, all types**                         | Ollama Cloud |          **432** |    **113** |           116 |             28 | subscription        |
| deepseek-v4-flash (0731), aggregation types          | Ollama Cloud |              413 |         99 |           113 |             26 | subscription        |
| deepseek/deepseek-v4.1-flash, aggregation, 4k budget | OpenRouter   |  423 (16 errors) |        104 |           116 |             21 | ≈$0.30              |
| deepseek/deepseek-v4.1-flash, aggregation, 16k       | OpenRouter   |              428 |        103 |           114 |             28 | $0.57               |

Reading the table:

- The reader, not retrieval, was the limit once recall reached 92%: with Luna, 43 questions
  with every evidence session in context were still wrong, 21 of them multi-session counts.
  GLM 5.3 fixes most of those; its accuracy on fully-evidenced questions is 92.5% against
  Luna's 89.4%.
- GLM 5.3 Flash equals the full model. Splitting readers (Flash for aggregation, Luna for the
  rest) and using Flash for everything are inside each other's noise (435 vs 432); one reader
  is the simpler system and posts the best multi-session score.
- DeepSeek v4 Flash is below Luna. DeepSeek v4.1 Flash is a reasoning model that empties a
  4k completion budget thinking; at 16k it lands between Luna and GLM.
- Two reader-side prompt patterns from the LongMemEval paper (dated notes then answer, and a
  two-call enumerate-then-answer) did not help Luna: 416 and 401. The gain came from the model.

**Current preference: glm-5.3-flash on Ollama Cloud for every type**, via the local daemon
(`--reader-model glm-5.3-flash:cloud --reader-base-url http://127.0.0.1:11434/v1`).

## Time-range extractor (temporal questions)

Only Luna has been tried. It returned a range for 20 of 66 dev temporal questions and refused
the rest, which is the behaviour the paper found necessary (a small model that guesses ranges
hurts). With temporal k 10 it took temporal from 52 to 56 of 66 on dev and 104 to 113–116 of
133 on the full set. One call per temporal question. **Current preference: Luna**; GLM 5.3
Flash would likely do the same job on the subscription and has not been tested in this role.

## Labeller (training data from real transcripts)

Only Luna has been used, through the product's own transcript extraction path with the guards
on. It labelled 3,400 real sessions for about $6, 2,459 with facts, 6.8 facts per session with
facts; its labels are verbose (`controller_is_new(gaming_controller)`), which is what pushed
r17 toward over-writing. **Current preference: Luna, with a cap on facts per session** before
the next real-session round; a cheaper labeller on the Ollama subscription (GLM 5.3) is the
untested alternative.

## Where each preference could change

- **Writer**: a second E2B run confirmed the benchmark numbers but the LongMemEval number is
  one run; r16b has not been run there. If a future round needs real-session data, cap Luna's
  facts per session so the writer does not inherit r17's verbosity.
- **Reader**: kimi-k3, deepseek-v4-pro and qwen3.5:397b are on the subscription and untested;
  each is one cached run of about 45 minutes. The judge-and-reader noise is about six answers
  per 261 questions, so a new reader needs to clear roughly ten on 500 to count.
- **Everything else**: costs are small enough that the constraint is wall-clock, not money.
  A full-500 LongMemEval run from the extraction cache is about 40 minutes and under a dollar
  of provider spend.
