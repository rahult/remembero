# Demos

Three runnable, real-life scenarios. Each one is a folder with a script, its data, a
README that says what you will see, and the captured transcript of one run so you can
read the result before running anything.

| demo | what it shows | needs |
| --- | --- | --- |
| [team-rules](team-rules/) | A team's project memory: rules with proofs, `why-not`, a write the constraint refuses, a what-if simulation, supersession with history | nothing but Node |
| [personal-timeline](personal-timeline/) | Weeks of ordinary statements, then the questions people ask: distances in days, totals, the current price, a promise, an honest no-match | the local writer (llama.cpp) |
| [approvals-desk](approvals-desk/) | A year of "may X approve Y on this date" from a delegations schedule, a Slack message and an amendment; ALLOW, DENY or UNKNOWN with quotes | the local writer (llama.cpp), Python 3.12 |

The local writer is Remembero's own model: a 2.3B Gemma 4 E2B fine-tune served as a Q8_0
GGUF by llama.cpp. How it was trained and how it scores is in
`docs/research/MODEL-COMPARISON.md`; building the GGUF is in `benchmarks/modal/README.md`.
The two model-driven demos accept any OpenAI-compatible endpoint instead.

```sh
npm run build:core
examples/team-rules/run.sh
WRITER_GGUF=/path/to/r23-gemma4-e2b-Q8_0.gguf examples/personal-timeline/run.sh
WRITER_GGUF=/path/to/r23-gemma4-e2b-Q8_0.gguf examples/approvals-desk/run.sh
```

Every run uses a fresh temporary memory root and touches nothing else. `examples/lib/show.mjs`
turns the CLI's JSON into the compact transcript you see; the raw JSON is one flag away.
