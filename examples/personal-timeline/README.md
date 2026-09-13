# Personal timeline: weeks of ordinary statements, then the real questions

Seven things a person might tell an assistant over a few weeks: a dentist visit, marathon
training, a gym price rise, a promise to a colleague, a book, an office move, a birthday.
Then seven questions, including two the memory must not pretend to answer.

Every model call is Remembero's own writer, a 2.3B Gemma 4 E2B fine-tune served locally by
llama.cpp. It turns each statement into facts and each question into a query. No model
phrases the answer: code assembles it from the facts, their source sentences, and
**computed notes**, which resolve every date against when it was said, state the distance
to today, and total quantities with their units.

```sh
brew install llama.cpp                    # once
npm run build:core                        # once
WRITER_GGUF=/path/to/r23-gemma4-e2b-Q8_0.gguf examples/personal-timeline/run.sh
```

Building the writer GGUF from a training run is in `benchmarks/modal/README.md`; a writer
already serving on port 8081 is picked up without the variable. Any OpenAI-compatible
endpoint works with `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`.

## What you will see

Nine `remember` calls, each printing the facts the writer extracted, then seven questions
answered in evidence mode: the rows, the claims behind them, the sentence each came from,
and a Computed section.

- **"When was my last dental check-up?"** The date, and under it the distance from
  3 August to today in weeks and days. The arithmetic is done by code, not the model.
- **"What long runs have I done so far?"** Three rows, and the notes resolve the three
  dates against when they were said and state the gaps between them.
- **"What is my monthly gym cost?"** Both figures the sentence stated come back, with the
  difference between them computed; the writer stored both, it did not supersede.
- **"When is the Atlas status update due?"** The deadline, and "5 days after the question
  date" beneath it.
- **"When did our office move to Collins Street?"** The move date.
- **"What could I get Sam for his birthday?"** A preference, returned as evidence rather
  than advice, with his birthday resolved to a date three weeks out.
- **"Who is my accountant?"** No fact supports an answer, so the status is `unanswerable`
  and nothing is invented.

## What the run also shows

The writer is a small model and the transcript keeps its rough edges on purpose. "I
promised Maya a status update on the Atlas project" became `deadline(atlas, ...)` and lost
who the promise was made to, so the question is asked about the deadline. Compound
statements extract more shallowly than short ones, which is why the statements are one or
two facts each, the way people talk to an assistant in short turns. Questions worded
another way can produce a query the engine refuses (an aggregate over non-numeric values,
or a wildcard where a variable was needed); the refusal is printed as such and nothing is
guessed. The captured output of one run is in `transcript.md`.
