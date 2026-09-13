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

- **"How long ago was my last dentist visit?"** The answer states the number of days from
  3 August to today and the due date six months on, each line quoting the sentence it came
  from. The arithmetic is done by code, not the model.
- **"How many kilometres of long runs?"** 12 + 16 + 21, summed by the notes, with the
  three dated runs listed.
- **"What does my gym cost now?"** The later value wins; the earlier one is kept as history.
- **"What have I promised Maya?"** A commitment with its deadline.
- **"What could I get Sam?"** A preference, returned as evidence rather than advice.
- **"Who is my accountant?"** No fact supports an answer, so the status is `no_match` and
  the nearest facts are listed under `related`. The memory does not guess.
