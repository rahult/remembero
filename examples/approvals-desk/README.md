# Approvals desk: "may X approve Y on this date", with the quotes

The finance inbox of a community foundation over one year. The delegations schedule is
ingested in March. Then the requests arrive: inside a limit, over it, during a suspension,
under a Slack delegation from the CEO, from a person nobody registered, from an initial
that could be two people, with a conflict of interest. In November an amendment raises one
limit, lowers another and lifts the suspension, and the same questions change answer.

Every decision is ALLOW, DENY or UNKNOWN with the document, page and paragraph it rests on.
UNKNOWN is a real answer: the documents do not settle it, and the report says what would.
The number that must stay at zero is unjustified ALLOW.

The only model in the loop is Remembero's own writer, a 2.3B fine-tune that extracts the
claims at ingest time. Everything after the trust boundary is code: schema validation,
grounding of every amount and date in its own quote, entity resolution, then the decision.

```sh
brew install llama.cpp                    # once
WRITER_GGUF=/path/to/r23-gemma4-e2b-Q8_0.gguf examples/approvals-desk/run.sh
```

The script creates `remembro-eval/.venv` the first time and starts the writer on demand.
Add `--keep` to keep the workspace and open it with the MCP server (`remembro-eval/README.md`).

## What you will see

Seventeen steps, each `decide` with what it expected and what it got, the last reasons and
the quotes. The interesting ones:

- **The Slack delegation.** Nadia's operational limit is 10,000. A message from the CEO
  lets her approve 15,000 on 1 September and not on 5 October, because the message said
  "until 30 September".
- **The suspension.** Owen's 55,000 is fine on 1 July and refused on 20 August. After the
  amendment lifts the suspension and raises his limit, 70,000 on 15 November is fine.
- **The initial.** "J. Ford" could be Julian or Julia. The answer is UNKNOWN with a
  `next_steps` entry naming the alias a human could record to settle it.
- **The stranger.** Sam Oduya is not in any document. DENY, not a guess.
- **The conflict.** The same request that was ALLOW becomes DENY when the approver benefits.

The script is `approvals-desk.yaml`; the policy and amendment are the fixtures in
`remembro-eval/fixtures/`. Write your own inbox the same way and put private documents under
`remembro-eval/exercises/private/`, which is never committed.
