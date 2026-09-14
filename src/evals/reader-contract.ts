/**
 * The reader contract: the deterministic structure a reader is handed before it reads.
 * The distiller renders training prompts with it, the evaluation runner takes the same
 * flags, and the training manifest records its id, so a trained reader is always
 * evaluated and served with exactly the prompt it learned from.
 */
export interface ReaderContract {
  /** Session headers state the distance to the question date. */
  dateDistances: boolean;
  /** The computed-notes block (dates resolved, gaps, quantities), every line quoting its sentence. */
  computedNotes: boolean;
  /** Context share per session weighted by the question's words it contains. */
  focusedBudget: boolean;
  /** The writer's own facts, dated, grounded and deduplicated, before the chats. */
  structuredEvidence: boolean;
  contextBytes: number;
}

export const DEFAULT_READER_CONTEXT_BYTES = 24 * 1024;

/** What reader v5 was distilled with: the two blocks with a measured gain. */
export const READER_CONTRACT_V5: ReaderContract = {
  dateDistances: true,
  computedNotes: true,
  focusedBudget: false,
  structuredEvidence: false,
  contextBytes: DEFAULT_READER_CONTEXT_BYTES,
};

const FLAGS: Array<
  [keyof Omit<ReaderContract, 'contextBytes'>, string, string]
> = [
  ['dateDistances', '--date-distances', 'dd'],
  ['computedNotes', '--computed-notes', 'notes'],
  ['focusedBudget', '--focused-budget', 'focus'],
  ['structuredEvidence', '--structured-evidence', 'evidence'],
];

export function contractId(contract: ReaderContract): string {
  const parts = FLAGS.filter(([key]) => contract[key]).map(
    ([, , short]) => short,
  );
  return `${parts.length === 0 ? 'plain' : parts.join('+')}@${contract.contextBytes}`;
}

export function contractFromFlags(
  argv: readonly string[],
  base: ReaderContract = READER_CONTRACT_V5,
): ReaderContract {
  const contract = {
    ...base,
    dateDistances: false,
    computedNotes: false,
    focusedBudget: false,
    structuredEvidence: false,
  };
  for (const [key, flag] of FLAGS)
    if (argv.includes(flag)) contract[key] = true;
  const bytes = argv.indexOf('--context-bytes');
  if (bytes >= 0 && argv[bytes + 1] !== undefined)
    contract.contextBytes = Number(argv[bytes + 1]);
  return contract;
}

/** The environment the distiller has honoured so far, kept so old commands still work. */
export function contractFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): ReaderContract {
  return {
    ...READER_CONTRACT_V5,
    computedNotes: env.REMEMBERO_READER_COMPUTED_NOTES === '1',
    focusedBudget: env.REMEMBERO_READER_FOCUSED_BUDGET === '1',
    structuredEvidence: env.REMEMBERO_READER_STRUCTURED_EVIDENCE === '1',
  };
}

/**
 * The flags that make run-longmemeval-answer build this exact prompt. The byte budget is
 * always stated: the runner's own default (57344) is not the distiller's (24576), so a
 * contract that left the flag off would be evaluated at a context size it never read at.
 */
export function contractRunnerFlags(contract: ReaderContract): string[] {
  const flags = FLAGS.filter(([key]) => contract[key]).map(([, flag]) => flag);
  flags.push('--context-bytes', String(contract.contextBytes));
  return flags;
}

/** The trailing positional arguments of buildLongMemEvalAnswerContext, in its order. */
export function contractBuilderArgs(
  contract: ReaderContract,
): [boolean, boolean, boolean, boolean] {
  return [
    contract.dateDistances,
    contract.computedNotes,
    contract.focusedBudget,
    contract.structuredEvidence,
  ];
}
