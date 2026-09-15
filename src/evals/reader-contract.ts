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
  /**
   * Context tiers: this many top-ranked sessions get their full text and every other
   * retrieved session a short code-built abstract. Null is the even split.
   */
  fullSessions: number | null;
  /** The byte cap of one abstract section, header included (tiers only). */
  abstractBytes: number;
}

export const DEFAULT_READER_CONTEXT_BYTES = 24 * 1024;
export const DEFAULT_ABSTRACT_BYTES = 320;
export const MIN_ABSTRACT_BYTES = 120;
export const MAX_ABSTRACT_BYTES = 2048;

/** The trailing tiers argument of buildLongMemEvalAnswerContext. */
export interface ContextTiers {
  fullSessions: number;
  abstractBytes: number;
}

/** What reader v5 was distilled with: the two blocks with a measured gain. */
export const READER_CONTRACT_V5: ReaderContract = {
  dateDistances: true,
  computedNotes: true,
  focusedBudget: false,
  structuredEvidence: false,
  contextBytes: DEFAULT_READER_CONTEXT_BYTES,
  fullSessions: null,
  abstractBytes: DEFAULT_ABSTRACT_BYTES,
};

const FLAGS: Array<
  [
    keyof Omit<ReaderContract, 'contextBytes' | 'fullSessions' | 'abstractBytes'>,
    string,
    string,
  ]
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
  const tiers =
    contract.fullSessions === null
      ? ''
      : `+full${contract.fullSessions}${contract.abstractBytes === DEFAULT_ABSTRACT_BYTES ? '' : `a${contract.abstractBytes}`}`;
  return `${parts.length === 0 ? 'plain' : parts.join('+')}${tiers}@${contract.contextBytes}`;
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
  if (bytes >= 0 && argv[bytes + 1] !== undefined) {
    const value = argv[bytes + 1]!;
    const parsed = Number(value);
    // A typo here would otherwise reach the contract id and the runner flags as NaN,
    // labelling a run with a byte budget nobody ever read at.
    if (!Number.isInteger(parsed) || parsed <= 0)
      throw new Error(`--context-bytes must be a positive integer, got ${value}`);
    contract.contextBytes = parsed;
  }
  const tiers = tiersFromFlags(argv, contract.abstractBytes);
  contract.fullSessions = tiers.fullSessions;
  contract.abstractBytes = tiers.abstractBytes;
  return contract;
}

function flagValue(argv: readonly string[], flag: string): string | undefined {
  const index = argv.indexOf(flag);
  return index >= 0 ? argv[index + 1] : undefined;
}

/**
 * `--full-sessions <n>` and `--abstract-bytes <n>`, validated, as the contract and the
 * evaluation runner both read them. Tiers and the focused budget are two budget policies;
 * an arm with both would not pair against either.
 */
export function tiersFromFlags(
  argv: readonly string[],
  baseAbstractBytes: number = DEFAULT_ABSTRACT_BYTES,
): { fullSessions: number | null; abstractBytes: number } {
  const full = flagValue(argv, '--full-sessions');
  const abstract = flagValue(argv, '--abstract-bytes');
  let fullSessions: number | null = null;
  let abstractBytes = baseAbstractBytes;
  if (full !== undefined) {
    const parsed = Number(full);
    if (!Number.isInteger(parsed) || parsed <= 0)
      throw new Error(`--full-sessions must be a positive integer, got ${full}`);
    fullSessions = parsed;
  }
  if (abstract !== undefined) {
    if (fullSessions === null)
      throw new Error('--abstract-bytes needs --full-sessions');
    const parsed = Number(abstract);
    if (
      !Number.isInteger(parsed) ||
      parsed < MIN_ABSTRACT_BYTES ||
      parsed > MAX_ABSTRACT_BYTES
    )
      throw new Error(
        `--abstract-bytes must be an integer from ${MIN_ABSTRACT_BYTES} to ${MAX_ABSTRACT_BYTES}, got ${abstract}`,
      );
    abstractBytes = parsed;
  }
  if (fullSessions !== null && argv.includes('--focused-budget'))
    throw new Error('--full-sessions cannot be combined with --focused-budget');
  return { fullSessions, abstractBytes };
}

/** The environment the distiller has honoured so far, kept so old commands still work. */
export function contractFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): ReaderContract {
  return {
    ...READER_CONTRACT_V5,
    computedNotes: env.REMEMBERO_READER_COMPUTED_NOTES === '1',
  };
}

/**
 * The flags that make run-longmemeval-answer build this exact prompt. The byte budget is
 * always stated: the runner's own default (57344) is not the distiller's (24576), so a
 * contract that left the flag off would be evaluated at a context size it never read at.
 */
export function contractRunnerFlags(contract: ReaderContract): string[] {
  const flags = FLAGS.filter(([key]) => contract[key]).map(([, flag]) => flag);
  if (contract.fullSessions !== null)
    flags.push(
      '--full-sessions',
      String(contract.fullSessions),
      '--abstract-bytes',
      String(contract.abstractBytes),
    );
  flags.push('--context-bytes', String(contract.contextBytes));
  return flags;
}

/** The trailing positional arguments of buildLongMemEvalAnswerContext, in its order. */
export function contractBuilderArgs(
  contract: ReaderContract,
): [boolean, boolean, boolean, boolean, ContextTiers | undefined] {
  return [
    contract.dateDistances,
    contract.computedNotes,
    contract.focusedBudget,
    contract.structuredEvidence,
    contract.fullSessions === null
      ? undefined
      : {
          fullSessions: contract.fullSessions,
          abstractBytes: contract.abstractBytes,
        },
  ];
}

/**
 * What the distill command writes into its manifest: the contract it rendered every
 * training prompt with, its id, and the flags that reproduce it in the evaluation runner.
 */
export function distillManifestContract(
  argv: readonly string[],
): ReaderContract & { id: string; runnerFlags: string[] } {
  const contract = contractFromFlags(argv);
  return {
    ...contract,
    id: contractId(contract),
    runnerFlags: contractRunnerFlags(contract),
  };
}
