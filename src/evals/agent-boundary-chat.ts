/**
 * Chat backends for the agent-boundary runner. `ollama` is the published
 * default. `openai` speaks the OpenAI-compatible /v1/chat/completions shape so
 * a Tinker fine-tune served through `tinker_cookbook.capture.proxy.serve`
 * (or any vLLM/LM Studio style server) is evaluated by the unchanged harness.
 */
export type ChatBackend = 'ollama' | 'openai';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export const CHAT_MAX_TOKENS = 400;

export function chatRequest(
  backend: ChatBackend,
  baseUrl: string,
  model: string,
  messages: ChatMessage[],
  seed: number,
): { url: string; body: Record<string, unknown> } {
  const base = baseUrl.replace(/\/$/, '');
  if (backend === 'openai') {
    return {
      url: `${base}/v1/chat/completions`,
      body: {
        model,
        messages,
        temperature: 0,
        seed,
        max_tokens: CHAT_MAX_TOKENS,
        stream: false,
      },
    };
  }
  return {
    url: `${base}/api/chat`,
    body: {
      model,
      messages,
      stream: false,
      options: {
        temperature: 0,
        seed,
        num_ctx: 4096,
        num_predict: CHAT_MAX_TOKENS,
      },
    },
  };
}

export function parseChatResponse(
  backend: ChatBackend,
  payload: unknown,
): string {
  const record = payload as Record<string, unknown> | null;
  let content: unknown;
  if (backend === 'openai') {
    const choices = record?.choices;
    const first = Array.isArray(choices)
      ? (choices[0] as Record<string, unknown>)
      : undefined;
    content = (first?.message as Record<string, unknown> | undefined)?.content;
  } else {
    content = (record?.message as Record<string, unknown> | undefined)?.content;
  }
  if (typeof content !== 'string') {
    throw new Error(`${backend} returned no message content`);
  }
  return content;
}

/** Resolve the backend from --chat-api / CHAT_API; anything but 'openai' is Ollama. */
export function resolveChatBackend(
  argv: readonly string[],
  env: NodeJS.ProcessEnv,
): ChatBackend {
  const flag = argv.indexOf('--chat-api');
  const value = flag >= 0 ? argv[flag + 1] : env.CHAT_API;
  return value === 'openai' ? 'openai' : 'ollama';
}

export interface AnswerLeg {
  model: string;
  backend: ChatBackend;
  /** Base URL for the answer leg; undefined means "same as the query leg". */
  url: string | undefined;
}

/**
 * The answer step may run on a different model than the query step
 * (--answer-model, --answer-chat-api, ANSWER_URL). This isolates query
 * authoring when the query model is a fine-tuned base model that was never
 * trained to read rows back. Defaults to the query model and backend.
 */
export function resolveAnswerLeg(
  argv: readonly string[],
  env: NodeJS.ProcessEnv,
  queryModel: string,
  queryBackend: ChatBackend,
): AnswerLeg {
  const modelFlag = argv.indexOf('--answer-model');
  const model = modelFlag >= 0 ? argv[modelFlag + 1] : queryModel;
  const apiFlag = argv.indexOf('--answer-chat-api');
  const apiValue = apiFlag >= 0 ? argv[apiFlag + 1] : undefined;
  const backend: ChatBackend =
    apiValue === 'openai'
      ? 'openai'
      : apiValue === 'ollama'
        ? 'ollama'
        : queryBackend;
  const url =
    env.ANSWER_URL ??
    (modelFlag >= 0 && backend !== queryBackend
      ? backend === 'openai'
        ? 'http://127.0.0.1:7462'
        : 'http://127.0.0.1:11434'
      : undefined);
  return { model, backend, url };
}
