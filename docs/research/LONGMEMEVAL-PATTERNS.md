# What measurably raises LongMemEval accuracy

Status: literature survey, primary sources only, 2026-09-11.
Question: which design patterns have reported gains on LongMemEval-S (Wu et al., ICLR
2025; 500 questions, 30 of them abstention), and which apply to Remembero's current
policy (whole-session BM25-like retrieval plus Datalog facts, top-4/top-5, 56 KiB reader
context, gpt-5.6-luna reader, gpt-4o judge, 82-84% dev, multi-session 50-57/69,
temporal 52/66; see [LONGMEMEVAL.md](LONGMEMEVAL.md)).

Caveat: the dataset was re-cleaned in September 2025 ([README](https://github.com/xiaowu0162/LongMemEval));
numbers below mix pre/post-cleaning runs, readers, and judges. Only the paper's tables
hold everything else fixed.

## 1. The paper's ablations

Source: [arXiv 2410.10813](https://arxiv.org/abs/2410.10813) (v1 14 Oct 2024, v2 4 Mar
2025); HTML tables at [arxiv.org/html/2410.10813v2](https://arxiv.org/html/2410.10813v2).
Setup (Sec. 5.1): Stella V5 1.5B dense retriever, Llama 3.1 8B extraction, user-side
utterances only as keys, items sorted by timestamp, CoN + JSON reading by default. The
indexing tables are on **LongMemEval-M** (500 sessions), not S.

| Pattern | Evidence (exact figures) | Where |
| --- | --- | --- |
| Value granularity: round vs session | Rounds "significantly" beat sessions for a GPT-4o reader, tie for Llama 8B; summaries/facts as *values* hurt overall but "fact decomposition consistently improves" multi-session; GPT-4o "continues to improve even with over 20k retrieved tokens", Llama 8B collapses past 3k. Figure only; no per-point numbers. | Fig. 5, Sec. 5.2 |
| Key expansion K = V + fact (facts prepended to the key) | Session, Stella: R@5 0.706 -> 0.732, R@10 0.783 -> 0.862; GPT-4o QA top-5 0.670 -> 0.714. Round: R@5 0.582 -> 0.644, R@10 0.692 -> 0.784; GPT-4o QA top-5 0.615 -> 0.657, top-10 0.670 -> 0.720. Average "+9.4% recall@k and +5.4% final accuracy". Facts *alone* as key are worse than the value (0.642 vs 0.706). | Table 3 |
| Same, with BM25 | Session: R@5 0.634 -> 0.683, R@10 0.710 -> 0.757. Round: R@5 0.472 -> 0.554. Summary/keyphrase expansion gives BM25 nothing (0.626 / 0.632). BM25 trails dense by ~9 R@5 points at session level (0.634 vs 0.720-0.723). | Table 9 |
| Merge into key, not a parallel index | "Rank merging has much lower performance than key merging"; index grows m+1 times. | Sec. E.3, Table 10 |
| Time-aware query expansion (temporal subset) | LLM extracts a date range; candidates filtered by timestamped events. Session K=V: R@5 0.639 -> 0.654 (GPT-4o extractor); K=V+fact: 0.684 -> 0.722, R@10 0.721 -> 0.797. Round K=V+fact: R@5 0.489 -> 0.526, R@10 0.550 -> 0.722. Summary: +6.8% (session) / +11.3% (round) recall. Llama 8B as extractor *hurts* (0.639 -> 0.624): it invents ranges when there is no time cue. | Table 4, Sec. E.4 |
| Reading: Chain-of-Note + JSON | Under oracle retrieval, worst-to-best prompt is "up to a 10-point absolute" gap for GPT-4o; JSON helps reliably only *with* CoN. Full-context S: GPT-4o 0.606 -> 0.640 with CoN (oracle 0.870 / 0.924). | Fig. 6, Fig. 3b |
| Error budget, best design, top-10 | 15-19% of questions are retrieved-right-but-answered-wrong (40-50% of errors). | Sec. E.5 |

## 2. Systems reporting LongMemEval-S numbers

| System (date) | Overall | Reader / judge | Credited mechanism | Source |
| --- | --- | --- | --- | --- |
| Zep / Graphiti (20 Jan 2025) | 71.2% (gpt-4o), 63.8% (gpt-4o-mini); full-context 60.2% / 55.4% | gpt-4o-2024-11-20 / gpt-4o-mini; gpt-4o judge | Bi-temporal graph; cosine + BM25 + BFS; RRF/MMR/cross-encoder rerank; **20 edges + nodes, ~1.6k tokens** | [arXiv 2501.13956](https://arxiv.org/abs/2501.13956) |
| Zep (6 Nov 2025) | "above 80%", "up 10% since our early 2025 paper" | not stated | halved LLM tokens; classical NLP replaces LLM steps | [blog](https://blog.getzep.com/scaling-agent-memory-zep-30x/) |
| Emergence "Simple" (18 Jun 2025) | 82.4%; Simple Fast 79.0%; Internal 86.0% (not public) | gpt-4o-2024-08-06 for both | **Match on turns, return whole sessions**, session score = NDCG of its cross-encoder-reranked turns; simple CoT. Fast: 42 turns, two-call extract-then-answer; 20 turns scored 76.8% | [blog](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag) |
| Supermemory (Nov 2025) | 81.6% (gpt-4o), 84.6% (gpt-5), 85.2% (gemini-3-pro) | listed; gpt-4o judge | chunk-based semantic search; mechanism not published | as transcribed in [Hindsight Table 3](https://arxiv.org/html/2512.12818) and [Mastra](https://mastra.ai/research/observational-memory); vendor page gives no figures |
| Hindsight (14 Dec 2025) | 83.6% (gpt-oss-20b), 89.0% (oss-120b), 91.4% (gemini-3-pro) | listed backbones; gpt-4o-style judge | Fact graph with temporal ranges + entity links; vector + BM25 + graph + temporal filter fused by RRF, cross-encoder rerank, token budget | [arXiv 2512.12818](https://arxiv.org/abs/2512.12818) |
| Mastra Observational Memory (9 Feb 2026) | 84.23% (gpt-4o), 94.87% (gpt-5-mini) | listed; gpt-4o judge; gemini-2.5-flash writes observations | **Retrieval-free**: dated observations (observation/referenced/relative date) kept in context, ~30k tokens avg, ~6x compression | [research page](https://mastra.ai/research/observational-memory) |
| Mem0 platform (16 Apr / 14 May 2026) | 94.4% at top-200; **94.8% at top-50** | harness defaults gpt-4o answerer and gpt-4o judge; ~6.8k tokens/query | ADD-only fact extraction; semantic + BM25 + entity + temporal-metadata score fusion; recency boost 1.5x / decay 0.3x | [blog](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm), [temporal update](https://mem0.ai/blog/the-token-efficient-memory-algorithm-now-has-temporal-reasoning), [harness README](https://github.com/mem0ai/memory-benchmarks) |
| ByteRover (31 Mar 2026) | 92.8% | Gemini 3.1 Pro reader; Gemini judge (not gpt-4o) | hierarchical "context tree" by domain/topic/session | [blog](https://www.byterover.dev/blog/benchmark_ai_agent_memory_real_production_byterover_top_market_accuracy_longmemeval) |
| MemReader survey (9 Apr 2026) | Zep 63.8%, Mem0 66.4%, MemOS 77.8%, EverMemOS 83.0% under one stack | GPT-4.1-mini reader and judge | fact-only stores collapse on SS-assistant (Mem0 26.8%) | [arXiv 2604.07877](https://arxiv.org/html/2604.07877) |
| Letta | no LongMemEval number found (LoCoMo only) | - | - | [blog](https://www.letta.com/blog/benchmarking-ai-agent-memory/) |

There is no official leaderboard; vendor comparison tables mix judges and readers.

## 3. Multi-session and temporal evidence

Per-type counts (Mem0/Mastra breakdowns): multi-session 133, temporal 133,
knowledge-update 78, SS-user 70, SS-assistant 56, SS-preference 30.

| Mechanism | Multi-session | Temporal | Source |
| --- | --- | --- | --- |
| Facts as values (uniform format) | "consistently improves" multi-session while hurting other types | - | Paper Fig. 5 |
| Time-range filter (strong extractor) | - | +6.8% / +11.3% recall; R@10 0.550 -> 0.722 (round, V+fact) | Paper Table 4 |
| Temporal KG vs full context (gpt-4o) | 44.3 -> 57.9 | 45.1 -> 62.4 | Zep Table 3 |
| Turn match, session return, rerank, CoT (gpt-4o) | Naive RAG 36.8 -> Fast 70.7 -> Simple 73.7 -> Internal 81.2 | 51.9 -> 76.7 -> 81.2 -> 85.7 | Emergence table |
| Graph + temporal retrieval vs same 20B full-context | 21.1 -> 79.7 | 31.6 -> 79.7 | Hindsight Table 3 |
| Write-time temporal metadata scored at read time | 86.5 -> 88.0 | 93.2 -> 97.0 | Mem0 May 2026 |
| Smaller k (200 -> 50) | 88.0 -> 93.2 | 97.0 -> 94.0 | Mem0 harness README |
| Dated observations in context (gpt-4o) | 79.7 | 85.7 | Mastra |

Multi-session is the floor everywhere: 81.2 (Emergence), 87.2 (Hindsight gemini-3,
Mastra gpt-5-mini), 88.0-93.2 (Mem0). Mastra calls ~87% a ceiling "regardless of approach".

## 4. Retrieval unit and top-k

| System | Unit indexed | Unit returned | k / budget |
| --- | --- | --- | --- |
| Paper best design | round (user turns), fact-expanded key | round | top-5/top-10; GPT-4o still gains past 20k tokens |
| Emergence Simple | turn | whole session (NDCG-aggregated) | not stated; Fast: 42 turns, 20 better |
| Zep | graph edges (facts) + entities | context string | 20 items, ~1.6k tokens |
| Mem0 | extracted facts | facts | top-50 beats top-200 overall; ~6.8k tokens |
| Hindsight | narrative facts | facts | token budget (value not given in text) |
| Mastra | none | all observations | ~30k tokens avg |

Turn-level *matching* beats session-level for capable readers (Fig. 5, Emergence), but
turn-level *recall* at fixed k is lower (BM25 R@5 0.472 vs 0.634) because each slot holds
less; the winning move is match on turns, return sessions.

## 5. Ranked patterns for Remembero

Remembero already has user-turn-only context (v2), a gated semantic rerank for
multi-session (v5), and the paper's k range. What remains, ranked by evidence strength
times fit with a lexical whole-session index:

1. **Fact-expanded keys (K = V + fact) merged into the BM25 document.** Prepend the
   rendered Datalog facts of each session to its indexed text. The one pattern with a
   BM25-specific number (+4.9 R@5, +4.7 R@10, Table 9) and the paper's largest end-to-end
   gain (+5.4). Do not rank-merge a parallel fact index (Table 10).
2. **Turn-level matching, session return, NDCG aggregation.** Score user turns,
   aggregate to sessions by NDCG of their turns (Emergence: 52% naive -> 82.4%), keep
   returning whole sessions so context stays ~56 KiB. Emergence's multi-session went
   36.8 -> 73.7.
3. **Time-range filtering for temporal questions**, a frontier model extracting the range
   and refusing when there is no cue. +6.8-11.3% recall (Table 4), +3.8 temporal points at
   Mem0. Temporal is the largest fixed gap (52/66) and the facts already carry dates.
4. **Extract-then-answer reading with JSON items.** Up to 10 points for GPT-4o under
   oracle retrieval (Fig. 6); 40-50% of residual errors are reader errors (E.5). The
   paper's Fig. 13 prompt is a drop-in to test on gpt-5.6-luna.
5. **A fact-shaped bundle for multi-session only.** Fact decomposition helps only
   multi-session (Fig. 5); GPT-4o improves past 20k tokens; Mem0's top-50 beat top-200 on
   multi-session by 5.2 points. Remembero's facts-in-context test matched raw without
   beating it, so this is the weakest bet; try it after 1-3 raise recall.
