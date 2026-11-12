# ModelRouter: research notes

Researched 2026-11-12. Facts marked (checked) were verified on this machine today (HF API, file downloads); the rest come from the papers and docs linked.

## 1. The field in one page

| Work | Idea | What I take from it |
|---|---|---|
| FrugalGPT, Chen et al. 2023 ([arXiv 2305.05176](https://arxiv.org/abs/2305.05176)) | Cascade: try the cheap model, a learned scorer decides whether to accept or escalate; plus prompt adaptation and caching | The escalation and caching mechanisms in the gateway; cascade as a second policy next to pure prediction |
| Hybrid LLM, Ding et al. 2024, ICLR ([arXiv 2404.14618](https://arxiv.org/abs/2404.14618)) | Train a router to predict the quality gap between a small and a large model; one threshold tunable at test time trades quality for cost; reports up to 40% fewer large-model calls with no quality drop (on their setup) | The framing: predict "will the small model be good enough", expose a test-time threshold |
| AutoMix, Madaan et al. 2023 ([arXiv 2310.12963](https://arxiv.org/abs/2310.12963)) | Small model self-verifies its own answer; a POMDP-style router escalates when confidence is low | Informs the low-confidence escalation path (post-hoc signal after the small model answers) |
| RouterBench, Hu et al. 2024 ([arXiv 2403.12031](https://arxiv.org/abs/2403.12031), [code](https://github.com/withmartian/routerbench), MIT) | 405k+ recorded outcomes of 11 LLMs; defines the cost-quality plane, the non-decreasing convex hull, and AIQ (area under it) | Dataset, and the evaluation protocol: because every model's answer is already recorded, a router can be scored offline with no API calls |
| RouteLLM, Ong et al. 2024 ([arXiv 2406.18665](https://arxiv.org/abs/2406.18665), [code](https://github.com/lm-sys/RouteLLM), Apache-2.0) | Routers (similarity-weighted, matrix factorisation, BERT, causal LLM) trained on Chatbot Arena preferences; metrics CPT(x%) = fewest strong-model calls needed to reach x% of the strong model's quality, and APGR = average performance gap recovered across call fractions; calibrates a threshold to a target call fraction | Metrics (CPT, APGR) and threshold calibration by quantile of predicted scores; OpenAI-compatible server design |
| LiteLLM Router ([docs](https://docs.litellm.ai/docs/routing)) | Self-hosted gateway: load balancing, fallbacks, cost/latency strategies across deployments; auto/complexity routers exist as sub-routers | Fallback semantics and a pricing map as config. It routes among deployments, it does not learn per-prompt difficulty |
| OpenRouter Auto Router ([docs](https://openrouter.ai/docs/guides/features/routers/auto-router)) | Hosted; picks a model per prompt, originally powered by Not Diamond; docs say the original is being replaced by a newer variant | Confirms the product shape: one endpoint, `model: auto`, answer reports which model ran |
| Not Diamond ([site](https://www.notdiamond.ai)) | Commercial learned router, claims cost and accuracy gains | Vendor claims, not reproduced here; I make no comparison with it |

Takeaways that shape the design:
1. Offline evaluation on recorded outcomes is the standard, honest way to measure a router (RouterBench). Cost-quality curves and AIQ/APGR/CPT are the accepted summaries.
2. Router must beat a random mix at the same call fraction, otherwise it is only trading linearly. Random routing is the real baseline, since interpolating between always-small and always-big is a straight line.
3. Prediction-before (Hybrid LLM / RouteLLM) and verification-after (FrugalGPT / AutoMix) are complementary. I implement predict-first plus a post-hoc escalation check.
4. Published gains depend heavily on the model pair. A weak/strong pair with a wide quality gap routes well; a narrow gap leaves little to gain. I will report this rather than hide it.

## 2. Datasets (licences and access, checked)

| Dataset | Content | Access | Licence | Verdict |
|---|---|---|---|---|
| [withmartian/routerbench](https://huggingface.co/datasets/withmartian/routerbench) | 0-shot file: 36,497 prompts x 11 LLMs; per-model correctness score (0..1), per-model cost, response text; 86 sub-benchmarks (hellaswag 10,042, GSM8K 7,450, MMLU, ARC, Winogrande, MBPP, MT-Bench, ...). 5-shot (171 MB) and raw (1.2 GB) also exist | Open, not gated. `routerbench_0shot.pkl` = 99.6 MB | HF card declares no licence. Code repo is MIT. Prompts come from public benchmarks, each with its own licence (MMLU MIT, GSM8K MIT, Hellaswag MIT, Winogrande Apache-2.0, ARC CC-BY-SA, MBPP CC-BY-4.0, MT-Bench CC-BY-4.0 per their cards, to re-verify in build) | **Chosen.** Do not commit it; download by script; README states the licence situation plainly |
| [routellm/gpt4_dataset](https://huggingface.co/datasets/routellm/gpt4_dataset) | 290 MB jsonl; prompts from lmsys-chat-1m with GPT-4 and Mixtral responses and judge scores | Open, not gated, Apache-2.0 tag | Tagged Apache-2.0, but prompts derive from LMSYS-Chat-1M, whose own terms are stricter and not visible from this card | Rejected as primary: provenance risk, 290 MB, labels are judge scores not ground truth |
| [routellm/gpt4_judge_battles](https://huggingface.co/datasets/routellm/gpt4_judge_battles) | 167 MB parquet of battle data | Open, Apache-2.0 tag | Same provenance concern | Rejected |
| LMSYS Arena preference data | Human votes | Gated / separate terms | Restrictive | Not used |

Why RouterBench: ground-truth correctness (benchmarks with known answers) rather than judge opinions, both a weak and a strong model recorded per prompt, real per-call costs, a diverse task mix so difficulty is learnable, and a permissively licensed code base with an established protocol.

**Security note.** The files are pickled pandas DataFrames. Unpickling untrusted data can execute code. I loaded the file here with an allow-list unpickler (only pandas/numpy reconstruction classes) and it loads fine: shape (36497, 37), columns `sample_id, prompt, eval_name, <model>, <model>|model_response, <model>|total_cost, oracle_model_to_route_to`. The build will do this once in `modelrouter data fetch`, pin the SHA-256 of the file, and store a slim Parquet (prompt, task, per-model score and cost; drop responses) so nothing else ever calls `pickle`. The slim Parquet stays gitignored and is rebuilt by script; only the recorded evaluation results (scores per threshold, a few thousand test prompts' predicted difficulty) go into `docs/`.

Measured on the file (mean score, mean recorded cost per request in USD, prices are Nov 2023):

| Model | Score | Cost / req |
|---|---|---|
| gpt-4-1106-preview | 0.781 | 0.00329 |
| Yi-34B-Chat | 0.647 | 0.00019 |
| (a third-party hosted model) | 0.636 | 0.00242 |
| gpt-3.5-turbo-1106 | 0.619 | 0.00024 |
| mixtral-8x7b-chat | 0.547 | 0.00014 |
| mistral-7b-chat | 0.306 | 0.00005 |

Chosen pair for the headline curve: **gpt-3.5-turbo-1106 (small) vs gpt-4-1106-preview (big)**: a 0.16 absolute quality gap and 13.7x cost gap, which is the pair the "small/big" story is about. Second pair, included for honesty about gap size: mixtral-8x7b vs gpt-4. Prompt length mean 778 chars, so the classifier sees realistic requests. Caveat to report: the recorded costs use late-2023 prices and models; the gateway's pricing config uses current prices separately, and the offline curve is labelled with its price date.

## 3. Models for the live local demo (checked)

| Model | Role | Size | Licence | Gated |
|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | local small tier | 494M params | Apache-2.0 | no |
| Qwen/Qwen2.5-1.5B-Instruct | local big tier | 1.54B params | Apache-2.0 | no |
| sentence-transformers/all-MiniLM-L6-v2 (already in HF cache) | prompt embeddings | 22.7M | Apache-2.0 | no |
| BAAI/bge-small-en-v1.5 (alternative) | embeddings | 33M | MIT | no |

Qwen2.5-3B is excluded (research licence). 0.5B and 1.5B are the same family so the quality gap is real but modest; CPU generation (6 torch threads, 12 cores) is the practical limit, so the live experiment uses short outputs (multiple-choice letter or a short number) and a few hundred prompts. A Qwen3-0.6B copy exists in the cache and can be a fallback.

Live-experiment ground truth: RouterBench has no gold answers, so the local experiment uses items with known answers from permissive sets (GSM8K, MIT; CommonsenseQA from `tau/commonsense_qa`, already cached, MIT; ARC-Easy/Challenge CC-BY-SA). Router for the local pair is retrained on the local pair's own small/big correctness labels with cross-validation, and I also report how the RouterBench-trained router transfers (expected: weaker, since it learned the GPT-3.5/GPT-4 pair).

## 4. Methods for the classifier

- Target: binary `small_ok` (small model scored >= 0.5 / correct), plus the regression variant (Hybrid LLM style, quality gap). Router output = P(small is good enough) -> difficulty = 1 - P.
- Features: MiniLM sentence embeddings (384-d) + cheap lexical features (char/word length, digit ratio, code-fence and math symbol presence, question count, line count, option-list pattern, max word length). Optionally TF-IDF. All cheap enough for CPU per-request inference (a few ms for MiniLM on a short prompt).
- Models compared with 5-fold stratified CV, grouped by sub-benchmark to test generalisation to unseen task types (leave-task-group-out) as well as the in-distribution split: logistic regression (lexical + embeddings), HistGradientBoosting, random forest, and a small MLP; calibrated with isotonic/Platt via `CalibratedClassifierCV`. Report AUC, Brier, ECE, then the routing metrics.
- Routing metrics, all computed on recorded outcomes at each threshold: quality retained = (mean score of routed mix) / (mean score of always-big); fraction to big; cost; CPT(90/95/99%); APGR; AIQ-style area under the convex hull; vs always-small, always-big, random (averaged over many seeds, with bands), and the oracle (route to big only when small is wrong and big is right).
- Test split by prompt, never by row after augmentation; a held-out test set untouched until final numbers.

## 5. Gateway design inputs

- OpenAI-compatible `POST /v1/chat/completions` (+ `GET /v1/models`, `/health`) so any OpenAI SDK points at it by `base_url`. Extra response field `router` carries predicted difficulty, tier, model, estimated cost and saving.
- Providers behind one interface: OpenAI chat completions, Anthropic Messages API (different request shape, `x-api-key`, `anthropic-version` header, system as a top-level field, `max_tokens` required), and local Qwen via transformers. Tested with `httpx.MockTransport`. Pricing config file with a lookup date; I will fetch current OpenAI and Anthropic price pages at build time and record the date, with the note that prices change.
- Policies: threshold, or target big-call fraction calibrated as a quantile of recent scores, or a budget (USD per 1k requests) enforced by an adaptive threshold. Escalation: small-tier error, timeout, empty output, or low confidence (mean token logprob under a threshold for the local model; a length/refusal heuristic for hosted) triggers re-run on the big tier, with the cost of both calls booked. Cache: exact-match LRU keyed on normalised messages + params. Ledger: SQLite, per request, with routed cost vs all-big counterfactual.

## 6. Risks found

1. RouterBench HF card has no licence field; mitigation above (code MIT, benchmark licences cited, not redistributed).
2. Pickle format: allow-list loader + conversion to Parquet.
3. The GPT-3.5/GPT-4 quality gap is not large; gains will be moderate. Will report actual numbers, including if the router barely beats random on some task mixes.
4. Local pair quality gap may be tiny on easy tasks and the router could show little benefit; report as is.
5. Tasks are benchmark-style (multiple choice, math), not open chat; the router may be learning "which benchmark" more than "difficulty". Leave-task-out CV exposes this.
6. CPU generation speed bounds the live experiment size.
