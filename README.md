# ModelRouter

Easy questions go to a small model, hard ones to a big model. A small scikit-learn classifier predicts, before any LLM runs, whether the cheap model will be good enough. An OpenAI-compatible gateway uses that score to route each request and keeps a ledger of what it saved.

![ModelRouter UI on desktop](docs/assets/served-desktop-top.png)

The `docs/` folder is a static copy of the interface that replays recorded results, so it can be hosted on GitHub Pages with no backend (see [Static demo](#static-demo)).

Numbers in this README are generated from `results/*.json` by `scripts/check_numbers.py` and checked by a test, so they cannot drift from the results files.

## What is in here

- **Offline lab.** A difficulty classifier trained on [RouterBench](https://github.com/withmartian/routerbench)'s recorded outcomes of <!--v:n_models-->11<!--/v--> LLMs on <!--v:n_total-->36,481<!--/v--> prompts. Because every model's answer is already recorded, a router can be scored with no API calls: for any threshold, quality and cost are read off the recorded outcomes of whichever model the router picked.
- **Gateway.** FastAPI, `POST /v1/chat/completions` (OpenAI schema, `model: "auto"`), threshold, target-fraction and budget policies, escalation from small to big on error, empty output or low confidence, an LRU cache, and a SQLite cost ledger that books routed cost against the send-everything-to-big counterfactual.
- **Providers.** OpenAI and Anthropic API adapters (httpx, retries, usage parsing) and a local transformers backend. No model ids are built in: tiers take their model ids from `config/gateway.yaml` or the environment. Prices come from an editable `config/pricing.yaml` with a lookup date.
- **Interface.** Real held-out prompts stream into a small lane and a big lane, a cost meter compares routed cost with all-big, and a threshold slider re-routes the whole evaluation set in the browser along the true cost-quality curve.

```mermaid
flowchart LR
  C[Client / OpenAI SDK] --> G[FastAPI gateway]
  G --> K{Cache}
  K -- hit --> C
  K -- miss --> F[MiniLM + lexical features]
  F --> R[Calibrated difficulty classifier]
  R --> P[Policy: threshold / fraction / budget]
  P -- easy --> S[Small tier]
  P -- hard --> B[Big tier]
  S --> E{Escalate?}
  E -- no --> L[(SQLite ledger)]
  E -- error / empty / low confidence --> B
  B --> L
  L --> C
  S & B -.-> A[OpenAI / Anthropic API / local Qwen adapters]
```

## Results

Design notes and the research behind the choices are in [`docs/notes/`](docs/notes). Reproduce everything with [the commands below](#reproduce).

### Offline: RouterBench recorded outcomes (the main evidence)

Headline pair A: `<!--v:small_a-->gpt-3.5-turbo-1106<!--/v-->` (small) against `<!--v:big_a-->gpt-4-1106-preview<!--/v-->` (big). Costs are RouterBench's recorded per-call costs at **November 2023** prices, kept separate from the current list prices in `config/pricing.yaml`. The data is split by prompt, stratified by sub-benchmark: <!--v:n_dev-->31,008<!--/v--> development prompts and <!--v:n_test-->5,473<!--/v--> held-out test prompts, which were scored once. The classifier is chosen by out-of-fold APGR on the development data: a <!--v:classifier-->calibrated random forest predicting that the small model gets the prompt wrong<!--/v-->. Features are all-MiniLM-L6-v2 embeddings plus <!--v:n_lexical-->16<!--/v--> lexical statistics, computed on the CPU in float32.

<!--BEGIN:offline_a-->
| Held-out test, pair A | value |
|---|---|
| always-small quality / always-big quality | 0.630 / 0.782 |
| AUC of the classifier | 0.720 |
| APGR (random routing is 0.5) | 0.693 |
| AIQ (random routing: 0.706) | 0.746 |
| CPT 90% / 95% / 99% (share of calls to the big model needed to keep that share of its quality) | 24% / 42% / 76% |
| at the 95% operating point (2,314 of 5,473 prompts to big) | quality 95.1% of big, cost 37% of all-big (63% cheaper; $1.23 vs $3.28 per 1,000 requests at Nov 2023 prices) |
| advantage over random routing at 30% to big (95% bootstrap interval) | +0.043 [+0.036, +0.048] |
| oracle CPT 95% (upper bound) | 11% |
<!--END:offline_a-->

In words: keeping <!--v:a_op_quality-->95.1%<!--/v--> of the big model's quality takes the <!--v:a_op_k-->2,314<!--/v--> hardest held-out prompts (<!--v:a_op_frac-->42.3%<!--/v-->, ties included) on the big model, which costs <!--v:a_op_cost-->37%<!--/v--> of sending everything there. The big model costs <!--v:a_cost_ratio-->13.6x<!--/v--> the small one per request on these recorded prices.

Pair B, `<!--v:small_b-->mistralai/mixtral-8x7b-chat<!--/v-->` against the same big model, has a wider quality gap (always-small quality <!--v:b_small_q-->0.551<!--/v-->): AUC <!--v:b_auc-->0.730<!--/v-->, APGR <!--v:b_apgr-->0.681<!--/v-->, and 95% of big's quality needs <!--v:b_cpt95-->60%<!--/v--> of calls on the big model (<!--v:b_op_cost-->66%<!--/v--> of the all-big cost, <!--v:b_op_quality-->95.0%<!--/v--> of its quality). Advantage over random routing at 30% to big: <!--v:b_adv30-->+0.066 [+0.060, +0.072]<!--/v-->.

Model comparison (4-fold CV on the development data, target "small model fails", pair A; calibration happens inside each fold):

<!--BEGIN:cv_a-->
| model | AUC | Brier | ECE | APGR | CPT 95% |
|---|---|---|---|---|---|
| logistic regression, lexical features only | 0.662 | 0.207 | 0.022 | 0.633 | 56% |
| logistic regression, embeddings + lexical | 0.695 | 0.200 | 0.012 | 0.638 | 55% |
| HistGradientBoosting | 0.705 | 0.197 | 0.008 | 0.650 | 53% |
| random forest (selected) | 0.702 | 0.198 | 0.008 | 0.652 | 53% |
| MLP | 0.689 | 0.202 | 0.021 | 0.631 | 56% |
| baseline: sub-benchmark identity only | 0.690 | 0.199 | 0.003 | 0.615 | 54% |
| baseline: benchmark family only | 0.606 | 0.214 | 0.001 | 0.583 | 63% |
<!--END:cv_a-->

### What the router actually learns (read this before trusting the headline)

Sub-benchmark identity alone gets APGR <!--v:cv_task_apgr-->0.615<!--/v--> in cross-validation, against <!--v:cv_rf_apgr-->0.652<!--/v--> for the selected model, so **most of the in-distribution skill is recognising which kind of task a prompt belongs to**, not judging how hard an individual item is. The leave-one-benchmark-family-out test makes that concrete: train on the other families, test on the held-out one.

<!--BEGIN:lofo_a-->
| held-out family | n | AUC | APGR | advantage vs random at 30% [95% CI] |
|---|---|---|---|---|
| arc-challenge | 1,470 | 0.576 | 0.584 | +0.013 [+0.004, +0.020] |
| gsm8k | 7,450 | 0.491 | 0.473 | -0.001 [-0.003, +0.001] |
| hellaswag | 10,042 | 0.505 | 0.519 | +0.007 [+0.001, +0.012] |
| mbpp | 425 | 0.504 | 0.458 | -0.002 [-0.018, +0.015] |
| mmlu | 14,037 | 0.594 | 0.542 | +0.007 [+0.004, +0.009] |
| mt-bench | 80 | 0.649 | 0.465 | -0.012 [-0.057, +0.027] |
| other | 1,710 | 0.457 | 0.571 | +0.006 [-0.001, +0.012] |
| winogrande | 1,267 | 0.496 | 0.496 | -0.001 [-0.015, +0.011] |
<!--END:lofo_a-->

On an unseen family the router is close to random: AUC is below 0.6 for <!--v:a_lofo_auc_lt60-->7<!--/v--> of <!--v:a_lofo_nfam-->8<!--/v--> families, the mean APGR is <!--v:a_lofo_mean_apgr-->0.514<!--/v--> and the best advantage over random routing is <!--v:a_lofo_max_adv-->+0.013<!--/v--> quality. The in-distribution gain is real on this benchmark mix, but it would not be safe to assume it transfers to different traffic without retraining on that traffic.

The same test for pair B (mean APGR <!--v:b_lofo_mean_apgr-->0.512<!--/v-->):

<!--BEGIN:lofo_b-->
| held-out family | n | AUC | APGR | advantage vs random at 30% [95% CI] |
|---|---|---|---|---|
| arc-challenge | 1,470 | 0.572 | 0.580 | +0.015 [+0.006, +0.024] |
| gsm8k | 7,450 | 0.495 | 0.496 | -0.000 [-0.003, +0.002] |
| hellaswag | 10,042 | 0.472 | 0.481 | -0.012 [-0.017, -0.008] |
| mbpp | 425 | 0.490 | 0.483 | -0.006 [-0.023, +0.011] |
| mmlu | 14,037 | 0.570 | 0.528 | +0.003 [+0.001, +0.005] |
| mt-bench | 80 | 0.635 | 0.635 | +0.016 [-0.012, +0.046] |
| other | 1,710 | 0.404 | 0.389 | -0.016 [-0.025, -0.007] |
| winogrande | 1,267 | 0.502 | 0.500 | +0.002 [-0.011, +0.015] |
<!--END:lofo_b-->

### Live: a small local pair on this machine (a demonstration, not evidence)

`<!--v:live_small-->Qwen2.5-0.5B-Instruct<!--/v-->` (small) and `<!--v:live_big-->Qwen2.5-1.5B-Instruct<!--/v-->` (big) answered <!--v:live_n-->300<!--/v--> multiple-choice questions (<!--v:live_per-->75<!--/v--> each from ARC-Easy, ARC-Challenge, CommonsenseQA and MMLU) through the same provider code the gateway uses: one model in memory at a time, one request at a time, on a <!--v:live_dev-->NVIDIA GeForce RTX 5060 Ti (cuda)<!--/v-->, <!--v:live_dtype-->float16<!--/v-->, <!--v:live_threads-->2<!--/v--> CPU threads, greedy decoding. The reply is forced to start with "The correct answer is" so the letter is emitted within four tokens.

<!--BEGIN:live_acc-->
| source | n | small | big |
|---|---|---|---|
| arc-challenge | 75 | 56.0% | 77.3% |
| arc-easy | 75 | 70.7% | 92.0% |
| commonsenseqa | 75 | 54.7% | 76.0% |
| mmlu | 75 | 48.0% | 68.0% |
| **all** | 300 | 57.3% | 78.3% |
| oracle (best of the two per question) | 300 | 82.0% | |
<!--END:live_acc-->

The big model was strictly better on <!--v:live_strict_n-->74<!--/v--> of <!--v:live_n-->300<!--/v--> questions (<!--v:live_strict-->25%<!--/v-->). Mean end-to-end latency per question was <!--v:live_lat_s-->1.04<!--/v--> s (small) and <!--v:live_lat_b-->1.84<!--/v--> s (big), on average <!--v:live_ptok-->111<!--/v--> prompt tokens, which is <!--v:live_tps_s-->107<!--/v--> and <!--v:live_tps_b-->60<!--/v--> prompt tokens per second. These are GPU figures, measured while the GPU and CPU were shared with other jobs, so they are indicative only; I did not measure CPU latency for this set. Cost for the local pair is a compute proxy (prompt tokens x parameters), not dollars.

<!--BEGIN:live_router-->
| router for this pair | AUC | APGR | advantage vs random at 30% [95% CI] |
|---|---|---|---|
| trained on the local labels, 5-fold CV | 0.493 | 0.492 | -0.018 [-0.046, +0.010] |
| benchmark identity only | 0.525 | 0.485 | -0.003 [-0.029, +0.023] |
| RouterBench router, transferred | 0.477 | 0.438 | -0.020 [-0.047, +0.008] |
<!--END:live_router-->

**Negative result.** With <!--v:live_n-->300<!--/v--> labels and four similar benchmarks, no router beats random routing here: every interval includes zero and the cross-validated router's AUC is <!--v:live_cv_auc-->0.493<!--/v-->. This experiment shows the mechanism end to end (two real models, real correctness labels, real latencies), not that routing helps for this pair. The gateway's default router is therefore the one trained on RouterBench, applied to the local pair as a demonstration of the plumbing.

## What was exercised for real

| path | status |
|---|---|
| RouterBench loading, classifier training, offline evaluation | run end to end here; numbers above |
| Local Qwen pair behind the gateway (playground, ledger, cache) | run end to end here |
| OpenAI and Anthropic API adapters | **tested against stubbed HTTP only** (`httpx.MockTransport`); never called against the live APIs, because I had no keys |
| Escalation, cache, policies, ledger | unit tests with stub providers, plus the local pair |

## Quickstart

```bash
git clone https://github.com/Aditya2616/ModelRouter && cd ModelRouter
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
modelrouter data                      # downloads RouterBench 0-shot (~100 MB), checks its SHA-256, writes data/routerbench_slim.parquet
python scripts/embed_routerbench.py   # MiniLM embeddings of the prompts (resumable)
modelrouter train --pair gpt35_vs_gpt4
modelrouter serve                     # http://127.0.0.1:8109
```

`modelrouter route "your prompt"` prints a difficulty score. The first request that runs a local model downloads Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct (about 4 GB). The local backend uses CUDA in float16 when it is available and otherwise the CPU in float32; override with `MODELROUTER_DEVICE=cpu` or `modelrouter live --device cpu`. Everything here, including the tests, works on a CPU-only machine; the live numbers above were produced on the GPU in float16 and I have not repeated that run on the CPU.

To use hosted tiers instead, copy `.env.example` to `.env`, set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, and set the tiers in `config/gateway.yaml` (or `MODELROUTER_SMALL_MODEL` / `MODELROUTER_BIG_MODEL`). The model ids are yours to choose: none are built in, and a hosted tier without one fails at start-up with a configuration error. Add the same ids to `config/pricing.yaml`. Keys are read from the environment and never logged or stored.

Any OpenAI client works against the gateway:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8109/v1", api_key="unused")
r = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "What is the capital of France?"}])
print(r.choices[0].message.content, r.model_extra["router"])   # difficulty, tier, cost, saving
```

### Prices

`config/pricing.yaml` holds list prices in US dollars per million tokens (input / output), looked up on <!--v:price_date-->2026-11-12<!--/v--> from the OpenAI pricing page. The default example pair is `<!--v:price_small_id-->gpt-6-luna<!--/v-->` at <!--v:price_small-->0.10 / 0.50<!--/v--> and `<!--v:price_big_id-->gpt-6.1-sol<!--/v-->` at <!--v:price_big-->2.00 / 10.00<!--/v-->. Prices change, so the file is meant to be edited, and it holds a commented template for an Anthropic API model rather than model ids. When the local pair runs, the gateway reports what the same tokens would cost on that pair; that is a labelled hypothetical, not a bill. These current prices are separate from the recorded November 2023 costs behind the offline results.

### Reproduce

```bash
python scripts/reproduce_results.py          # data, both pairs, live experiment, web data, docs build
python scripts/reproduce_results.py --skip-live
python scripts/check_numbers.py --write      # regenerate the numbers in this README from results/
pytest                                       # fast offline tests, including the README/results sync check
ruff check .
```

Run `scripts/embed_routerbench.py` first, otherwise `train` embeds the prompts itself. `pytest -m slow` runs a model test that needs the Qwen download.

### Static demo

`docs/` is the same interface with recorded data, ready for GitHub Pages: `python -m http.server -d docs 8109`. It ships only derived per-item numbers (predicted difficulty, each tier's recorded score and cost, task label). Prompt text is shown for a few dozen items from MMLU and GSM8K, whose dataset cards give an MIT licence, shortened and attributed; every other item shows its task label only. `scripts/screenshot.py` checks the page at 1440x900 and 390x844 (touch).

![Phone](docs/assets/served-phone-top.png)

## Design decisions

- **Predict, then verify.** A pre-call classifier decides the tier; a cheap post-call check (error, empty answer, low mean token log-probability locally) escalates to the big tier and books both costs.
- **Random routing is the baseline**, not always-small: mixing two models at random already gives a straight line between them, so a router must beat that line. Intervals come from a paired bootstrap.
- **Calibrated probabilities.** Scores are calibrated inside each fold, so one threshold means the same thing across policies; the fraction policy maps a target share to a score quantile.
- **Safe loading.** RouterBench ships as a pickle. `modelrouter data` checks a pinned SHA-256 and unpickles with an allow-list that only permits pandas and numpy reconstruction classes (tested against a crafted malicious pickle), then stores Parquet.
- **Numbers are mechanical.** The tables above and the figures in the prose are rendered from the results files, and the web data in `web/data` and `docs/data` is checked against them.

## Limitations

- RouterBench is benchmark-style (multiple choice, maths, code), not open-ended chat traffic, and its models and prices are from late 2023.
- The router does not generalise to unseen benchmark families (table above). Treat the headline savings as in-distribution.
- The live experiment is small (one run, four multiple-choice sources) and shows no routing benefit.
- The hosted adapters were never called against the real APIs. The prices in `config/pricing.yaml` were copied from the OpenAI pricing page on the lookup date and may have changed.
- Latencies come from a shared GPU.

## Data, models and licences

Code: MIT (see `LICENSE`). **RouterBench** (Hu et al., 2024): the dataset card on Hugging Face carries no licence, its code repository is MIT, and the data is not included here; it is downloaded at run time. Its prompts come from public benchmarks with their own licences. Also used: sentence-transformers `all-MiniLM-L6-v2` (Apache-2.0), Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct (Apache-2.0), ARC (CC-BY-SA-4.0), CommonsenseQA (MIT) and MMLU (MIT), as stated on their Hugging Face cards. Ideas and metrics from FrugalGPT, Hybrid LLM, AutoMix and RouteLLM (CPT and APGR); see `docs/notes/research.md`.

---

Built by [Aditya Jain](https://adityajain16.vercel.app) · [github.com/Aditya2616/ModelRouter](https://github.com/Aditya2616/ModelRouter)
