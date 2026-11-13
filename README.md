# ModelRouter

Easy questions go to a small model, hard ones to a big model. A small scikit-learn classifier predicts, before any LLM runs, whether the cheap model will be good enough. An OpenAI-compatible gateway uses that score to route each request and keeps a ledger of what it saved.

![The request stream mid-flow on desktop](docs/assets/served-desktop-stream.png)

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

Headline pair A: `<!--v:small_a-->gpt-3.5-turbo-1106<!--/v-->` (small) against `<!--v:big_a-->gpt-4-1106-preview<!--/v-->` (big). Costs are RouterBench's recorded per-call costs at **November 2023** prices, kept separate from the current list prices in `config/pricing.yaml`. The data is split by prompt, stratified by sub-benchmark: <!--v:n_dev-->31,008<!--/v--> development prompts and <!--v:n_test-->5,473<!--/v--> held-out test prompts, which were scored once. No prompt can sit on both sides of the split: comparing prompts by a normalised key (lower case, single spaces, and for GSM8K the real question after the shared few-shot block) gives <!--v:dup_keys-->36,481<!--/v--> unique keys for <!--v:n_total-->36,481<!--/v--> prompts, <!--v:dup_rows-->0<!--/v--> duplicates (`scripts/check_duplicates.py`; `train` refuses to run if that stops being true), so grouping the split by prompt changes nothing and the numbers below did not need re-running. The classifier is chosen by out-of-fold APGR on the development data: a <!--v:classifier-->calibrated random forest predicting that the small model gets the prompt wrong<!--/v-->. Features are all-MiniLM-L6-v2 embeddings plus <!--v:n_lexical-->16<!--/v--> lexical statistics, computed on the CPU in float32.

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

### Live: a small local pair on this machine

`<!--v:live_small-->Qwen2.5-0.5B-Instruct<!--/v-->` (small) and `<!--v:live_big-->Qwen2.5-1.5B-Instruct<!--/v-->` (big) answered <!--v:live_n-->3,000<!--/v--> multiple-choice questions (<!--v:live_per-->750<!--/v--> each from ARC-Easy, ARC-Challenge, CommonsenseQA and MMLU; licences CC-BY-SA-4.0, CC-BY-SA-4.0, MIT and MIT, as stated on the dataset cards) with the same prompt and the same forced answer prefix ("The correct answer is", so the letter comes out within four tokens), on a <!--v:live_dev-->NVIDIA GeForce RTX 5060 Ti (cuda)<!--/v-->, <!--v:live_dtype-->float16<!--/v-->, greedy decoding, batches of <!--v:live_batch-->32<!--/v-->, one model in memory at a time. Questions are de-duplicated across the whole pool before any split, then split by seed within each source into <!--v:live_ntr-->2,000<!--/v--> train, <!--v:live_nva-->500<!--/v--> validation and <!--v:live_nte-->500<!--/v--> test questions (<!--v:live_te_per-->125<!--/v--> per source in the test set); the analysis asserts that no prompt sits in two splits. Routers are fitted on train, chosen on validation and scored once on test. The target is "the big model is strictly right where the small one is wrong". Cost is a compute proxy (prompt tokens x parameters), not dollars.

<!--BEGIN:live_acc-->
| source | n | small | big | oracle |
|---|---|---|---|---|
| arc-challenge | 750 | 49.9% | 76.1% | 80.8% |
| arc-easy | 750 | 70.5% | 90.4% | 92.3% |
| commonsenseqa | 750 | 52.7% | 69.9% | 77.2% |
| mmlu | 750 | 46.7% | 59.9% | 69.7% |
| **all** | 3,000 | 54.9% | 74.1% | 80.0% |
| **test split only** | 500 | 53.4% | 70.8% | 78.2% |
<!--END:live_acc-->

The big model was strictly better on <!--v:live_strict_n-->752<!--/v--> of <!--v:live_n-->3,000<!--/v--> questions (<!--v:live_strict-->25%<!--/v-->), so the oracle that always picks the better model reaches <!--v:live_acc_o-->80.0%<!--/v--> and a router that never wastes a big-model call would need only <!--v:live_oracle_cpt95-->14%<!--/v--> of calls on the big model to keep 95% of its accuracy on the test set.

<!--BEGIN:live_router-->
| router for this pair (test set) | AUC | APGR | CPT 95% | advantage vs random at 30% | validation APGR |
|---|---|---|---|---|---|
| trained on the local labels | 0.463 [0.406, 0.524] | 0.474 [0.392, 0.555] | 85% [62%, 90%] | -0.007 [-0.026, +0.016] | 0.520 |
| benchmark identity only | 0.548 [0.486, 0.605] | 0.533 [0.462, 0.603] | 82% [72%, 87%] | +0.023 [+0.004, +0.044] | 0.554 |
| RouterBench router, transferred | 0.478 [0.423, 0.542] | 0.382 [0.276, 0.469] | 84% [77%, 93%] | -0.028 [-0.052, -0.004] | 0.473 |
| oracle (upper bound) | | | 14% | | |
<!--END:live_router-->

Intervals are 95% bootstrap intervals over the <!--v:live_nte-->500<!--/v--> test questions. "Trained on the local labels" is a calibrated <!--v:live_local_chosen-->lr_C0.01<!--/v--> on MiniLM embeddings plus lexical features, picked from seven candidates (six logistic regressions and gradient boosting) by validation APGR. The benchmark-identity row predicts each question's chance from its source alone, so it is **not deployable** in a gateway (a request does not arrive labelled with its benchmark); it is there to show what knowing the source is worth. The transferred row applies the pair-A RouterBench router unchanged.

**What this shows.** Of the routers that can score a raw prompt, none beats random routing on the test set: the locally trained router's advantage over random at 30% to big is <!--v:live_local_adv-->-0.007 [-0.026, +0.016]<!--/v--> (its AUC is <!--v:live_local_auc-->0.463<!--/v-->) and the transferred RouterBench router is significantly worse than random (<!--v:live_tr_adv-->-0.028 [-0.052, -0.004]<!--/v-->). The only router with an interval above zero is the benchmark-identity one, which needs the label. That matches the offline finding that most of the skill is recognising the benchmark: here the source label is worth a little (the one row above zero) and the embedding adds nothing on top of it. I did not investigate further why the 1.5B model's gains are unpredictable from the question text. It is a negative result for this pair and these questions, not a claim that routing cannot help.

Holding one source out entirely (train on the other three, test on all <!--v:live_src_n-->750<!--/v--> questions of the fourth, same model type):

<!--BEGIN:live_lofo-->
| held-out source | n | AUC | APGR | advantage vs random at 30% [95% CI] |
|---|---|---|---|---|
| arc-challenge | 750 | 0.507 | 0.521 | -0.002 [-0.018, +0.016] |
| arc-easy | 750 | 0.534 | 0.534 | +0.011 [-0.004, +0.025] |
| commonsenseqa | 750 | 0.498 | 0.520 | +0.004 [-0.013, +0.019] |
| mmlu | 750 | 0.495 | 0.483 | -0.003 [-0.021, +0.016] |
<!--END:live_lofo-->

**Which router the gateway uses.** The default is the router trained on the local labels (`router_pair: local_qwen` in `config/gateway.yaml`, written by `modelrouter live`; the gateway falls back to the RouterBench router if that file does not exist yet). It was chosen because, among the routers that score a raw prompt, it has the best validation APGR (<!--v:live_val_local-->0.520<!--/v--> against <!--v:live_val_tr-->0.473<!--/v--> for the transferred one; the choice used no test data). "Best" here means least bad: its test advantage over random is indistinguishable from zero, so for this pair the gateway's routing is, within error, as good as a coin flip at the same big-model share. Its policy is the fraction policy at <!--v:gw_frac-->30%<!--/v--> big. Use it as a demonstration of the plumbing, and expect no saving that you could not get by splitting traffic at random.

Latency and throughput on the GPU (measured while the machine was shared with other jobs; just before each timing run the GPU was <!--v:live_gpu_util_s-->27<!--/v-->% busy with <!--v:live_gpu_mem_s-->4,908<!--/v--> MiB in use by other processes, so treat these as indicative):

<!--BEGIN:live_latency-->
| tier | single request, mean / median / 95th percentile | prompt tokens per second, single | questions per second, batches of 32 |
|---|---|---|---|
| small (Qwen2.5-0.5B-Instruct) | 107 / 100 / 133 ms | 932 | 160 |
| big (Qwen2.5-1.5B-Instruct) | 120 / 113 / 149 ms | 826 | 73 |
<!--END:live_latency-->

A single request is only <!--v:live_lat_ratio-->1.1<!--/v-->x slower on the big model (<!--v:live_lat_s-->107<!--/v--> ms against <!--v:live_lat_b-->120<!--/v--> ms, averages over <!--v:live_lat_n-->120<!--/v--> test questions, mean <!--v:live_ptok1-->99<!--/v--> prompt tokens), because a 100-token prefill on a modern GPU is dominated by per-step overhead rather than model size. The size difference shows up when the GPU is busy: batched, the small model answers <!--v:live_bat_ratio-->2.2<!--/v-->x as many questions per second. I did not measure CPU latency for this set.

## What was exercised for real

| path | status |
|---|---|
| RouterBench loading, classifier training, offline evaluation | run end to end here; numbers above |
| Live experiment: both Qwen models on 3,000 questions, three routers compared on a held-out test split | run end to end here on the GPU (float16); numbers above |
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
modelrouter live                      # optional: 3,000 questions through both Qwen models, trains the local router (~4 GB of models; minutes on a GPU, long on a CPU)
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

Run `scripts/embed_routerbench.py` first, otherwise `train` embeds the prompts itself. `pytest -m slow` runs two model tests that need the Qwen download.

I ran `scripts/reproduce_results.py` once from a fresh clone of the committed repository (no data, no models, no caches in the folder) on a <!--v:repro_gpu-->NVIDIA GeForce RTX 5060 Ti<!--/v--> machine with <!--v:repro_threads-->12<!--/v--> CPU threads and PyTorch <!--v:repro_torch-->2.11.0+cu128<!--/v-->, with the Hugging Face cache already holding the models and the live-experiment datasets. It took <!--v:repro_min-->52<!--/v--> minutes in total, while other jobs shared the machine:

<!--BEGIN:repro_steps-->
| step | seconds |
|---|---|
| download and convert RouterBench (about 100 MB) | 7 |
| MiniLM embeddings of 36,481 prompts (CPU, float32) | 928 |
| duplicate check | 1 |
| train and evaluate, pair A (CPU) | 1,054 |
| train and evaluate, pair B (CPU) | 1,009 |
| live experiment: 3,000 questions through both models, three routers (GPU, float16) | 147 |
| export web data | 1 |
| build docs | 0 |
| regenerate README numbers | 1 |
| **total** | 3,149 |
<!--END:repro_steps-->

At the end the script compares each regenerated file in `results/` with the committed one. All <!--v:repro_nfiles-->7<!--/v--> files (the JSON results, the test-set predictions and the live test items) regenerated identically, apart from timings (training seconds, latency, throughput and the GPU load recorded before each timing run), which depend on what else the machine is doing and differ between runs. The README numbers that depend on those timings therefore moved slightly in the clone and were not copied back. The full record, with the comparison, is in `results/reproduce.json`.

### Static demo

`docs/` is the same interface with recorded data, ready for GitHub Pages: `python -m http.server -d docs 8109`. It ships only derived per-item numbers (predicted difficulty, each tier's recorded score and cost, task label). Prompt text is shown only for sources whose dataset card gives a licence that permits redistribution, shortened, with an attribution line in the page: MMLU (Hendrycks et al., MIT licence) and GSM8K (Cobbe et al., MIT licence). Every other item shows its task label only. `scripts/ui_check.py` checks the page at 1440x900 and 390x844 (touch) and writes the screenshots in `docs/assets/`.

![Phone](docs/assets/served-phone-presets.png)

## Design decisions

- **Predict, then verify.** A pre-call classifier decides the tier; a cheap post-call check (error, empty answer, low mean token log-probability locally) escalates to the big tier and books both costs.
- **Random routing is the baseline**, not always-small: mixing two models at random already gives a straight line between them, so a router must beat that line. Intervals come from a paired bootstrap.
- **Calibrated probabilities.** Scores are calibrated inside each fold, so one threshold means the same thing across policies; the fraction policy maps a target share to a score quantile.
- **Safe loading.** RouterBench ships as a pickle. `modelrouter data` checks a pinned SHA-256 and unpickles with an allow-list that only permits pandas and numpy reconstruction classes (tested against a crafted malicious pickle), then stores Parquet.
- **Numbers are mechanical.** The tables above and the figures in the prose are rendered from the results files, and the web data in `web/data` and `docs/data` is checked against them.

## Limitations

- RouterBench is benchmark-style (multiple choice, maths, code), not open-ended chat traffic, and its models and prices are from late 2023.
- The router does not generalise to unseen benchmark families (table above). Treat the headline savings as in-distribution.
- The live experiment is one run on four multiple-choice sources with a 500-question test set, and it shows no routing benefit for the routers that can score a raw prompt. Both Qwen models may have seen these public benchmarks during training; I did not check.
- The hosted adapters were never called against the real APIs. The prices in `config/pricing.yaml` were copied from the OpenAI pricing page on the lookup date and may have changed.
- Latencies come from a shared GPU and are indicative; there is no CPU latency measurement.

## Data, models and licences

Code: MIT (see `LICENSE`). **RouterBench** (Hu et al., 2024): the dataset card on Hugging Face carries no licence, its code repository is MIT, and the data is not included here; it is downloaded at run time. Its prompts come from public benchmarks with their own licences. Also used: sentence-transformers `all-MiniLM-L6-v2` (Apache-2.0), Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct (Apache-2.0), ARC (CC-BY-SA-4.0), CommonsenseQA (MIT) and MMLU (MIT), as stated on their Hugging Face cards. Ideas and metrics from FrugalGPT, Hybrid LLM, AutoMix and RouteLLM (CPT and APGR); see `docs/notes/research.md`.

---

Built by [Aditya Jain](https://adityajain16.vercel.app) · [github.com/Aditya2616/ModelRouter](https://github.com/Aditya2616/ModelRouter)
