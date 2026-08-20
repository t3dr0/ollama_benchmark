# Results

Result data is **not stored in this repository**. It is deposited on Zenodo and cited by DOI, so
that the dataset has a permanent, citable identifier rather than depending on a repository path
that may change.

> **DOI: _to be added once the Zenodo deposit is published._**
> Cite the **version** DOI, not the concept DOI, so that a reader receives exactly the data behind
> the published figures. The concept DOI always resolves to the newest version, which is the wrong
> target for a published paper.

This repository holds the code; Zenodo holds the data. See [`../correctness/`](../correctness/) for
the evaluation code and [`../ollama_benchmark.py`](../ollama_benchmark.py) for the throughput
benchmark.

## What the deposit contains

### Throughput

`ollama_benchmark_results_21May2026.csv` — the single benchmark run reported in the paper.

| | |
|---|---|
| Host | Dell Precision T7810 |
| GPU | NVIDIA GeForce RTX 3060, 12 GB VRAM |
| Date | 2026-05-21 |
| Ollama | v0.24.0 |
| Coverage | 8 models × 12 measurement points × 5 repetitions |

The twelve measurement points are the ten prompts `P1`–`P10`, with `P9` (multi-turn dialogue)
sampled at turns 1, 3 and 5. Each row reports mean, variance and standard deviation of both
tokens/second and wall-clock elapsed time.

Only this run is published. Other benchmark runs exist locally but were provisional, or were taken
on hardware not covered by the paper, so they are deliberately excluded rather than shipped as
apparent supporting data.

### Correctness

For each model and benchmark:

- `results_*.json` — scores, task configuration, model arguments and generation settings
- `samples_*.jsonl` — **the per-item generations**, one record per benchmark item

The `samples_*.jsonl` files are the most useful artefact in the deposit. Because generation is
greedy and therefore deterministic, they allow any scoring procedure to be re-applied without
re-running a single model. Every correction reported in the paper was derived from these files
rather than by regenerating output.

Benchmarks are GSM8K, HumanEval (via the custom chat-aware task in `../correctness/tasks/`) and
TruthfulQA generative. Models covered: `gemma4:e4b`, `gemma3:12b`, `qwen3:14b`, `granite4.1:8b`,
`phi4:14b`, `deepseek-r1:1.5b`, `mistral-nemo:12b` and `gpt-oss:20b`, with `mistral-nemo:12b`
additionally evaluated at `Q4_K_M` as a quantisation control against the `Q4_0` served by its
default tag.

## Reproducing the originally published figures

The manuscript's first-submission HumanEval figures were produced by an earlier version of the
extraction code, retained unchanged at
[`../correctness/tasks/humaneval_local_chat/utils.py.ORIGINAL-as-submitted`](../correctness/tasks/humaneval_local_chat/utils.py.ORIGINAL-as-submitted).
Running it over the deposited generations reproduces those figures.

One caveat, stated plainly. This holds exactly for the models whose generations are identical
between runs. For `deepseek-r1:1.5b` and `qwen3:14b` the generation budget was raised from 4096 to
16384 tokens, because at the smaller budget those models exhausted their allowance while reasoning
and returned no answer at all on a substantial fraction of items. Their deposited generations are
therefore longer and genuinely different, and the original scores cannot be re-derived from them.

## Note on the CSV format

The throughput CSV carries two comment lines before the header, recording the timestamp, hostname
and GPU. Most readers handle this with `skiprows=2`, or `comment='#'` in `pandas.read_csv`.
