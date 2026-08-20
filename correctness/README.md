# Correctness evaluation

Code for the output-correctness axis of Curington & Lano (2026), *Reusing Obsolete Windows 10 PCs
for On-Premises Large Language Model Inference*. The throughput axis is the top-level
`ollama_benchmark.py`; this directory covers the benchmark scoring reported in Table 8.

**Results are not stored here.** Scores, per-item generations and derived summaries are deposited
on Zenodo; see `../results/README.md`.

## What this is for

Models are scored against three public benchmarks using EleutherAI's
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), served locally by
Ollama through its OpenAI-compatible chat-completions endpoint:

- **GSM8K** — grade-school arithmetic, scored by flexible-extract exact match
- **HumanEval** — Python code correctness, scored pass@1 against each problem's unit tests
- **TruthfulQA (generative)** — scored by BLEU and ROUGE-1 `acc`

## Layout

    tasks/
      humaneval_local_chat/
        humaneval_local_chat.yaml          task definition
        utils.py                           prompt construction, code extraction, pass@k
        utils.py.ORIGINAL-as-submitted     the pre-correction extractor (see below)
      truthfulqa_gen_chat.yaml             TruthfulQA variant for reasoning models
    scripts/
      run_full_sweep.sh                    original sweep, all models x all benchmarks
      rerun_fixed_benchmarks.sh            corrected TruthfulQA and HumanEval reruns
      rerun_humaneval_faultB.sh            regeneration at a larger token budget
      rerun_mistral_q4km_control.sh        quantisation control (Q4_0 vs Q4_K_M)
      run_gptoss_sweep.sh                  gpt-oss:20b sweep
      p1_p5_token_counts.py                generated-token counts for prompts P1 and P5
      gptoss_probe.py                      one-request check that gpt-oss:20b can be served

## Why a custom HumanEval task

Neither HumanEval variant packaged with the harness works correctly against a chat-completions
backend. The standard `humaneval` task assumes a raw-completion model and stops generation on
sequences a chat-tuned model does not emit at that position. The `humaneval_instruct` variant
relies on prefilling the assistant turn, which Ollama's OpenAI-compatible endpoint accepts and
silently ignores. `humaneval_local_chat` bridges this.

## Failure modes this code exists to avoid

Each of the following silently produces false negatives rather than raising an error, so none is
detectable from the reported scores alone. Anyone scoring chat-served or reasoning models should
expect to meet them.

1. **Wrong fenced block.** Models often emit a solution followed by a usage example or a
   `doctest` call. Selecting the last fenced block therefore scores the example and discards the
   answer. `extract_code()` selects the last block that *defines the function under test*.

2. **The model's own test assertions.** Models frequently append their own `assert` statements,
   which are not always correct, and which abort the module before the hidden test runs. Standard
   practice scores the completion rather than the model's scaffolding, so top-level statements
   other than imports, definitions and assignments are stripped.

3. **Reasoning that never terminates.** Reasoning models return their chain of thought in a
   separate response field, and the harness reads only the answer field. A model that exhausts its
   token budget while reasoning is recorded as returning an empty answer, indistinguishable from a
   wrong one. Use a generous `--gen_kwargs max_tokens=...`; 4096 is not sufficient for HumanEval.

4. **Stop sequences matching inside reasoning traces.** TruthfulQA's default stop sequence can
   match within a reasoning trace, truncating before any answer is produced. Use
   `truthfulqa_gen_chat` for reasoning-capable models and `truthfulqa_gen` for the rest. Applying
   the wrong variant to a reasoning model yields a large fraction of empty responses.

## Reproducing the figures as originally published

`utils.py.ORIGINAL-as-submitted` is the extractor used for the manuscript's first-submission
figures, retained unchanged. Because the published per-item generations are greedy and
deterministic, running that extractor over them reproduces the original scores.

One caveat. This holds exactly for the models unaffected by failure mode 3, whose generations are
byte-identical between runs. For `deepseek-r1:1.5b` and `qwen3:14b` the token budget was raised
from 4096 to 16384, so items previously truncated mid-reasoning produce genuinely different,
longer output; their original scores cannot be re-derived from the published data.

## Requirements

    pip install "lm-eval[api]"

HumanEval executes model-generated code, so it requires `HF_ALLOW_CODE_EVAL=1` and
`--confirm_run_unsafe_code`. Run it in a scratch directory.

The shell scripts assume an `lm-eval` virtual environment at `$HOME/lm-eval-env` and a local Ollama
daemon on `localhost:11434`. Adjust those paths for your environment.
