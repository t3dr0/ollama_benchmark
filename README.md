# Ollama LLM Local Inference Benchmarking Suite

A standalone, production-ready Python suite designed to execute local Large Language Model (LLM) performance audits via the official `ollama` daemon API.
 It measures two things that are easy to confuse and often move independently: **how fast** a model responds, and **whether
 the response is correct**.

 The throughput benchmark runs an automated matrix of **10 distinct prompt dimensions** across a specified array of
 open-weights models, collecting raw token processing metrics directly from the inference engine to eliminate external
 IPC/network overhead. The correctness benchmark scores the same models against **three public benchmarks with gold
 answers** (GSM8K, HumanEval and TruthfulQA) via EleutherAI's lm-evaluation-harness. See
 [`correctness/`](correctness/).

---

## 🌟 Introduction

### What is this?
This tool is essentially a "speedometer" and stress-tester for Artificial Intelligence models running locally on your own computer.
 Instead of sending text over the internet to services like OpenAI's ChatGPT, many users run open-source AI models (like Meta's Llama or DeepSeek)
 entirely on their own hardware using a background program called **Ollama**.
 This suite puts those local models through a rigorous battery of tests to see exactly how fast they process information
 and generate answers, and then a second battery to check whether the answers are actually right. A fast wrong answer is
 worth nothing, and speed turns out to be a poor predictor of accuracy, so both are measured.

### Supported Environments & Hardware
To ensure maximum portability, the suite automatically detects, adapts to, and audits performance across a massive range of computing setups:
* **Operating Systems (OS):** Fully compatible across **Linux** (Ubuntu, Debian, Fedora, Arch, etc.), **macOS** (Intel and Apple Silicon), and **Windows** (via native Command Prompt, PowerShell, or WSL).
* **Graphics Processing Units (GPUs):** Built to profile high-performance discrete hardware accelerators including **NVIDIA** (Tensor Core architectures from consumer RTX cards to enterprise A100/H100/B200 tracks) and **AMD** (Radeon and Instinct MI-series via ROCm environments), as well as **Apple Silicon Unified Memory** (M1/M2/M3/M4 Pro, Max, and Ultra chips).
* **Central Processing Units (CPUs):** Optimized to gracefully fallback and measure traditional compute architectures on both standard **x86_64** (Intel Core/Xeon and AMD Ryzen/EPYC) and energy-efficient **ARM64** processors when VRAM boundaries are exceeded.

### Why was it created?
When you run an AI model locally, its performance depends entirely on your machine's hardware—specifically your graphics card (GPU) and processor (CPU).
A model might run lightning-fast when answering a short trivia question, but slow to a crawl when reading a long document or writing complex code. 

This benchmarking suite was created to provide a **scientific, standardized way to measure AI performance**.
 Rather than relying on guesswork ("it feels fast today"), this script runs every model through 10 distinct types of real-world tasks,
 repeats each test 5 times, and calculates exact, reliable speeds.

Speed alone is not enough to choose a model, however. A model can be quick, stable, and consistently wrong. The correctness
 benchmark therefore scores the same models on arithmetic reasoning, code that must actually execute and pass its unit tests,
 and resistance to plausible-but-false answers, so that the two axes can be weighed together. It also includes automatic safeguards
 to dynamically catch and repair common network routing traps (like the `0.0.0.0` client deadlock) that often frustrate users setting up local AI servers.

### Who is it for?
* **Hardware Enthusiasts & Gamers:** Anyone wanting to see exactly how their premium graphics card or unified memory rig handles heavy AI workloads.
* **Developers & System Administrators:** Engineers designing local AI applications who need empirical metrics to choose the right balance between a model's size (intelligence) and token delivery speed.
* **Academic Researchers:** Scholars requiring hard, empirical data on local LLM efficiency, variance, and hardware stability for peer-reviewed scientific papers.

### Research Reference
The implementation was created to support the project and paper
**Reusing Obsolete Windows 10 PCs for On-Premises Large Language Model Inference**
by I. Curington and K. Lano (2026).


### Elapsed Time Warning
Neither benchmark is quick. The throughput benchmark takes approximately **five to six hours** on an RTX 3060.
The correctness benchmark is considerably longer, since it runs roughly 2,300 items per model rather than sixty:
budget **one to four hours per model per benchmark**, depending on the model's size and whether it reasons by default.
Models that exceed available VRAM and spill to CPU are slower again. Run both unattended.

---

## 📊 Key Architectural Features

* **Multi-Dimensional Evaluation:** Tests 10 diverse constraints including short factual retrieval, 2,300+ word long-context KV cache stress testing, strict instruction following, structural coding patterns, multi-step math logic, and automated 5-turn dialog state accumulation.
* **Rigorous Statistical Aggregations:** Executes multiple sequential evaluation iterations ($N=5$) preceded by an unrecorded model warm-up phase to isolate initial VRAM weight mapping latency. Automatically computes Mean ($\bar{X}$), Variance ($\sigma^2$), and Standard Deviation ($\sigma$) for both *Tokens per Second (TPS)* and *Elapsed Wall-Clock Time*.
* **Automated Network Fail-Safe:** Dynamically detects server-side `0.0.0.0` address configurations in the `OLLAMA_HOST` variable and patches local execution context routing natively to loopback `127.0.0.1`, avoiding unroutable client HTTP connection faults.
* **Dual Output Engine:** Generates a highly scannable ASCII dashboard table in the console terminal while streaming a fully typed, commented, and timestamped CSV file for formal downstream data analysis.

---

## 🛠️ Prerequisites & Local Environment Requirements

Before initiating the suite, ensure your environment has the core dependencies installed.

### External Dependencies

This benchmarking suite requires Ollama to be installed and running locally as a system service. 

* **Ollama Engine:** v0.3.0+ recommended (v0.1.48 minimum required for API compatibility).
* **Download:** [https://ollama.com/download](https://ollama.com/download)
* **Version caveat:** `gpt-oss:20b` fails to serve under Ollama v0.32.3 with a tensor-size error affecting its MXFP4
  mixture-of-experts layout. This is fixed by v0.32.6. If you intend to benchmark that model, use v0.32.6 or later.
* **For the correctness benchmark only:** EleutherAI's lm-evaluation-harness, installed with the API extra:
  `pip install "lm-eval[api]"`. The base install omits `tenacity`, which the API-backed model classes require.

Ensure the Ollama server is running natively (`ollama serve`) and that your target models are pulled before running the benchmark driver.

### Supported Benchmark Model Matrix

Both benchmarks target the same eight open-weights models, so the throughput and correctness results are directly
comparable model by model. Ensure the Ollama server is running natively (`ollama serve`) and that your target models are
pulled before running either driver.

```bash
ollama pull gemma4:e4b
ollama pull gemma3:12b
ollama pull qwen3:14b
ollama pull granite4.1:8b
ollama pull phi4:14b
ollama pull deepseek-r1:1.5b
ollama pull mistral-nemo:12b
ollama pull gpt-oss:20b
```

A ninth tag is used only as a control. `mistral-nemo:12b` is the one model whose default tag serves `Q4_0` rather than
`Q4_K_M`, so it is additionally evaluated at matched quantisation to confirm that the difference is attributable to the
model rather than to the quantisation its default tag happens to supply:

```bash
ollama pull mistral-nemo:12b-instruct-2407-q4_K_M
```

> **Worth knowing:** Ollama's default tag does not guarantee a consistent quantisation across models, and the choice is
 not surfaced during a normal pull. Practitioners comparing models on default tags may unknowingly be comparing
 different quantisations.

> **Note:** The suite contains an integrated system discovery check.
 If models are missing, the script will isolate them gracefully,
 output instructions for pulling them, and safely proceed with the active inventory found on your machine.

The benchmark code includes this simple list of models in one place - easily modified for a different collection of models.
Once launched, it runs uninterrupted until completion, without prompting for any input.

---

## 🚀 Execution Guide

### Prerequisites

Before executing the benchmark or generating publication assets, ensure your local environment has the required dependencies installed by running:
```bash
pip install -r requirements.txt
```

Run the self-contained script from your preferred terminal emulator:

```bash
python ollama_benchmark.py

```

### Overriding Host Environment Values Natively
The script connects to an Ollama service using the environment variable **OLLAMA_HOST**.
If your background daemon maps to an alternate port or an external server interface, pass it inline during the thread call:

```bash
# Linux / macOS
OLLAMA_HOST=127.0.0.1:11434 python ollama_benchmark.py

# Windows PowerShell
$env:OLLAMA_HOST="127.0.0.1:11434"; python ollama_benchmark.py

```

### Running the Correctness Benchmark

The correctness axis is driven by lm-evaluation-harness rather than by this repository's own script. Install it, then
point it at the custom task definitions in [`correctness/tasks/`](correctness/tasks/):

```bash
pip install "lm-eval[api]"
mkdir -p local_results

# GSM8K
lm_eval run --model local-chat-completions \
  --model_args model=gemma4:e4b,base_url=http://localhost:11434/v1/chat/completions,num_concurrent=1,tokenized_requests=False \
  --tasks gsm8k --apply_chat_template --log_samples \
  --gen_kwargs max_tokens=16384 --output_path local_results/

# HumanEval, using the chat-aware task in correctness/tasks/
HF_ALLOW_CODE_EVAL=1 lm_eval run --model local-chat-completions \
  --model_args model=gemma4:e4b,base_url=http://localhost:11434/v1/chat/completions,num_concurrent=1,tokenized_requests=False \
  --tasks humaneval_local_chat --include_path correctness/tasks \
  --apply_chat_template --log_samples --confirm_run_unsafe_code \
  --gen_kwargs max_tokens=16384 --output_path local_results/
```

Three things matter here and are easy to get wrong:

* **Use a generous `max_tokens`.** Reasoning models return their chain of thought in a separate response field, and the
  harness reads only the answer field. A model that exhausts its budget while reasoning is recorded as returning an empty
  answer, indistinguishable from a wrong one. 4096 is not sufficient for HumanEval.
* **HumanEval executes model-generated code**, so it needs `HF_ALLOW_CODE_EVAL=1` and `--confirm_run_unsafe_code`. Run it
  from a scratch directory.
* **Pick the right TruthfulQA variant:** `truthfulqa_gen_chat` for reasoning-capable models, `truthfulqa_gen` for the rest.
  Applying the wrong one to a reasoning model returns a large fraction of empty responses.

Output is written to `local_results/` rather than `results/`, which in this repository holds only a pointer to the
deposited datasets. Add `local_results/` to your `.gitignore` if you intend to commit from a working clone.

Ready-made sweep scripts covering all models and benchmarks are in
[`correctness/scripts/`](correctness/scripts/). Expect the correctness sweep to take considerably longer than the
throughput benchmark: it is thousands of items per model rather than sixty.

---

## 🧪 Evaluation Dimension Mapping

The test harness evaluates structural inference bottlenecks across 10 core dimensions:

| Dimension ID | Benchmark Domain | Stress/Testing Mechanism |
| --- | --- | --- |
| **P1** | Short Factual Retrieval | Baseline token initialization speed, minimal output state |
| **P2** | Long Structured Prose | Extended autoregressive generation consistency |
| **P3** | Arithmetic Reasoning | Chain-of-Thought (CoT) tracking capability |
| **P4** | Structural Code Generation | Code compilation, syntax constraints, and markdown format handling |
| **P5** | Long-Context Summarization | Stresses KV cache memory bandwidth with a ~2,300-word payload |
| **P6** | Constraint Adherence | Negative word exclusions, precise line/paragraph boundaries |
| **P7** | Deductive Logic | Non-mathematical step-by-step reasoning sequences |
| **P8** | Multilingual Matrix | Evaluates translation matrix shifts and alternative token structures |
| **P9** | Multi-Turn Dialogue | Accumulates session context over 5 turns; audits performance at T1, T3, T5 |
| **P10** | Refusal Alignment | Safety alignments vs. functional edge-case profiling |

---

## 📄 Output Reporting Formats

### 1. Terminal Console Metrics Table

Upon benchmark completion, a clean ASCII report prints system configurations (Hostname, OS, Accelerator GPU, and VRAM limits) alongside finalized statistical performance matrices:

```text
=====================================================================================
 SUMMARY BENCHMARK OUTPUT ENGINE REPORT
=====================================================================================
Model                  | ID     | Metric Description        | Mean Value    | Std Dev   
-------------------------------------------------------------------------------------
deepseek-R1:1.5b       | P1     | Tokens_Per_Sec            |  145.20 t/sec |     3.10
deepseek-R1:1.5b       | P1     | Elapsed_Sec               |    0.20 sec   |     0.00
qwen3:14b              | P4     | Tokens_Per_Sec            |   61.90 t/sec |     0.70
...
=====================================================================================

```

### 2. Analytical Structured CSV Export

A data file named `ollama_benchmark_results.csv` is written to the root working directory. It contains comment headers tracking runtime machine metadata,
 perfectly formatted for academic ingestion scripts, spreadsheet software, or plotting applications like `matplotlib`.

Result data is **not stored in this repository**. The benchmark run reported in the paper, together
with the correctness scores and per-item generations, is deposited on Zenodo and cited by DOI; see
[`results/README.md`](results/README.md).

---

### Analysis & Visualization Utilities

The benchmarking suite includes two specialized post-processing utilities located in the root directory.
 These scripts ingest the generated raw benchmark CSV files to produce publication-grade assets for research papers and technical presentations:

1. **LaTeX Table Generator (`csv_to_tex.py`):** Automatically processes the evaluation metrics to compile structured data tables ready for academic publication.
 It outputs both a condensed, abridged version (`*_summary.tex`) for the main body of a paper,
 and a complete landscape table (`*_full.tex`) optimized for comprehensive appendix documentation.
2. **High-Density Heatmap Plotter (`csv_to_heatmap.py`):** Translates raw tabular rows
 into publication-quality matrix visualizations using `matplotlib` and `seaborn`.
 It features dual operational modes: profiling raw inference velocity via Mean Tokens Per Second (`--mode tps`) to instantly highlight model offloading cliffs,
 or profiling systemic variance via the Coefficient of Variation (`--mode cov`) to map runtime stability and hardware memory stress.
 Passing `--barchart` additionally produces the grouped bar chart of mean throughput per model used as a figure in the paper.
3. **HumanEval Failure Triage (`inspect_humaneval_samples.py`):** Classifies failed HumanEval items
 by cause, distinguishing genuinely incorrect code from harness-side extraction failures, and can
 re-score saved generations without re-running any model. See `correctness/README.md`.

---

## 🧠 Correctness Evaluation

Throughput describes how fast a model responds, not whether the response is right. A second axis
scores output correctness against three public benchmarks with gold answers, using EleutherAI's
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) against the same
locally-served Ollama models:

| Benchmark | Measures | Scoring |
|---|---|---|
| **GSM8K** | grade-school arithmetic reasoning | flexible-extract exact match |
| **HumanEval** | Python code correctness | pass@1, executed against each problem's unit tests |
| **TruthfulQA** (generative) | resistance to plausible-but-false answers | BLEU and ROUGE-1 `acc` |

Everything for this axis lives in **[`correctness/`](correctness/)**: the task definitions, the
sweep and rerun scripts, and a README explaining the method.

### Why a custom HumanEval task

Neither HumanEval variant packaged with the harness works correctly against a chat-completions
backend. The standard task assumes a raw-completion model and stops on sequences a chat-tuned model
does not emit at that position; the `instruct` variant relies on prefilling the assistant turn,
which Ollama's OpenAI-compatible endpoint accepts and silently ignores. `humaneval_local_chat`
bridges this.

### Four failure modes worth knowing about

Each of these silently produces false negatives rather than raising an error, so none is visible
from the scores alone. Anyone scoring chat-served or reasoning models locally should expect to meet
them; `correctness/README.md` documents each in full.

1. **Wrong fenced block** — models often emit a solution followed by a usage example, so taking the
   last code block scores the example and discards the answer.
2. **The model's own test assertions** — frequently appended, not always correct, and they abort the
   module before the hidden test runs.
3. **Reasoning that never terminates** — reasoning models return chain-of-thought in a separate
   response field, so a model that exhausts its token budget while thinking is recorded as returning
   an empty answer, indistinguishable from a wrong one.
4. **Stop sequences matching inside reasoning traces** — use `truthfulqa_gen_chat` for
   reasoning-capable models and `truthfulqa_gen` for the rest.

---

## 🤝 Contributing & Academic Use

Feel free to fork this repository, add additional models to the execution list, or modify the test suite prompts.
 If using these benchmarks or formatting metrics in an academic journal or conference paper, please ensure proper citation of hardware states and seed settings.

License: [AGPLv3](https://www.gnu.org/licenses/agpl-3.0.html)




