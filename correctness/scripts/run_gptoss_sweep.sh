#!/bin/bash
# Full correctness sweep for gpt-oss:20b - closes the last gap in Table 8.
#
# Background. gpt-oss:20b is one of the eight models in the throughput study (Table 7) but was
# absent from the correctness results (Table 8), because on 2026-07-29 every request failed
# under Ollama v0.32.3 with a "blk.0.ffn_down_exps.weight size overflow" error: an
# Ollama/llama.cpp defect handling the model's MXFP4 mixture-of-experts tensor layout. It was
# never a hardware limitation.
#
# Probed again 2026-08-19 under Ollama v0.32.6: the model now serves correctly. This runs the
# same three benchmarks as the other seven models so Table 8 can report 8 of 8.
#
# Settings match the corrected runs for the other models: the custom chat-aware HumanEval
# adapter, truthfulqa_gen, and a 16384-token budget. gpt-oss:20b reasons (277 thinking chars
# on a one-word probe), so the larger budget matters - at 4096 it would risk the same
# empty-generation fault that affected deepseek-r1:1.5b and qwen3:14b.
#
# EXPECT THIS TO BE SLOW. gpt-oss:20b exceeds the 12 GB VRAM budget and spills to CPU,
# measuring 17-19 tokens/s in Table 7 - the slowest model in the study. Estimate ~4h40m, but
# treat that as soft: a reasoning model on a CPU-spilled path may run considerably longer.

set -u
BASE="$HOME/lm-eval-env/full-results-corrected"
CUSTOM="$HOME/lm-eval-env/custom_tasks"
LOG="$BASE/rerun_progress.log"
M="gpt-oss:20b"
safe="gpt-oss_20b"

mkdir -p "$BASE"
export HF_ALLOW_CODE_EVAL=1

echo "=== GPT-OSS SWEEP START: $(date) ===" | tee -a "$LOG"
ollama --version | tee -a "$LOG"

for task in gsm8k humaneval_local_chat truthfulqa_gen; do
  echo "=== $M / $task === START $(date)" | tee -a "$LOG"
  "$HOME/lm-eval-env/Scripts/lm_eval.exe" run \
    --model local-chat-completions \
    --model_args "model=$M,base_url=http://localhost:11434/v1/chat/completions,num_concurrent=1,tokenized_requests=False" \
    --tasks "$task" \
    --include_path "$CUSTOM" \
    --apply_chat_template \
    --log_samples \
    --gen_kwargs max_tokens=16384 \
    --confirm_run_unsafe_code \
    --output_path "$BASE/" \
    > "$BASE/${safe}_${task}.log" 2>&1
  echo "=== $M / $task === EXIT $? at $(date)" | tee -a "$LOG"
done

echo "=== GPT-OSS SWEEP COMPLETE: $(date) ===" | tee -a "$LOG"

echo | tee -a "$LOG"
echo "Empty-generation count (fault B check):" | tee -a "$LOG"
"$HOME/lm-eval-env/Scripts/python.exe" - "$BASE" <<'PY' 2>&1 | tee -a "$LOG"
import glob, json, os, sys
base = sys.argv[1]
for f in sorted(glob.glob(os.path.join(base, "gpt-oss__20b", "samples_*.jsonl"))):
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    def txt(r):
        x = r.get("resps") or r.get("filtered_resps")
        while isinstance(x, list) and x:
            x = x[0]
        return x if isinstance(x, str) else ""
    empty = sum(1 for r in rows if not txt(r).strip())
    print(f"  {os.path.basename(f)[:44]:46} {empty:>5} / {len(rows)} empty")
PY
