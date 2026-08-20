#!/bin/bash
# Regenerate HumanEval for the models affected by fault B (empty generations).
#
# Fault B: reasoning models consume the whole generation budget in the API's `reasoning`
# field and never emit `content`, so lm-eval records an empty string and scores zero.
# This cannot be repaired by re-scoring saved generations — the items must be regenerated
# with a larger budget.
#
# Faults A (wrong fenced block) and C (model self-tests executed) ARE repaired by
# re-scoring, and have already been applied to the other four models. This script covers
# only the three models with unanswered items.
#
# Prerequisites:
#   - OpenWebUI stopped (it holds models in VRAM and competes for the GPU)
#   - custom_tasks/humaneval_local_chat/utils.py updated with the corrected extractor
#   - original results already archived to full-results-ORIGINAL-2026-07-31/
#
# Writes to a NEW output directory; the original run is never overwritten.

set -u

BUDGET="${BUDGET:-16384}"
MODELS="${MODELS:-deepseek-r1:1.5b qwen3:14b gemma4:e4b}"

BASE="$HOME/lm-eval-env/full-results-corrected"
CUSTOM="$HOME/lm-eval-env/custom_tasks"
LOG="$BASE/rerun_progress.log"

mkdir -p "$BASE"
export HF_ALLOW_CODE_EVAL=1

echo "=== FAULT-B REGENERATION START: $(date) ===" | tee -a "$LOG"
echo "budget=$BUDGET  models=$MODELS" | tee -a "$LOG"

for m in $MODELS; do
  safe=$(echo "$m" | tr ':' '_' | tr '.' '_')
  echo "=== $m / humaneval_local_chat === START $(date)" | tee -a "$LOG"

  "$HOME/lm-eval-env/Scripts/lm_eval.exe" run \
    --model local-chat-completions \
    --model_args "model=$m,base_url=http://localhost:11434/v1/chat/completions,num_concurrent=1,tokenized_requests=False" \
    --tasks humaneval_local_chat \
    --include_path "$CUSTOM" \
    --apply_chat_template \
    --log_samples \
    --gen_kwargs "max_tokens=$BUDGET" \
    --confirm_run_unsafe_code \
    --output_path "$BASE/" \
    > "$BASE/${safe}_humaneval_local_chat.log" 2>&1

  echo "=== $m / humaneval_local_chat === EXIT $? at $(date)" | tee -a "$LOG"
done

echo "=== FAULT-B REGENERATION COMPLETE: $(date) ===" | tee -a "$LOG"

# Post-run: how many items still returned nothing at the larger budget?
echo | tee -a "$LOG"
echo "Empty-generation count after regeneration:" | tee -a "$LOG"
"$HOME/lm-eval-env/Scripts/python.exe" - "$BASE" <<'PY' 2>&1 | tee -a "$LOG"
import glob, json, os, sys
base = sys.argv[1]
for f in sorted(glob.glob(os.path.join(base, "*", "samples_humaneval_local_chat_*.jsonl"))):
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    empty = sum(1 for r in rows if not r["resps"][0][0].strip())
    model = os.path.basename(os.path.dirname(f)).replace("__", ":")
    print(f"  {model:<22} {empty:>4} / {len(rows)} still empty")
PY
