#!/bin/bash
# Complete the Q4_K_M control for mistral-nemo:12b (Reviewer 2, comment 3).
#
# HumanEval already run 2026-08-18 (65.2%). This adds GSM8K and TruthfulQA so the control
# row in Table 8 is complete rather than carrying blanks.
#
# Settings deliberately MATCH the original 2026-07-30 Q4_0 run so the comparison is
# like-for-like: task truthfulqa_gen (not the _chat variant, which only the reasoning models
# required) and max_tokens=4096. mistral-nemo produced zero empty generations on any
# benchmark, so the budget choice is immaterial - matching simply removes a confound.

set -u
BASE="$HOME/lm-eval-env/full-results-corrected"
LOG="$BASE/rerun_progress.log"
M="mistral-nemo:12b-instruct-2407-q4_K_M"
safe="mistral-nemo_12b-instruct-2407-q4_K_M"

export HF_ALLOW_CODE_EVAL=1
echo "=== Q4_K_M CONTROL START: $(date) ===" | tee -a "$LOG"

for task in gsm8k truthfulqa_gen; do
  echo "=== $M / $task === START $(date)" | tee -a "$LOG"
  "$HOME/lm-eval-env/Scripts/lm_eval.exe" run \
    --model local-chat-completions \
    --model_args "model=$M,base_url=http://localhost:11434/v1/chat/completions,num_concurrent=1,tokenized_requests=False" \
    --tasks "$task" \
    --apply_chat_template \
    --log_samples \
    --gen_kwargs max_tokens=4096 \
    --output_path "$BASE/" \
    > "$BASE/${safe}_${task}.log" 2>&1
  echo "=== $M / $task === EXIT $? at $(date)" | tee -a "$LOG"
done

echo "=== Q4_K_M CONTROL COMPLETE: $(date) ===" | tee -a "$LOG"
