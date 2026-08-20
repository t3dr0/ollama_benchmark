#!/bin/bash
export HF_ALLOW_CODE_EVAL=1
BASE="$HOME/lm-eval-env/full-results"
LOG="$BASE/sweep_progress.log"
CUSTOM="$HOME/lm-eval-env/custom_tasks"

echo "RERUN-FIXED-BENCHMARKS START: $(date)" >> "$LOG"

# --- TruthfulQA rerun (fixed until stop-sequence) for the 3 affected models ---
TQ_MODELS="deepseek-r1:1.5b qwen3:14b gemma4:e4b"
for m in $TQ_MODELS; do
  safe=$(echo "$m" | tr ':' '_' | tr '.' '_')
  echo "=== $m / truthfulqa_gen_chat (rerun, fixed) === START $(date)" >> "$LOG"
  "$HOME/lm-eval-env/Scripts/lm_eval.exe" run --model local-chat-completions \
    --model_args model=$m,base_url=http://localhost:11434/v1/chat/completions,num_concurrent=1,tokenized_requests=False \
    --tasks truthfulqa_gen_chat --include_path "$CUSTOM" \
    --apply_chat_template --log_samples \
    --gen_kwargs max_tokens=4096 \
    --output_path "$BASE/" > "$BASE/${safe}_truthfulqa_gen_chat.log" 2>&1
  echo "=== $m / truthfulqa_gen_chat (rerun, fixed) === EXIT $? at $(date)" >> "$LOG"
done

# --- HumanEval rerun (fixed custom chat-aware task) for all 7 models ---
ALL_MODELS="deepseek-r1:1.5b qwen3:14b gemma4:e4b gemma3:12b granite4.1:8b phi4:14b mistral-nemo:12b"
for m in $ALL_MODELS; do
  safe=$(echo "$m" | tr ':' '_' | tr '.' '_')
  echo "=== $m / humaneval_local_chat (rerun, fixed) === START $(date)" >> "$LOG"
  "$HOME/lm-eval-env/Scripts/lm_eval.exe" run --model local-chat-completions \
    --model_args model=$m,base_url=http://localhost:11434/v1/chat/completions,num_concurrent=1,tokenized_requests=False \
    --tasks humaneval_local_chat --include_path "$CUSTOM" \
    --apply_chat_template --log_samples \
    --gen_kwargs max_tokens=4096 \
    --confirm_run_unsafe_code \
    --output_path "$BASE/" > "$BASE/${safe}_humaneval_local_chat.log" 2>&1
  echo "=== $m / humaneval_local_chat (rerun, fixed) === EXIT $? at $(date)" >> "$LOG"
done

echo "RERUN-FIXED-BENCHMARKS COMPLETE: $(date)" >> "$LOG"
