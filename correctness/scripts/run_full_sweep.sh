#!/bin/bash
# Full qualitative benchmark sweep: 7 working models x 3 tasks (gsm8k, humaneval, truthfulqa_gen)
# gpt-oss:20b excluded - broken on this Ollama install (tensor size overflow error), tracked separately.
export HF_ALLOW_CODE_EVAL=1

MODELS="deepseek-r1:1.5b qwen3:14b gemma4:e4b gemma3:12b granite4.1:8b phi4:14b mistral-nemo:12b"
TASKS="gsm8k humaneval truthfulqa_gen"
BASE="$HOME/lm-eval-env/full-results"
mkdir -p "$BASE"

echo "SWEEP START: $(date)" > "$BASE/sweep_progress.log"

for m in $MODELS; do
  safe=$(echo "$m" | tr ':' '_' | tr '.' '_')
  for t in $TASKS; do
    echo "=== $m / $t === START $(date)" >> "$BASE/sweep_progress.log"
    extra_flags=""
    if [ "$t" = "humaneval" ]; then
      extra_flags="--confirm_run_unsafe_code"
    fi
    "$HOME/lm-eval-env/Scripts/lm_eval.exe" run --model local-chat-completions \
      --model_args model=$m,base_url=http://localhost:11434/v1/chat/completions,num_concurrent=1,tokenized_requests=False \
      --tasks $t --apply_chat_template --log_samples \
      --gen_kwargs max_tokens=4096 \
      $extra_flags \
      --output_path "$BASE/" > "$BASE/${safe}_${t}.log" 2>&1
    echo "=== $m / $t === EXIT $? at $(date)" >> "$BASE/sweep_progress.log"
  done
done

echo "SWEEP COMPLETE: $(date)" >> "$BASE/sweep_progress.log"
