"""Measure generated-token counts for P1 and P5 (Reviewer 2, comment 4).

The reviewer asked for generated-token counts for P1 and P5. These were never persisted:
ollama_benchmark.py computes eval_count but writes only tokens/s and elapsed seconds. They
cannot be recovered from the archived CSVs, and deriving them as TPS x elapsed overestimates
P5 badly, because prompt processing of its ~2,574-word payload is a large share of elapsed
time - exactly the comparison at issue.

Token counts depend on the model, the prompt and the decoding settings, NOT on the GPU, so
they are measurable on current hardware under the same argument Section 4.3.3 already makes
for the correctness axis. No timing claim is made here.

Settings deliberately replicate the original throughput benchmark: the native /api/chat
endpoint with NO options payload, so every decoding parameter stays at each model's packaged
default - including reasoning behaviour, which is the point at issue.
"""
import importlib.util, json, os, statistics, sys, time, urllib.request

N = 3
TIMEOUT = 900
# Hard stop. Checked before every request; set 15 min before the true cutoff so that even a
# request that runs the full TIMEOUT cannot overshoot it. Author instruction 2026-08-19.
DEADLINE = time.strptime('19:45', '%H:%M')
MODELS = ["gemma4:e4b", "gemma3:12b", "qwen3:14b", "granite4.1:8b",
          "phi4:14b", "deepseek-r1:1.5b", "mistral-nemo:12b"]

_BENCHMARK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "..", "ollama_benchmark.py")
spec = importlib.util.spec_from_file_location("ob", os.path.normpath(_BENCHMARK))
ob = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(ob)
except Exception:
    pass

P1 = "Name the SI unit of electrical resistance and state its symbol."
P5 = ("Summarise the following text in five bullet points, each no longer than 30 words. "
      "\n\n" + ob.generate_long_text())


def ask(model, prompt):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "stream": False}).encode()          # no options: defaults apply
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())
    return {"eval_count": d.get("eval_count", 0),
            "prompt_eval_count": d.get("prompt_eval_count", 0),
            "eval_duration_s": d.get("eval_duration", 0) / 1e9,
            "total_duration_s": d.get("total_duration", 0) / 1e9,
            "wall_s": round(time.time() - t0, 1),
            "content_chars": len(d.get("message", {}).get("content", "") or ""),
            "thinking_chars": len(d.get("message", {}).get("thinking", "") or "")}


def past_deadline():
    now = time.localtime()
    return (now.tm_hour, now.tm_min) >= (DEADLINE.tm_hour, DEADLINE.tm_min)


def save(results):
    json.dump(results, open("p1_p5_token_counts.json", "w"), indent=2)


results = {}
stopped_early = []
print(f"{'model':22}{'prompt':>7}{'gen tokens':>12}{'prompt tok':>12}{'think ch':>10}{'wall s':>9}")
print("-" * 72)
for m in MODELS:
    for pid, ptext in (("P1", P1), ("P5", P5)):
        if past_deadline():
            stopped_early.append(f'{m}|{pid}')
            continue
        runs = []
        for _ in range(N):
            try:
                runs.append(ask(m, ptext))
            except Exception as e:
                print(f"{m:22}{pid:>7}  ERROR {type(e).__name__}", flush=True)
                break
        if not runs:
            continue
        gen = [r["eval_count"] for r in runs]
        results[f"{m}|{pid}"] = {
            "gen_tokens_mean": statistics.mean(gen),
            "gen_tokens_runs": gen,
            "prompt_tokens": runs[0]["prompt_eval_count"],
            "thinking_chars_mean": statistics.mean(r["thinking_chars"] for r in runs),
            "wall_s_mean": statistics.mean(r["wall_s"] for r in runs)}
        save(results)   # incremental: a kill or deadline stop still leaves usable data
        print(f'{m:22}{pid:>7}{statistics.mean(gen):>12.0f}'
              f'{runs[0]["prompt_eval_count"]:>12}'
              f'{statistics.mean(r["thinking_chars"] for r in runs):>10.0f}'
              f'{statistics.mean(r["wall_s"] for r in runs):>9.1f}', flush=True)

json.dump(results, open("p1_p5_token_counts.json", "w"), indent=2)
print("\nwrote p1_p5_token_counts.json")
print(f"N={N} per cell; no options payload sent, so all decoding parameters are model defaults.")
