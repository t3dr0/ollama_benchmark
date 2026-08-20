"""Single-request probe: can gpt-oss:20b be served by this Ollama installation?

Background. gpt-oss:20b is one of the eight models in the throughput study (Table 7) but is
absent from the correctness results (Table 8), because on 2026-07-29 every request to it
failed on this machine under Ollama v0.32.3 with:

    {"error":{"message":"tensor \"blk.0.ffn_down_exps.weight\" size overflow"}}

an Ollama/llama.cpp-side defect handling this model's MXFP4 mixture-of-experts tensor layout.
It is unrelated to lm-evaluation-harness, to the custom task, and to the GPU.

The machine now runs v0.32.6. This asks for a one-word reply and reports what happens. It is
deliberately not an evaluation: if the model answers, the full correctness sweep becomes worth
running; if it fails again, the failure is recorded against a second Ollama version, which is
better evidence for the exclusion than the manuscript's current wording.
"""
import json
import urllib.error
import urllib.request

MODEL = "gpt-oss:20b"
URL = "http://127.0.0.1:11434/api/chat"

body = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
    "stream": False,
}).encode()

req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})

try:
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    msg = d.get("message", {}) or {}
    content = (msg.get("content") or "").strip()
    print("RESULT: SERVED OK")
    print("  content        :", repr(content[:200]))
    print("  eval_count     :", d.get("eval_count"))
    print("  thinking chars :", len(msg.get("thinking") or ""))
    print()
    print("  -> gpt-oss:20b can now be served. Running the full correctness sweep would")
    print("     close the remaining gap in Table 8 (currently 7 of 8 models).")
except urllib.error.HTTPError as e:
    detail = e.read().decode("utf-8", "ignore")[:500]
    print("RESULT: STILL FAILING - HTTP", e.code)
    print("  body:", detail)
    print()
    print("  -> failure now recorded against a second Ollama version; cite this in")
    print("     Section 4.3.3 as the reason for excluding the model.")
except Exception as e:
    print("RESULT: STILL FAILING -", type(e).__name__, str(e)[:300])
    print()
    print("  -> failure now recorded against a second Ollama version; cite this in")
    print("     Section 4.3.3 as the reason for excluding the model.")
