#!/usr/bin/env python3
"""
inspect_humaneval_samples.py — diagnose and correct HumanEval pass@1 measurement faults.

Answers the question a reviewer asks of any low pass@1 score: are these failures
*incorrect code*, or artefacts of the evaluation tooling?

Terminology, because two distinct layers are involved and conflating them misattributes
the fault:

  lm-eval-harness      EleutherAI's evaluation runner (third-party, cited as gao2024lmeval).
                       Not at fault.
  HumanEval dataset    The 164 problems, their function stubs, and their hidden unit tests
                       (openai/openai_humaneval). Not at fault.
  code_eval            HuggingFace `evaluate`'s pass@k implementation. Not at fault.
  OUR ADAPTER          `humaneval_local_chat` — the custom task written for this study that
                       builds the prompt and decides which text from the model's reply gets
                       executed. ~45 lines. THIS is where the faults are.

Three faults were found in our adapter (2026-08-17), all in how the model's reply is
turned into executable code:

  A  Wrong fenced block. The original extractor took the LAST fenced block. Models that
     write the solution first and a usage example second had their solution discarded.
  B  Empty generations. Reasoning models spend the whole token budget in the API's
     `reasoning` field and never emit `content`. NOT fixable by re-scoring — the affected
     items must be regenerated with a larger budget.
  C  Model self-tests. Models often append their own assertions to the same block. Those
     assertions are frequently wrong, and they execute before the hidden test, aborting
     the module. Standard HumanEval practice scores the function, not the model's test
     scaffolding, so these are stripped.

Usage
-----
    # classify failures; static only, executes nothing
    python inspect_humaneval_samples.py --triage <samples>.jsonl

    # definitive: re-extract with all three fixes and re-execute against the hidden tests
    python inspect_humaneval_samples.py --verify <samples>.jsonl --i-understand-code-execution

    # human-readable review file for a single model
    python inspect_humaneval_samples.py --triage <samples>.jsonl --report review.md --sample 10

--verify RUNS MODEL-GENERATED CODE. Run it from a scratch directory, using the lm-eval
virtualenv's interpreter (it needs `evaluate`).

Part of the replication package for Curington & Lano (2026). AGPLv3.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCED_RE = re.compile(r"```(?:python)?[ \t]*\n(.*?)```", re.DOTALL)
_DANGLING_FENCE_RE = re.compile(r"```(?:python)?[ \t]*\n(.*)", re.DOTALL)


# ---------------------------------------------------------------------------
# The original extractor, reproduced verbatim and left unfixed
#
# This is the code that produced the pass@1 figures in the submitted manuscript.
# It is kept so those figures stay reproducible and the fault stays inspectable.
# Do not "tidy" it.
# ---------------------------------------------------------------------------

def extract_code_original(resp: str) -> str:
    """As submitted. Selects the LAST complete fenced block (fault A)."""
    if not isinstance(resp, str):
        return ""
    text = _THINK_RE.sub("", resp)
    complete = _FENCED_RE.findall(text)
    if complete:
        return complete[-1]
    dangling = _DANGLING_FENCE_RE.search(text)
    if dangling:
        return dangling.group(1)
    return text


# ---------------------------------------------------------------------------
# Corrected extraction
# ---------------------------------------------------------------------------

def _defines(block: str, name: str) -> bool:
    """True if `block` defines a function or class called `name`.

    Falls back to a regex where the block does not parse, since a truncated
    generation still carries evidence of what the model was writing.
    """
    if not name:
        return False
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(name)}\s*\(", block, re.M) is not None
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name
        for n in ast.walk(tree)
    )


def _has_any_definition(block: str) -> bool:
    return re.search(r"^\s*(?:async\s+def|def|class)\s+\w+", block, re.M) is not None


def has_self_tests(block: str) -> bool:
    """True if the block carries top-level assertions or bare calls (fault C)."""
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return True
        if isinstance(node, ast.If):  # if __name__ == "__main__": ...
            return True
    return False


_KEEP = (
    ast.Import, ast.ImportFrom,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Assign, ast.AnnAssign,
)


def strip_self_tests(block: str) -> str:
    """Remove the model's own test scaffolding, keeping only the solution (fault C).

    Retains imports, definitions and module-level assignments. Drops top-level
    assertions, bare calls (`print(...)`), and `if __name__ == "__main__"` blocks —
    all of which execute before the hidden test and can abort the module.

    Slices the original source by line range rather than round-tripping through
    `ast.unparse`, so formatting and in-function comments survive intact.
    Blocks that do not parse are returned unchanged: they may be truncated, and
    guessing at them would do more harm than good.
    """
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return block

    keep = [n for n in tree.body if isinstance(n, _KEEP)]
    if not keep:
        return block

    lines = block.splitlines()
    out: list[str] = []
    for node in keep:
        start = node.lineno - 1
        for dec in getattr(node, "decorator_list", []):
            start = min(start, dec.lineno - 1)
        out.extend(lines[start:node.end_lineno])
    return "\n".join(out) + "\n"


def extract_code_corrected(resp: str, entry_point: str | None = None) -> str:
    """Corrected extraction: fixes fault A, then fault C.

    Block selection preference:
      1. the last block defining `entry_point` (the function under test);
      2. the last block containing any definition;
      3. the last block (original behaviour);
      4. a dangling/unclosed fence, then the raw text.
    """
    if not isinstance(resp, str):
        return ""
    text = _THINK_RE.sub("", resp)
    blocks = _FENCED_RE.findall(text)

    chosen: str | None = None
    if blocks:
        if entry_point:
            for block in reversed(blocks):
                if _defines(block, entry_point):
                    chosen = block
                    break
        if chosen is None:
            for block in reversed(blocks):
                if _has_any_definition(block):
                    chosen = block
                    break
        if chosen is None:
            chosen = blocks[-1]
    else:
        dangling = _DANGLING_FENCE_RE.search(text)
        chosen = dangling.group(1) if dangling else text

    return strip_self_tests(chosen)


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------

VERDICTS = {
    "EMPTY_GENERATION": "fault B: no content returned; needs regeneration, not re-scoring",
    "WRONG_BLOCK": "fault A: our extractor discarded the model's solution",
    "MODEL_SELF_TEST": "fault C: model's own (often incorrect) assertions aborted the module",
    "GENUINE_FAILURE": "solution extracted intact and unobstructed; the code itself is wrong",
}


def triage_one(row: dict[str, Any]) -> dict[str, Any]:
    doc = row["doc"]
    entry_point = doc.get("entry_point")
    raw = row["resps"][0][0]
    text = _THINK_RE.sub("", raw)

    original = extract_code_original(raw)
    corrected = extract_code_corrected(raw, entry_point)
    blocks = _FENCED_RE.findall(text)

    orig_defines = _defines(original, entry_point)

    if not raw.strip():
        verdict = "EMPTY_GENERATION"
    elif not orig_defines and _defines(corrected, entry_point):
        verdict = "WRONG_BLOCK"
    elif orig_defines and has_self_tests(original):
        verdict = "MODEL_SELF_TEST"
    elif not orig_defines:
        verdict = "EMPTY_GENERATION"
    else:
        verdict = "GENUINE_FAILURE"

    return {
        "doc_id": row["doc_id"],
        "task_id": doc.get("task_id", f"doc_{row['doc_id']}"),
        "entry_point": entry_point,
        "n_fenced_blocks": len(blocks),
        "original_extract": original,
        "corrected_extract": corrected,
        "executed": row["filtered_resps"][0],
        "hidden_test": doc.get("test", ""),
        "stub": doc.get("prompt", ""),
        "raw": raw,
        "extraction_changed": original.strip() != corrected.strip(),
        "verdict": verdict,
        "published_pass": row["pass@1"],
    }


def load_rows(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def model_name_from_path(path: str) -> str:
    return Path(path).parent.name.replace("__", ":")


# ---------------------------------------------------------------------------
# Verification — re-executes generated code against the hidden tests
# ---------------------------------------------------------------------------

def verify_all(rows: list[dict[str, Any]]) -> dict[int, bool]:
    """Re-extract every item with the corrected pipeline and execute the hidden tests.

    Every item is checked, not only the failures, so that any regression introduced
    by the fix is visible rather than hidden behind a net improvement.
    """
    try:
        import evaluate as hf_evaluate
    except ImportError:
        sys.exit(
            "--verify needs the `evaluate` package.\n"
            "Run it with the lm-eval virtualenv interpreter, e.g.\n"
            "  ~/lm-eval-env/Scripts/python.exe tooling/inspect_humaneval_samples.py --verify ..."
        )

    code_eval = hf_evaluate.load("code_eval")

    predictions, references, ids = [], [], []
    for row in rows:
        doc = row["doc"]
        corrected = extract_code_corrected(row["resps"][0][0], doc.get("entry_point"))
        predictions.append([doc["prompt"] + corrected])
        references.append(doc["test"] + f"\ncheck({doc['entry_point']})")
        ids.append(row["doc_id"])

    _, per_item = code_eval.compute(references=references, predictions=predictions, k=[1])

    passed: dict[int, bool] = {}
    for idx, results in per_item.items():
        passed[ids[idx]] = any(r[1]["passed"] for r in results)
    return passed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

HOW_TO_READ = """## How to read this file

**What the benchmark does.** HumanEval gives the model a Python function *stub* — a signature and
a docstring, with no body — and asks it to write the body. Unit tests that the model never sees are
then executed against the result. The model writes code; it is never asked to check existing code.

**Whose text is whose.**

| Block | Written by | Notes |
|---|---|---|
| *The hidden test* | The HumanEval dataset | Never shown to the model. This alone decides correctness. |
| *Full model response* | The model, entirely | Prose and code exactly as returned. |
| *What the original run executed* | **Two sources spliced together** | Opens with the dataset's stub; everything after the docstring is the model's text, as picked by our original extractor. |
| *What corrected extraction produces* | The model | Same response, correct block, self-tests stripped. |

**Ignore what the model says about itself.** Lines like "This solution passes the given test case",
predicted `# Output:` comments, and the model's own `assert` statements are unverified claims — the
model executed nothing. None of it is used for scoring, and it is frequently wrong.

**The one question you are answering:** did our tooling prevent a genuine attempt from being tested
fairly? Not "is this code correct" — the hidden test settles that, and where `--verify` has been run
its real verdict is shown per item.

| Verdict | Means | Counts against the model? |
|---|---|---|
| `WRONG_BLOCK` | A real solution is in the response but our extractor took a different block. | No — our fault. |
| `MODEL_SELF_TEST` | The model appended its own assertions; they ran first and aborted the module. | No — our fault, standard practice is to strip them. |
| `EMPTY_GENERATION` | No content returned; the token budget was consumed by hidden reasoning. | No — needs regeneration. |
| `GENUINE_FAILURE` | The model's real attempt was executed unobstructed and failed. | Yes. |

**What to do:** read each item, tick the box if you agree with the verdict, or note a correction.
The resulting agreement rate is the figure reported to the reviewer.
"""


def _quote(text: str) -> str:
    return "\n".join("> " + line if line else ">" for line in text.strip().splitlines())


def render_report(model: str, triaged: list[dict[str, Any]], sample_n: int, seed: int,
                  verified: dict[int, bool] | None) -> str:
    counts: dict[str, int] = {}
    for t in triaged:
        counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1

    out: list[str] = [f"# HumanEval failure review — `{model}`\n"]
    out.append(
        "Generated by `tooling/inspect_humaneval_samples.py`. The faults described here are in "
        "**our own task adapter** (`humaneval_local_chat`), not in EleutherAI's lm-eval-harness, "
        "the HumanEval dataset, or the `code_eval` scorer.\n"
    )
    out.append(HOW_TO_READ)

    out.append("\n## Triage of all failures\n")
    out.append("| Verdict | Count | Meaning |")
    out.append("|---|---:|---|")
    for verdict, meaning in VERDICTS.items():
        out.append(f"| `{verdict}` | {counts.get(verdict, 0)} | {meaning} |")
    out.append(f"\n**Total failures: {len(triaged)}**\n")

    if verified is not None:
        recovered = sum(1 for t in triaged if verified.get(t["doc_id"]))
        out.append(
            f"Re-executed against the hidden tests after correction: **{recovered} of "
            f"{len(triaged)} previously-failing items now pass.**\n"
        )

    rng = random.Random(seed)
    by_verdict: dict[str, list[dict[str, Any]]] = {}
    for t in triaged:
        by_verdict.setdefault(t["verdict"], []).append(t)

    chosen: list[dict[str, Any]] = []
    for items in by_verdict.values():
        share = max(1, round(sample_n * len(items) / max(len(triaged), 1)))
        chosen.extend(rng.sample(items, min(share, len(items))))
    chosen.sort(key=lambda t: t["doc_id"])

    out.append(f"## Review sample ({len(chosen)} of {len(triaged)} failures, seed={seed})\n")
    for t in chosen:
        out.append(f"### {t['task_id']} — verdict: **{t['verdict']}**\n")
        if verified is not None:
            now = verified.get(t["doc_id"])
            out.append(f"**Hidden test after correction: {'PASS' if now else 'STILL FAILS'}**\n")
        out.append(f"- Function under test: `{t['entry_point']}`")
        out.append(f"- Fenced code blocks in the response: {t['n_fenced_blocks']}\n")

        out.append("<details><summary>The hidden test (from the dataset — the model never saw this)</summary>\n")
        out.append("```python")
        out.append(t["hidden_test"].strip())
        out.append("```\n</details>\n")

        out.append("<details><summary>Full model response (everything below is the model's own text)</summary>\n")
        out.append(_quote(t["raw"]) if t["raw"].strip() else "> *(empty — the model returned no content)*")
        out.append("\n</details>\n")

        out.append("**What the original run executed** — dataset stub, then our extractor's pick:\n")
        out.append("```python")
        executed = t["executed"]
        out.append((executed[0] if isinstance(executed, list) else executed).strip() or "(nothing)")
        out.append("```\n")

        if t["extraction_changed"]:
            out.append("**What corrected extraction produces instead:**\n")
            out.append("```python")
            out.append(t["corrected_extract"].strip() or "(nothing — needs regeneration)")
            out.append("```\n")

        out.append("- [ ] I agree with this verdict")
        out.append("- Correction, if any:\n")
        out.append("---\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", help="samples_humaneval_local_chat_*.jsonl (globs allowed)")
    ap.add_argument("--triage", action="store_true", help="static classification; executes nothing")
    ap.add_argument("--verify", action="store_true", help="re-extract and re-execute against the hidden tests")
    ap.add_argument("--i-understand-code-execution", action="store_true",
                    help="required acknowledgement for --verify")
    ap.add_argument("--report", metavar="FILE.md", help="write a human-readable review file")
    ap.add_argument("--sample", type=int, default=12, help="failures to include in the review sample")
    ap.add_argument("--seed", type=int, default=20260817, help="sampling seed, for a reproducible selection")
    ap.add_argument("--json-out", metavar="FILE.json", help="write the corrected results as JSON")
    args = ap.parse_args()

    if not (args.triage or args.verify):
        ap.error("choose --triage and/or --verify")
    if args.verify and not args.i_understand_code_execution:
        ap.error("--verify executes model-generated code; pass --i-understand-code-execution")
    if args.verify:
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

    paths: list[str] = []
    for pattern in args.samples:
        paths.extend(sorted(glob.glob(pattern)) or ([pattern] if Path(pattern).exists() else []))
    if not paths:
        sys.exit("no sample files matched")

    summary = []
    for path in paths:
        model = model_name_from_path(path)
        rows = load_rows(path)
        failures = [r for r in rows if r.get("pass@1") == 0.0]
        triaged = [triage_one(r) for r in failures]

        counts: dict[str, int] = {}
        for t in triaged:
            counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1

        published = sum(r["pass@1"] for r in rows) / len(rows) * 100
        verified = None
        corrected_rate = regressions = recovered = None

        if args.verify:
            print(f"verifying {model} ({len(rows)} items) ...", flush=True)
            verified = verify_all(rows)
            corrected_rate = sum(verified.values()) / len(rows) * 100
            recovered = sum(1 for r in rows if r["pass@1"] == 0.0 and verified.get(r["doc_id"]))
            regressions = sum(1 for r in rows if r["pass@1"] == 1.0 and not verified.get(r["doc_id"]))

        summary.append((model, len(rows), published, corrected_rate, counts, recovered, regressions))

        if args.report:
            stem = Path(args.report)
            out = stem if len(paths) == 1 else stem.with_name(
                f"{stem.stem}-{model.replace(':', '_')}{stem.suffix}")
            out.write_text(render_report(model, triaged, args.sample, args.seed, verified), encoding="utf-8")
            print(f"wrote {out}")

    print()
    head = f"{'model':<22}{'published':>11}"
    if args.verify:
        head += f"{'corrected':>11}{'recovered':>11}{'regressed':>11}"
    head += "".join(f"{v[:11]:>13}" for v in VERDICTS)
    print(head)
    print("-" * len(head))
    for model, n, pub, corr, counts, rec, reg in summary:
        line = f"{model:<22}{pub:>10.1f}%"
        if args.verify:
            line += f"{corr:>10.1f}%{rec:>11}{reg:>11}"
        line += "".join(f"{counts.get(v, 0):>13}" for v in VERDICTS)
        print(line)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps([
            {"model": m, "n": n, "published_pass@1": pub, "corrected_pass@1": corr,
             "recovered": rec, "regressed": reg, "triage": c}
            for m, n, pub, corr, c, rec, reg in summary
        ], indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
