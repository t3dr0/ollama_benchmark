"""Code extraction and pass@k scoring for the `humaneval_local_chat` task.

This task exists because neither built-in lm-eval HumanEval variant works against an
OpenAI-style chat-completions backend: the plain task's stop sequences assume a
raw-completion model, and the "instruct" variant relies on an assistant-turn prefill that
this backend silently ignores.

CHANGE HISTORY
--------------
2026-07-31  First version. Used for the pass@1 figures in the submitted manuscript.
            Preserved verbatim as `utils.py.ORIGINAL-as-submitted` so those figures stay
            reproducible.

2026-08-18  Three faults corrected after inspecting the retained per-sample logs, in
            response to Reviewer 2's request to confirm that low pass@1 scores reflect
            incorrect code rather than extraction failures. They did not.

  A  Wrong fenced block. `extract_code` returned the LAST fenced block. Models that write
     the solution first and a usage example second had the solution discarded and the
     example scored. Affected 129/145 of mistral-nemo:12b's failures, 61/84 of
     granite4.1:8b's, 53/89 of qwen3:14b's.
     Fix: prefer the last block that actually defines the task's entry point.

  C  Model self-tests executed. Models frequently append their own assertions to the same
     block. Those assertions are often wrong and execute before the hidden test, aborting
     the module. Verified case: mistral-nemo:12b on HumanEval/24 wrote a correct
     `largest_divisor` that passes all five hidden assertions, then added
     `assert largest_divisor(28) == 7` (actually 14). Affected 34/37 of phi4:14b's failures.
     Fix: keep imports, definitions and module-level assignments; drop top-level
     assertions, bare calls and `if __name__ == "__main__"` blocks. This matches standard
     HumanEval practice, which scores the completion rather than the model's scaffolding.

  B  Empty generations (NOT fixable here). Reasoning models spend the whole token budget in
     the API's `reasoning` field and never emit `content`. Requires regeneration at a larger
     budget; see rerun_humaneval_faultB.sh. Note that calibration showed the behaviour is
     bimodal rather than budget-limited: raising 4096 -> 16384 recovered 2 of 5 sampled
     deepseek-r1:1.5b items and left 3 producing ~65,000 characters of reasoning with no
     answer.

Re-scoring the saved generations with A and C corrected changed five of seven models'
scores and produced zero regressions across all 1,148 items.
"""

import ast
import re

import evaluate as hf_evaluate

# Copied from lm_eval/tasks/humaneval/utils.py (lm_eval==0.4.12) so this task has no
# runtime dependency on the installed package's internal file layout.
compute_ = hf_evaluate.load("code_eval")
compute_.compute(
    references=["assert add(2, 3)==5"],
    predictions=[["def add(a,b): return a*b"]],
    k=[1],
)


def pass_at_k(references, predictions, k=None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    return compute_.compute(references=references, predictions=predictions, k=k)[0]


_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCED_RE = re.compile(r"```(?:python)?[ \t]*\n(.*?)```", re.DOTALL)
_DANGLING_FENCE_RE = re.compile(r"```(?:python)?[ \t]*\n(.*)", re.DOTALL)

# Node types kept when stripping the model's own test scaffolding (fault C).
_KEEP = (
    ast.Import, ast.ImportFrom,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Assign, ast.AnnAssign,
)


def _defines(block, name):
    """True if `block` defines a function or class called `name`.

    Falls back to a regex where the block does not parse: a truncated generation still
    carries evidence of what the model was writing.
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


def _has_any_definition(block):
    return re.search(r"^\s*(?:async\s+def|def|class)\s+\w+", block, re.M) is not None


def strip_self_tests(block):
    """Drop the model's own test scaffolding, keeping the solution (fault C).

    Slices the original source by line range rather than round-tripping through
    `ast.unparse`, so formatting and in-function comments survive. Blocks that do not
    parse are returned unchanged: they may be truncated, and guessing would do more harm
    than good.
    """
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return block
    keep = [n for n in tree.body if isinstance(n, _KEEP)]
    if not keep:
        return block
    lines = block.splitlines()
    out = []
    for node in keep:
        start = node.lineno - 1
        for dec in getattr(node, "decorator_list", []):
            start = min(start, dec.lineno - 1)
        out.extend(lines[start:node.end_lineno])
    return "\n".join(out) + "\n"


def extract_code(resp, entry_point=None):
    """Pull the model's solution out of a chat reply. Corrects faults A then C.

    Block preference: the last block defining `entry_point`; else the last block with any
    definition; else the last block; else a dangling fence, then the raw text.
    """
    if not isinstance(resp, str):
        return ""
    text = _THINK_RE.sub("", resp)  # never mine code out of reasoning traces
    blocks = _FENCED_RE.findall(text)

    chosen = None
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


def build_predictions_chat_regex(resps, docs):
    # The dataset stub is prepended so that completions which omit the signature still
    # execute. Where the model re-emits the full function, its definition simply shadows
    # the stub's empty one, which is harmless.
    return [
        [doc["prompt"] + extract_code(r, doc.get("entry_point")) for r in resp]
        for resp, doc in zip(resps, docs)
    ]
