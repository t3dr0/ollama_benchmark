#!/usr/bin/env python3
"""
csv_to_tex.py
═══════════════════════════════════════════════════════════════════════════════
Convert Ollama benchmark CSV output to LaTeX tables for academic publication.

Produces two .tex files:
  <stem>_summary.tex  —  abridged table for paper body   (6 selected prompts)
  <stem>_full.tex     —  complete table for appendix      (all prompts, landscape)

Usage
─────
  python csv_to_tex.py benchmark.csv
  python csv_to_tex.py benchmark.csv --output results
  python csv_to_tex.py benchmark.csv --summary-prompts P1 P2 P4 P5 P9_T5 P10

Required LaTeX packages (add to preamble)
──────────────────────────────────────────
  \\usepackage{booktabs}    % toprule / midrule / bottomrule
  \\usepackage{multirow}    % model name spanning two metric rows
  \\usepackage{pdflscape}   % landscape environment (full table only)
  \\usepackage{array}       % extended column specifiers

Reference
─────────
  Curington & Lano (2026) "Reusing Obsolete Windows 10 PCs for On-Premises
  Large Language Model Inference", Frontiers in Computer Science.
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not found.  Install with:  pip install pandas")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration — prompt order, headers, and metric labels
# ═══════════════════════════════════════════════════════════════════════════════

# Default prompts shown in the paper body summary table
DEFAULT_SUMMARY_PROMPTS = ["P1", "P2", "P4", "P5", "P9_T5", "P10"]

# Canonical order for the full appendix table
ALL_PROMPTS_ORDERED = [
    "P1", "P2", "P3", "P4", "P5", "P6",
    "P7", "P8", "P9_T1", "P9_T3", "P9_T5", "P10",
]

# LaTeX column header for each prompt ID
PROMPT_HEADER = {
    "P1":    r"$P_1$ (Fact)",
    "P2":    r"$P_2$ (Prose)",
    "P3":    r"$P_3$ (Arith.)",
    "P4":    r"$P_4$ (Code)",
    "P5":    r"$P_5$ (Context)",
    "P6":    r"$P_6$ (Constr.)",
    "P7":    r"$P_7$ (Logic)",
    "P8":    r"$P_8$ (Transl.)",
    "P9_T1": r"$P_9\text{-}T_1$",
    "P9_T3": r"$P_9\text{-}T_3$",
    "P9_T5": r"$P_9\text{-}T_5$ (T5)",
    "P10":   r"$P_{10}$ (Refusal)",
}

# Row metric labels (matching the paper template)
TPS_LABEL  = r"$\bar{X}_{\text{TPS}} \pm \sigma$"
ELAP_LABEL = r"$\bar{T}_{\text{Elap}} \pm \sigma$"

MISSING_CELL = r"\multicolumn{1}{c}{---}"


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV reading and metadata extraction
# ═══════════════════════════════════════════════════════════════════════════════

def read_csv(path: str) -> tuple:
    """
    Read benchmark CSV, skipping the comment header block (lines starting #).
    Returns (DataFrame, meta_dict).
    meta_dict keys: timestamp, hostname, gpu_info.
    """
    meta = {"timestamp": "", "hostname": "", "gpu_info": ""}

    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()

 # Extract metadata from comment lines
    for line in lines:
        clean = line.strip().strip('"')
        if not clean.startswith("#"):
            continue
            
        # Parse fields dynamically if they exist together on the same comment row
        if "Timestamp:" in clean:
            # Isolate the Timestamp token up until any trailing tabs/commas or other fields
            ts_part = clean.split("Timestamp:", 1)[-1]
            # If fields are tab/comma separated on one line, extract the clean timestamp segment
            meta["timestamp"] = ts_part.split("#")[0].split(",")[0].strip()
            
        if "Host:" in clean:
            host_part = clean.split("Host:", 1)[-1]
            meta["hostname"] = host_part.split("#")[0].split(",")[0].strip()
        elif "Host Node Name:" in clean:
            meta["hostname"] = clean.split("Host Node Name:", 1)[-1].strip()
            
        if "GPU:" in clean:
            gpu_part = clean.split("GPU:", 1)[-1]
            meta["gpu_info"] = gpu_part.split("#")[0].split(",")[0].strip()
        elif "Accelerator Info:" in clean:
            meta["gpu_info"] = clean.split("Accelerator Info:", 1)[-1].strip()

    # Find the row that contains the CSV header (starts with "Model")
    skip = 0
    for i, line in enumerate(lines):
        stripped = line.strip().strip('"')
        if stripped.startswith("Model,"):
            skip = i
            break

    df = pd.read_csv(path, skiprows=skip)
    # Strip whitespace from string columns
    for col in ["Model", "Size", "Prompt_ID", "Prompt_Name", "Metric_Type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df, meta


# ═══════════════════════════════════════════════════════════════════════════════
#  Data structuring
# ═══════════════════════════════════════════════════════════════════════════════

def build_pivot(df: pd.DataFrame) -> dict:
    """
    Build nested dict:  data[model][prompt_id][metric] = (mean, std_dev)
    metric key is 'tps' (tokens/sec) or 'elap' (elapsed seconds).
    """
    data: dict = {}
    for _, row in df.iterrows():
        model  = row["Model"]
        size   = row.get("Size", "---")
        pid    = row["Prompt_ID"]
        metric = row["Metric_Type"]
        try:
            mean = float(row["Mean"])
            std  = float(row["Std_Dev"])
        except (ValueError, KeyError):
            continue

        data.setdefault(model, {})["_size"] = size
        data.setdefault(model, {}).setdefault(pid, {})
        key = "tps" if metric == "Tokens_Per_Sec" else "elap"
        data[model][pid][key] = (mean, std)

    return data


def models_in_order(df: pd.DataFrame) -> list:
    """Return model names in the order they first appear in the CSV."""
    seen: list = []
    for m in df["Model"]:
        if m not in seen:
            seen.append(m)
    return seen


# ═══════════════════════════════════════════════════════════════════════════════
#  Formatting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_tps(mean: float, std: float) -> str:
    return rf"${mean:.1f} \pm {std:.1f}$"


def fmt_elap(mean: float, std: float) -> str:
    return rf"${mean:.1f} \pm {std:.1f}$"


def get_cell(data: dict, model: str, pid: str, key: str, fmt_fn) -> str:
    """Return formatted cell, or MISSING_CELL if data not available."""
    try:
        mean, std = data[model][pid][key]
        return fmt_fn(mean, std)
    except KeyError:
        return MISSING_CELL


def tex_escape(s: str) -> str:
    """Escape characters that are special in LaTeX text mode."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def model_label(name: str) -> str:
    """Format model name for LaTeX: monospaced bold, escaped."""
    return r"\textbf{\texttt{" + tex_escape(name) + r"}}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Row generation (shared by both tables)
# ═══════════════════════════════════════════════════════════════════════════════

def build_data_rows(data: dict, models: list, prompts: list) -> list:
    """
    Generate LaTeX row strings for all models.
    Each model produces two rows: TPS then elapsed, separated by \\midrule.
    Uses \\multirow{2}{*} for the model name column.
    """
    rows = []
    for i, model in enumerate(models):
        if i > 0:
            rows.append(r"    \midrule")

        size_str = tex_escape(data.get(model, {}).get("_size", "---"))

        # TPS row — model name and size span both metric rows via \multirow
        tps_cells = " & ".join(
            get_cell(data, model, p, "tps", fmt_tps) for p in prompts
        )
        rows.append(
            rf"    \multirow{{2}}{{*}}{{{model_label(model)}}} & "
            rf"\multirow{{2}}{{*}}{{{size_str}}} & "
            rf"{TPS_LABEL} & {tps_cells} \\"
        )

        # Elapsed row — model and size cells left blank (covered by \multirow)
        elap_cells = " & ".join(
            get_cell(data, model, p, "elap", fmt_elap) for p in prompts
        )
        rows.append(
            rf"    & & {ELAP_LABEL} & {elap_cells} \\"
        )

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary table (paper body)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary_table(
    data:    dict,
    models:  list,
    prompts: list,
    meta:    dict,
    label:   str = "tab:inference_benchmarks",
) -> str:
    """
    Abridged table for the paper body — selected prompts, full-width (table*).
    Matches the template style in the paper draft.
    """
    col_spec = "lll" + "c" * len(prompts)

    gpu  = tex_escape(meta.get("gpu_info",  ""))
    host = tex_escape(meta.get("hostname",  ""))
    ts   =            meta.get("timestamp", "")

    # Build column header line  (Model | Size | Metric | P1 | P2 | ...)
    header_cols = r"\textbf{Size} & \textbf{Metric} &" + "\n" + " &\n".join(
        r"\textbf{" + PROMPT_HEADER[p] + r"}" for p in prompts
    )

    rows = build_data_rows(data, models, prompts)

    parts = [
        "% ─────────────────────────────────────────────────────────────────────",
        "% Summary inference benchmark table  —  paper body",
        f"% Generated: {ts}  |  Host: {host}  |  GPU: {gpu}",
        "% Required packages: booktabs, multirow, graphicx",
        "% ─────────────────────────────────────────────────────────────────────",
        "",
        r"\begin{table*}[t]",
        r"\caption{Quantitative evaluation of local inference performance metrics"
        r" across diverse task dimensions ($N=5$)."
        r" Processing velocity ($\bar{X}_{\text{TPS}}$) is in tokens per second;"
        r" execution time ($\bar{T}_{\text{Elap}}$) in seconds."
        r" Statistical dispersion ($\sigma$) reflects variance across five"
        rf" independent runs on the evaluation host ({host};"
        rf" GPU: {gpu}).}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",   # tighten inter-column padding
        r"\resizebox{\textwidth}{!}{%",    # scale to fit text width exactly
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\textbf{Model Architecture} &", header_cols + r" \\",
        r"\midrule",
    ]

    parts.extend(rows)

    parts += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}% end resizebox",
        r"\end{table*}",
    ]

    return "\n".join(parts) + "\n"


# ═══════════════════════════════════════════════════════════════════════════════
#  Full appendix table (all prompts, landscape)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_full_table(
    data:    dict,
    models:  list,
    prompts: list,
    meta:    dict,
    label:   str = "tab:inference_benchmarks_full",
) -> str:
    """
    Complete appendix table — all prompts, landscape orientation.
    Requires \\usepackage{pdflscape} in the document preamble.
    """
    col_spec = "lll" + "c" * len(prompts)

    gpu  = tex_escape(meta.get("gpu_info",  ""))
    host = tex_escape(meta.get("hostname",  ""))
    ts   =            meta.get("timestamp", "")

    # Build column header line  (Model | Size | Metric | P1 | P2 | ...)
    header_cols = r"\textbf{Size} & \textbf{Metric} &" + "\n" + " &\n".join(
        r"\textbf{" + PROMPT_HEADER[p] + r"}" for p in prompts
    )

    rows = build_data_rows(data, models, prompts)

    parts = [
        "% ─────────────────────────────────────────────────────────────────────",
        "% Full inference benchmark table  —  appendix (all prompts, landscape)",
        f"% Generated: {ts}  |  Host: {host}  |  GPU: {gpu}",
        "% Required packages: booktabs, multirow, pdflscape",
        "% ─────────────────────────────────────────────────────────────────────",
        "",
        r"\begin{landscape}",
        r"\begin{table*}[p]",
        r"\caption{Complete inference benchmark results across all prompt dimensions"
        r" and model architectures ($N=5$). All ten prompt workloads ($P_1$--$P_{10}$)"
        r" are shown, including all three measured turns of the multi-turn dialogue ($P_9$)."
        r" $\bar{X}_{\text{TPS}}$: mean tokens per second;"
        r" $\bar{T}_{\text{Elap}}$: mean elapsed seconds;"
        r" $\sigma$: standard deviation across five independent runs."
        rf" Host: {host}; GPU: {gpu}.}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\scriptsize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\textbf{Model Architecture} &", header_cols + r" \\",
        r"\midrule",
    ]

    parts.extend(rows)

    parts += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        r"\end{landscape}",
    ]

    return "\n".join(parts) + "\n"


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert Ollama benchmark CSV to LaTeX tables for publication.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python csv_to_tex.py results.csv\n"
            "  python csv_to_tex.py results.csv --output paper_tables\n"
            "  python csv_to_tex.py results.csv --summary-prompts P1 P2 P4 P5 P9_T5 P10\n"
        ),
    )
    parser.add_argument("csv",
        help="Input CSV file (Ollama benchmark output)")
    parser.add_argument("--output", default=None, metavar="STEM",
        help="Output filename stem (default: input filename without extension)")
    parser.add_argument("--summary-prompts", nargs="+",
        default=DEFAULT_SUMMARY_PROMPTS, metavar="PID",
        help=(
            "Prompt IDs to include in the summary table "
            f"(default: {' '.join(DEFAULT_SUMMARY_PROMPTS)})"
        ))
    parser.add_argument("--summary-label", default="tab:inference_benchmarks",
        help="LaTeX \\label for summary table (default: tab:inference_benchmarks)")
    parser.add_argument("--full-label", default="tab:inference_benchmarks_full",
        help="LaTeX \\label for full appendix table")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: Input file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    stem         = args.output or csv_path.stem
    summary_path = Path(f"{stem}_summary.tex")
    full_path    = Path(f"{stem}_full.tex")

    # ── Read ──────────────────────────────────────────────────────────────────
    print(f"Reading: {csv_path}")
    df, meta = read_csv(str(csv_path))

    print(f"  Timestamp : {meta['timestamp']}")
    print(f"  Host      : {meta['hostname']}")
    print(f"  GPU       : {meta['gpu_info']}")
    print(f"  Models    : {df['Model'].nunique()} — {', '.join(df['Model'].unique())}")
    print(f"  Prompt IDs: {sorted(df['Prompt_ID'].unique().tolist())}")
    print(f"  Rows      : {len(df)}")

    # ── Build pivot ───────────────────────────────────────────────────────────
    data   = build_pivot(df)
    models = models_in_order(df)

    # Filter to prompts actually present in the data
    available       = set(df["Prompt_ID"].unique())
    summary_prompts = [p for p in args.summary_prompts  if p in available]
    full_prompts    = [p for p in ALL_PROMPTS_ORDERED   if p in available]

    skipped = [p for p in args.summary_prompts if p not in available]
    if skipped:
        print(f"\n  Warning: prompts absent from data, skipped from summary: {skipped}")

    # ── Generate ──────────────────────────────────────────────────────────────
    summary_tex = generate_summary_table(
        data, models, summary_prompts, meta, label=args.summary_label
    )
    full_tex = generate_full_table(
        data, models, full_prompts, meta, label=args.full_label
    )

    summary_path.write_text(summary_tex, encoding="utf-8")
    full_path.write_text(full_tex,    encoding="utf-8")

    print(f"\nOutput files:")
    print(f"  Summary table → {summary_path}   ({len(summary_prompts)} prompts)")
    print(f"  Full table    → {full_path}   ({len(full_prompts)} prompts, landscape)")
    print(f"\nLaTeX preamble requirements:")
    print(r"  \usepackage{booktabs}")
    print(r"  \usepackage{multirow}")
    print(r"  \usepackage{pdflscape}   % full/appendix table only")
    print(r"  \usepackage{array}")


if __name__ == "__main__":
    main()
