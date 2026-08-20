#!/usr/bin/env python3
"""
csv_to_heatmap.py
===============================================================================
Generates high-density publication heatmaps from Ollama benchmark CSV files.
Supports structural profiling across two separate operational modes:
  1. 'tps' : Mean Tokens Per Second (reveals raw throughput cliffs).
  2. 'cov' : Coefficient of Variation (reveals architectural stability metrics).
  
Research Reference:
    Implementation supporting the project and paper by I. Curington and K. Lano (2026):
    "Reusing Obsolete Windows 10 PCs for On-Premises Large Language Model Inference"

Version: 0.3.0
Author: Ian Curington
License: AGPLv3
===============================================================================
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Academic styling preset adjustments
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

def parse_csv_data(csv_path):
    """Parses custom metadata comments and extracts structured evaluation metrics."""
    meta = {"timestamp": "Unknown", "hostname": "Unknown", "gpu_info": "Unknown", "duration": "Unknown"}
    skip_rows = 0
    
    # Read file lines to extract metadata blocks
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        clean = line.strip().strip('"')
        if not clean.startswith("#"):
            if clean.startswith("Model,"):
                skip_rows = i
            continue
        if "Timestamp:" in clean:
            meta["timestamp"] = clean.split("Timestamp:", 1)[-1].split("#")[0].split(",")[0].strip()
        if "Host:" in clean:
            meta["hostname"] = clean.split("Host:", 1)[-1].split("#")[0].split(",")[0].strip()
        elif "Host Node Name:" in clean:
            meta["hostname"] = clean.split("Host Node Name:", 1)[-1].strip()
        if "GPU:" in clean:
            meta["gpu_info"] = clean.split("GPU:", 1)[-1].split("#")[0].split(",")[0].strip()
        elif "Accelerator Info:" in clean:
            meta["gpu_info"] = clean.split("Accelerator Info:", 1)[-1].strip()
        if "Duration:" in clean:
            meta["duration"] = clean.split("Duration:", 1)[-1].strip()

    # Load into dataframe
    df = pd.read_csv(csv_path, skiprows=skip_rows)
    for col in ["Model", "Size", "Prompt_ID", "Metric_Type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    return df, meta

def generate_heatmap(df, meta, mode='tps', output_path=None):
    """Pivots data structure and generates matrix color maps."""
    # Filter for standard prompt identifiers to align rows
    df_filtered = df[df['Prompt_ID'].str.match(r'^P\d+(_T\d+)?$')].copy()
    
    # Extract unique model size associations to append to labels
    size_map = df_filtered.groupby('Model')['Size'].first().to_dict()
    
    # Split tokens per second records
    tps_df = df_filtered[df_filtered['Metric_Type'] == 'Tokens_Per_Sec'].copy()
    
    # Construct pivot tables
    mean_pivot = tps_df.pivot(index='Model', columns='Prompt_ID', values='Mean')
    std_pivot = tps_df.pivot(index='Model', columns='Prompt_ID', values='Std_Dev')
    
    # Clean up column ordering (P1, P2, ..., P9_T1, P10)
    def sort_key(col):
        if '_T' in col:
            base, turn = col.split('_T')
            return (int(base[1:]), int(turn))
        return (int(col[1:]), 0)
    
    ordered_columns = sorted(mean_pivot.columns, key=sort_key)
    mean_pivot = mean_pivot[ordered_columns]
    std_pivot = std_pivot[ordered_columns]
    
    # Remap model names to include footprint sizes on Y axis
    new_index = [f"{model} ({size_map.get(model, '---')})" for model in mean_pivot.index]
    mean_pivot.index = new_index
    std_pivot.index = new_index

    # Configure operational modes
    if mode == 'tps':
        plot_data = mean_pivot
        cmap = 'plasma'
        cbar_label = 'Mean Inference Velocity (Tokens/Sec)'
        title = f"Ollama Local Inference Throughput Profiles (VRAM Baseline: 12GB)\nHost: {meta['hostname']} | Hardware: {meta['gpu_info']}"
        fmt = ".1f"
    elif mode == 'cov':
        # Calculate Coefficient of Variation: (Sigma / Mean)
        plot_data = std_pivot / mean_pivot
        cmap = 'YlOrRd'
        cbar_label = 'Coefficient of Variation ($\\sigma$ / $\\mu$)'
        title = f"Systemic Inference Variance and Stability Profiles (VRAM Baseline: 12GB)\nHost: {meta['hostname']} | Hardware: {meta['gpu_info']}"
        fmt = ".3f"
    else:
        raise ValueError("Invalid operational mode selected. Choose 'tps' or 'cov'.")

    # Render figure configuration context
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    sns.heatmap(
        plot_data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.75,
        linecolor='#e0e0e0',
        cbar_kws={'label': cbar_label},
        ax=ax,
        robust=True
    )
    
    ax.set_title(title, pad=20, weight='bold')
    ax.set_ylabel("Evaluated Model Architecture (Physical Size)")
    ax.set_xlabel("Benchmarked Prompt Functional Domains")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[+] Graphical matrix map successfully exported to: {output_path}")
    else:
        plt.show()
    plt.close()

# Original __main__ block replaced by _extended_main() below.

# ==============================================================================
# Bar Chart Extension  (added below original heatmap code)
# ==============================================================================
# generate_barchart()
#
# Produces a publication-quality grouped bar chart in SVG format.
# Each bar represents one model; the Y-axis shows the grand mean of
# Tokens_Per_Second aggregated across all prompt IDs.  Error bars show
# ±1 standard deviation of the *per-prompt means* for that model,
# capturing prompt-type variance rather than within-prompt run noise.
#
# Design constraints honoured:
#   - Two colours only: black and a single accent (steel blue #2171B5).
#   - All spines, gridlines, ticks, and annotations in black or grey.
#   - Output: SVG vector format (infinitely scalable for print).
# ==============================================================================

def generate_barchart(df, meta, output_path=None):
    """
    Generates a two-colour publication bar chart: mean tokens/sec per model
    with ±1 SD error bars. Exports in both SVG and EPS formats.
    """
    import numpy as np

    # ------------------------------------------------------------------ data
    tps = df[df['Metric_Type'] == 'Tokens_Per_Sec'].copy()
    tps = tps[tps['Prompt_ID'].str.match(r'^P\d+(_T\d+)?$')]

    agg = (tps.groupby(['Model', 'Size'])['Mean']
              .agg(grand_mean='mean', cross_prompt_sd='std')
              .reset_index())

    agg = agg.sort_values('grand_mean', ascending=False).reset_index(drop=True)

    # ----------------------------------------------------------- Label Logic
    labels = []
    for row in agg.itertuples():
        if ':' in row.Model:
            m_base, m_suffix = row.Model.split(':', 1)
            labels.append(f"{m_base}\n{m_suffix} ({row.Size})")
        else:
            labels.append(f"{row.Model}\n({row.Size})")

    # ----------------------------------------------------------- style constants
    ACCENT   = '#2171B5'
    BLACK    = '#000000'
    GRIDGREY = '#CCCCCC'

    n = len(agg)
    x = np.arange(n)
    bar_width = 0.55

    # ------------------------------------------------------------- figure setup
    # Aspect ratio: 12x6 (20% wider than 10x6)
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        x,
        agg['grand_mean'],
        width=bar_width,
        color=ACCENT,
        edgecolor=BLACK,
        linewidth=0.8,
        zorder=3,
        label='Grand mean tokens/sec'
    )

    ax.errorbar(
        x,
        agg['grand_mean'],
        yerr=agg['cross_prompt_sd'],
        fmt='none',
        ecolor=BLACK,
        elinewidth=1.2,
        capsize=5,
        capthick=1.2,
        zorder=4,
        label='±1 SD (cross-prompt)'
    )

    # Annotations
    for bar, mean_val, sd_val in zip(bars, agg['grand_mean'], agg['cross_prompt_sd']):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            mean_val + sd_val + (ax.get_ylim()[1] * 0.005),
            f'{mean_val:.1f}',
            ha='center', va='bottom',
            fontsize=10, color=BLACK,
            fontfamily='serif'
        )

    # ------------------------------------------------------------------ axes
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontfamily='serif')
    ax.set_ylabel('Mean Inference Throughput (tokens s\u207b\u00b9)',
                  fontsize=13, fontfamily='serif', color=BLACK)
    ax.set_xlabel('Model (parameter footprint)',
                  fontsize=13, fontfamily='serif', color=BLACK)

    ax.yaxis.grid(True, color=GRIDGREY, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BLACK)
    ax.spines['bottom'].set_color(BLACK)
    ax.tick_params(colors=BLACK)

    ax.axhline(0, color=BLACK, linewidth=0.8)

    # ------------------------------------------------------------------ title
    title_str = (
        f'On-Premises LLM Inference Throughput — Ollama Benchmark\n'
        f'Host: {meta["hostname"]}  |  GPU: {meta["gpu_info"]}'
    )
    ax.set_title(title_str, fontsize=14, fontfamily='serif',
                 fontweight='bold', color=BLACK, pad=14)

    # ----------------------------------------------------------------- legend
    ax.legend(frameon=False, fontsize=11, loc='upper right', bbox_to_anchor=(0.95, 1))

    plt.tight_layout()

    # ------------------------------------------------------------------ output
    if output_path is None:
        output_path = f"barchart_{meta['hostname']}.svg"
    
    # Ensure file has .svg extension for the primary save
    if not output_path.endswith('.svg'):
        output_path += '.svg'

    # Save SVG
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    
    # Save EPS
    eps_path = output_path.replace('.svg', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    
    print(f"[+] Bar charts exported to: {output_path} and {eps_path}")
    plt.close()

# ---------------------------------------------------------------- CLI wiring
# Extend the existing argument parser to accept --barchart
import sys as _sys

def _extended_main():
    import os
    parser = argparse.ArgumentParser(
        description="Ollama CSV Heatmap + Bar Chart Plotter.")
    parser.add_argument("csv_file",
        help="Path to raw input benchmark CSV data file.")
    parser.add_argument("--mode", choices=['tps', 'cov'], default='tps',
        help="Heatmap mode: 'tps' or 'cov'.")
    parser.add_argument("--output",
        help="Destination file path for heatmap output.")
    parser.add_argument("--barchart", action='store_true',
        help="Generate the bar chart (SVG) in addition to, or instead of, the heatmap.")
    parser.add_argument("--barchart-only", action='store_true',
        help="Generate only the bar chart; skip the heatmap.")
    parser.add_argument("--barchart-output",
        help="Destination file path for the bar chart SVG.")

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"[-] Input path error: File '{args.csv_file}' not found.")
        _sys.exit(1)

    data_df, metadata = parse_csv_data(args.csv_file)

    # Bar chart
    if args.barchart or args.barchart_only:
        bc_out = args.barchart_output
        if bc_out is None:
            stem, _ = os.path.splitext(args.csv_file)
            bc_out = f"{stem}_barchart.svg"
        generate_barchart(data_df, metadata, output_path=bc_out)

    # Heatmap (unless --barchart-only was given)
    if not args.barchart_only:
        hm_out = args.output
        if hm_out is None:
            stem, _ = os.path.splitext(args.csv_file)
            hm_out = f"{stem}_heatmap_{args.mode}.png"
        generate_heatmap(data_df, metadata, mode=args.mode, output_path=hm_out)


if __name__ == "__main__":
    _extended_main()
