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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama CSV Execution Heatmap Plotter Engine.")
    parser.add_argument("csv_file", help="Path to raw input benchmark CSV data file.")
    parser.add_argument("--mode", choices=['tps', 'cov'], default='tps', help="Visual encoding mode: 'tps' (Inference Velocity) or 'cov' (Coefficient of Variation/Stability).")
    parser.add_argument("--output", help="Optional explicit destination file path (e.g. output_heatmap.png).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"[-] Input path error: File '{args.csv_file}' not found.")
        sys.exit(1)
        
    data_df, metadata = parse_csv_data(args.csv_file)
    
    # Auto-generate file name targets if none are explicitly declared
    if not args.output:
        stem, _ = os.path.splitext(args.csv_file)
        args.output = f"{stem}_heatmap_{args.mode}.png"
        
    generate_heatmap(data_df, metadata, mode=args.mode, output_path=args.output)