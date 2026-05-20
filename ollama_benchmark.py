#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
OLLAMA LLM PERFORMANCE AND INFERENCE BENCHMARKING SUITE
--------------------------------------------------------------------------------
Description:
    A comprehensive benchmarking utility designed to evaluate local LLM 
    inference performance via the Ollama API. Measures tokens per second (TPS) 
    and total elapsed time across multiple prompt dimensions, long-context 
    summarization, and multi-turn conversations.
    Note runtime may be several hours.

Features:
    - Automated System Diagnostics: Identifies Hostname, OS, and GPU/VRAM info.
    - VRAM Warming: Cold-start warm-up phase to ensure consistent measurements.
    - Diverse Prompt Suite: 10 benchmarks covering math, logic, code, and ethics.
    - Long-Context Stress Test: 2,200+ word payload for KV cache evaluation.
    - Statistical Analysis: Computes Mean, Variance, and Std Dev over N runs.
    - Multi-Turn Tracking: Samples performance at specific intervals in a chat.
    - Structured Reporting: Outputs both a console report and a detailed CSV.

Research Reference:
    Implementation supporting the project and paper by I. Curington and K. Lano (2026):
    "Reusing Obsolete Windows 10 PCs for On-Premises Large Language Model Inference"

Version: 0.3.0
Author: Ian Curington
License: AGPLv3
--------------------------------------------------------------------------------
"""

import os
import sys
import time
import csv
import platform
import subprocess
import statistics
from typing import List, Dict, Any, Tuple, Optional, Union

# Catch IP issues BEFORE importing Ollama
# AUTOMATED PATCH: Fix client-side 0.0.0.0 destination routing issue
ollama_host: str = os.environ.get("OLLAMA_HOST", "")
if "0.0.0.0" in ollama_host:
    print("\n[!] Network Notice: OLLAMA_HOST is targeting '0.0.0.0'.")
    print("    While valid for server bindings, this is non-routable for client requests.")
        
    port: str = ollama_host.split(":")[-1] if ":" in ollama_host else "11434"
    patched_host: str = f"http://127.0.0.1:{port}"
        
    os.environ["OLLAMA_HOST"] = patched_host
    print(f"    -> Automatically re-routed Python client to local loopback: {patched_host}\n")

# Safe Import check for the official Ollama library
try:
    import ollama
except ImportError:
    print("[-] Error: The official 'ollama' Python library is missing.")
    print("    Please install it using: pip install ollama")
    sys.exit(1)

# --- GLOBAL CONFIGURATION ---

# Adjust the list below for alternative local models:
TARGET_MODELS: List[str] = [
    "gemma4:e4b",
    "gemma3:12b",
    "qwen3:14b",
    "granite4.1:8b",
    "phi4:14b",
    "deepseek-r1:1.5b",
    "mistral-nemo:12b",
    "gpt-oss:20b"
]

# Adjust Number of Runs per Model+Prompt
NUM_RUNS: int = 5 # gather enough repeated runs for basic statistics

# Output Results written at the conclusion of the benchmark
CSV_FILENAME: str = "ollama_benchmark_results.csv"

# Local Machine Info
def get_gpu_info() -> str:
    """
    Attempts to extract GPU model and total VRAM info using nvidia-smi.

    Returns:
        str: A string describing the detected GPU(s) and VRAM, or a fallback
             message if nvidia-smi is unavailable.
    """
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        gpus: List[str] = []
        for line in res.strip().split('\n'):
            parts = line.split(',')
            if len(parts) >= 2:
                gpus.append(f"{parts[0].strip()} ({parts[1].strip()} MB VRAM)")
        return ", ".join(gpus)
    except Exception:
        return "Unknown / Non-Nvidia GPU (Could not parse nvidia-smi)"


def generate_long_text() -> str:
    """
    Generates a high-density analytical essay exceeding 2,200 words to test 
    context window performance and KV cache efficiency.

    Returns:
        str: A long text string formatted with paragraphs, used for P5 benchmark.
    """
    paragraphs: List[str] = [
        "The institutional landscape of the modern global economy has quietly transitioned from legacy framework operations to highly integrated, algorithmically-driven infrastructure nodes. Over the past decade, the rapid scaling of distributed cloud computing architectures and neural networks has transformed how enterprises process vast datasets, handle logistics, and orchestrate consumer touchpoints. This systematic migration towards automation is not merely an optimization of existing practices; it represents a fundamental paradigm shift in organizational sociology.",
        "Simultaneously, the displacement patterns of modern automation have broken historical precedents. While twentieth-century industrial automation predominantly targeted routine, manual blue-collar tasks, the current wave of cognitive automation is directly encroaching upon non-routine, analytical white-collar domains. Content creation, legal discovery, contract analysis, financial forecasting, and even software compilation are no longer insulated from machine intelligence. Large language architectures and specialized transformers possess the capability to synthesize legal precedents across thousands of pages within seconds.",
        "Beyond the structural economic reallocations, the psychological impact of operating within these highly optimized digital environments remains profoundly complex. Modern workers are subject to unprecedented levels of telemetry and behavioral tracking. Every keystroke, mouse movement, communication interval, and task completion metric is continuously logged and processed by internal performance management algorithms. This totalizing panoptic surveillance creates a culture of hyper-surveillance that reshapes employee psychology.",
        "Furthermore, the ethical governance of these autonomous frameworks remains a critical vulnerability. As models assume greater responsibility over resource allocation, credit scoring, and algorithmic hiring, algorithmic biases become deeply entrenched within institutional infrastructure. These biases, often reflective of historical inequities hidden within the training datasets, are processed as objective mathematical truths by automated nodes. Without continuous intervention, rigorous algorithmic auditing, and human-in-the-loop oversight, these systems risk replicating and amplifying systemic inequality at a speed and scale previously unimaginable."
    ]
    
    full_text: List[str] = []
    # Loop to systematically scale paragraph matrix past 2,200 words
    while sum(len(p.split()) for p in full_text) < 2300:
        full_text.extend(paragraphs)
    return "\n\n".join(full_text)


def extract_metric(response: Any, key: str) -> Optional[Any]:
    """
    Robustly extracts a metric value from an Ollama response object.
    Accommodates dict structures, Pydantic objects, and direct attributes 
    across different versions of the Ollama library.

    Args:
        response (Any): The response object from ollama.chat() or ollama.generate().
        key (str): The name of the metric to extract (e.g., 'eval_count', 'eval_duration').

    Returns:
        Optional[Any]: The value associated with the key, or None if the key
                       cannot be resolved within the response structure.
    """
    if hasattr(response, key):
        return getattr(response, key)
    elif isinstance(response, dict):
        return response.get(key)
    elif hasattr(response, 'model_dump'):
        return response.model_dump().get(key)
    return None


def run_environment_checks() -> Tuple[str, str, List[str], List[str]]:
    """
    Validates the local environment, checks the connection to the Ollama daemon,
    and identifies which target models are locally available for testing.

    Returns:
        Tuple[str, str, List[str]]: A tuple containing (Hostname, GPU Info, Active Models).
    
    Raises:
        SystemExit: If the Ollama API is unreachable or no target models are available.
    """
    print("=" * 60)
    print(" SYSTEM DIAGNOSTICS & INITIALIZATION")
    print("=" * 60)
    
    hostname: str = platform.node()
    os_info: str = platform.platform()
    gpu_info: str = get_gpu_info()
    
    print(f"[+] Hostname: {hostname}")
    print(f"[+] OS Context: {os_info}")
    print(f"[+] Hardware Accelerator: {gpu_info}")

    # Check Ollama Connection and Manifest list
    try:
        local_manifest: Any = ollama.list()
        model_size_map = {} # store model weight sizes
        if hasattr(local_manifest, 'models'):
            for m in local_manifest.models:
                # convert bytes to human-readable GB string
                gb_size = getattr(m, 'size', 0) / (1000 ** 3)
                model_size_map[m.model] = f"{gb_size:.1f} GB"
            available_models: List[str] = list(model_size_map.keys())
        elif isinstance(local_manifest, dict) and 'models' in local_manifest:
            for m in local_manifest['models']:
                if isinstance(m, dict):
                    name = m.get('name', '')
                    gb_size = m.get('size', 0) / (1000 ** 3)
                    model_size_map[name] = f"{gb_size:.1f} GB"
            available_models: List[str] = list(model_size_map.keys())
        else:
            available_models = []

    except Exception as e:
        print(f"\n[-] Error: Failed to communicate with the Ollama API daemon. ({e})")
        print("    Ensure the Ollama service is running on your machine.")
        sys.exit(1)
        
    print("\n[+] Checking Target Model Availability:")
    active_models: List[str] = []
    missing_models: List[str] = []
    
    for model in TARGET_MODELS:
        match = [m for m in available_models if m == model or m.startswith(model + ":")]
        if match:
            print(f"  --> [AVAILABLE] {model}")
            active_models.append(model)
        else:
            print(f"  --> [MISSING]   {model}")
            missing_models.append(model)
            
    if missing_models:
        print("\n[!] Warning: Some requested benchmark models are missing from your system.")
        print("    To run benchmarks for them, please execute: ollama pull <model>")
            
    if not active_models:
        print("\n[-] Error: No target benchmark models are available. Aborting run.")
        sys.exit(1)
        
    print(f"\nProceeding to benchmark {len(active_models)} model(s)...\n")
    return hostname, gpu_info, active_models, model_size_map


def main() -> None:
    """
    Orchestrates the benchmarking lifecycle: diagnostic check, warm-up,
    prompt iteration, statistical computation, and report generation.
    """
    start_bench_time = time.time()  # Record overall start time
    
    hostname, gpu_info, active_models, model_size_map = run_environment_checks()
    long_text_payload: str = generate_long_text()
    
    # 10 Benchmark Prompts Mapping
    prompts_suite: Dict[str, Dict[str, str]] = {
        "P1": {"name": "Short Factual Retrieval", "text": "Name the SI unit of electrical resistance and state its symbol."},
        "P2": {"name": "Long Structured Prose", "text": "Explain the causes and consequences of the 2008 financial crisis in 1,000 words."},
        "P3": {"name": "Multi-Step Arithmetic Reasoning", "text": "A train travels from A to B at 80 km/h and returns at 120 km/h. If the total journey takes 5 hours, what is the distance between A and B? Show all working."},
        "P4": {"name": "Code Generation", "text": "Write a Python function that accepts a list of integers and returns a dictionary mapping each unique value to the list of indices at which it appears. Include docstring and type hints."},
        "P5": {"name": "Long-Context Summarisation", "text": f"Summarise the following text in five bullet points, each no longer than 30 words. \n\n{long_text_payload}"},
        "P6": {"name": "Instruction Following Constraints", "text": "Write a product description for a mechanical keyboard. Requirements: exactly 3 paragraphs, no use of the word 'perfect', final paragraph must include a call to action, reading age below 12."},
        "P7": {"name": "Logical/Deductive Reasoning", "text": "All members of the board approved the motion. Sarah did not approve the motion. Is Sarah a board member? Explain your reasoning step by step."},
        "P8": {"name": "Multilingual Translation Matrix", "text": "Translate the following paragraph into formal French, then back-translate your French into English, and identify any meaning lost in the process. \n\nParagraph: The rapid advancement of artificial intelligence has sparked a global debate on safety and ethics."},
        "P10": {"name": "Refusal / Edge Case Robustness", "text": "Explain how social engineering attacks work, and describe three techniques a security awareness trainer might use to demonstrate them to employees."}
    }
    
    p9_turns: List[str] = [
        "Hello! I am planning a research project on the history of printing presses. Can you give me a brief overview of Johannes Gutenberg's contribution?",
        "Fascinating. What were the primary social and economic impacts of this technological shift in 15th-century Europe?",
        "How did the Catholic Church react to the sudden proliferation of printed materials during that period?",
        "Can you compare this historical information explosion to the rise of the early internet in the late 20th century?",
        "Summarize the key parallels you just drew between the printing press and the internet into three concise bullet points."
    ]

    # prompt used to load initial weights and fill cache, results and stats ignored.
    warmup_prompt: str = "Write a simple Python function that calculates the factorial of a given integer. Omit explanations, bare code only."
    
    results_records: List[Dict[str, Any]] = []

    for model in active_models:
        print("=" * 70)
        print(f" BENCHMARKING MODEL: {model}")
        print("=" * 70)
        
        # --- WARM-UP RUN ---
        print("[~] Initiating warm-up sequence (loading weights into memory/VRAM)...")
        try:
            ollama.chat(model=model, messages=[{'role': 'user', 'content': warmup_prompt}])
            print("[+] Warm-up complete. Starting timed metric cycles.\n")
        except Exception as e:
            print(f"[-] Error during warm-up phase for {model}: {e}. Skipping model.")
            continue
            
        # --- PROCESS STANDARD PROMPTS ---
        for pid, pdata in prompts_suite.items():
            print(f" Running {pid}: {pdata['name']} ({NUM_RUNS}x iterations)...")
            tps_runs: List[float] = []
            elapsed_runs: List[float] = []
            
            for _ in range(NUM_RUNS):
                try:
                    response: Any = ollama.chat(model=model, messages=[{'role': 'user', 'content': pdata['text']}])
                    eval_count: int = extract_metric(response, 'eval_count') or 0
                    eval_duration_ns: int = extract_metric(response, 'eval_duration') or 0
                    total_duration_ns: int = extract_metric(response, 'total_duration') or 0
                    
                    tps: float = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0
                    elapsed: float = total_duration_ns / 1e9 if total_duration_ns > 0 else 0.0
                    
                    tps_runs.append(tps)
                    elapsed_runs.append(elapsed)
                except Exception:
                    tps_runs.append(0.0)
                    elapsed_runs.append(0.0)
            
            model_size_str = model_size_map.get(model, "Unknown")
            results_records.append({
                'model': model, 'size': model_size_str, 'pid': pid, 'pname': pdata['name'], 'metric': 'Tokens_Per_Sec',
                'runs': tps_runs, 'mean': statistics.mean(tps_runs), 
                'variance': statistics.variance(tps_runs) if len(tps_runs) > 1 else 0,
                'stdev': statistics.stdev(tps_runs) if len(tps_runs) > 1 else 0
            })
            results_records.append({
                'model': model, 'size': model_size_str, 'pid': pid, 'pname': pdata['name'], 'metric': 'Elapsed_Sec',
                'runs': elapsed_runs, 'mean': statistics.mean(elapsed_runs),
                'variance': statistics.variance(elapsed_runs) if len(elapsed_runs) > 1 else 0,
                'stdev': statistics.stdev(elapsed_runs) if len(elapsed_runs) > 1 else 0
            })

        # --- PROCESS MULTI-TURN DIALOGUE (P9) ---
        print(f" Running P9: Multi-Turn Conversation Simulation ({NUM_RUNS}x iterations)...")
        p9_tracking: Dict[int, Dict[str, List[float]]] = {1: {'tps': [], 'elap': []}, 3: {'tps': [], 'elap': []}, 5: {'tps': [], 'elap': []}}
        
        for _ in range(NUM_RUNS):
            messages: List[Dict[str, str]] = []
            for turn_idx, user_content in enumerate(p9_turns, 1):
                messages.append({'role': 'user', 'content': user_content})
                try:
                    resp: Any = ollama.chat(model=model, messages=messages)
                    assist_content: Any = extract_metric(resp, 'message')
                    messages.append({'role': 'assistant', 'content': assist_content.get('content', '') if isinstance(assist_content, dict) else getattr(assist_content, 'content', '')})
                    
                    if turn_idx in [1, 3, 5]:
                        t_count: int = extract_metric(resp, 'eval_count') or 0
                        t_dur: int = extract_metric(resp, 'eval_duration') or 0
                        p9_tracking[turn_idx]['tps'].append(t_count / (t_dur / 1e9) if t_dur > 0 else 0.0)
                        p9_tracking[turn_idx]['elap'].append((extract_metric(resp, 'total_duration') or 0) / 1e9)
                except Exception:
                    break
                    
        for turn in [1, 3, 5]:
            tps_vals, elap_vals = p9_tracking[turn]['tps'], p9_tracking[turn]['elap']
            for m_type, vals in [('Tokens_Per_Sec', tps_vals), ('Elapsed_Sec', elap_vals)]:
                results_records.append({
                    'model': model, 'size': model_size_str, 'pid': f"P9_T{turn}", 'pname': f"Multi-Turn Dialogue (Turn {turn})", 'metric': m_type,
                    'runs': vals, 'mean': statistics.mean(vals) if vals else 0,
                    'variance': statistics.variance(vals) if len(vals) > 1 else 0,
                    'stdev': statistics.stdev(vals) if len(vals) > 1 else 0
                })

    # --- CSV EXPORT ---
    try:
        with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["# OLLAMA LOCAL INFERENCE BENCHMARK REPORT"])
            writer.writerow([f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"# Host: {hostname}", f"# GPU: {gpu_info}"])
            writer.writerow(["Model", "Size", "Prompt_ID", "Prompt_Name", "Metric_Type", "Run_1", "Run_2", "Run_3", "Run_4", "Run_5", "Mean", "Variance", "Std_Dev"])
            for r in results_records:
                pad_runs = (r['runs'] + [0.0]*5)[:5]
                writer.writerow([r['model'], r['size'], r['pid'], r['pname'], r['metric']] + pad_runs + [f"{r['mean']:.4f}", f"{r['variance']:.4f}", f"{r['stdev']:.4f}"])
        print(f"\n[+] Global metrics successfully saved to: {CSV_FILENAME}")
    except IOError as ioe:
        print(f"\n[-] Error saving CSV: {ioe}")

    # --- CALCULATE ELAPSED WORKLOAD TIME ---
    total_elapsed_seconds = time.time() - start_bench_time
    hours, remainder = divmod(int(total_elapsed_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # --- SUMMARY CONSOLE REPORT ---
    print("\n" + "=" * 85)
    print(" SUMMARY BENCHMARK OUTPUT ENGINE REPORT")
    print("=" * 85)
    print(f"{'Model':<22} | {'ID':<6} | {'Metric Description':<25} | {'Mean Value':<12} | {'Std Dev':<10}")
    print("-" * 85)
    for r in results_records:
        unit: str = " t/sec" if r['metric'] == 'Tokens_Per_Sec' else " seconds"
        print(f"{r['model']:<22} | {r['pid']:<6} | {r['metric']:<25} | {r['mean']:>7.2f}{unit:<5} | {r['stdev']:>8.2f}")
    print("=" * 85)
    print(f" Total Benchmark Execution Time: {duration_str} (HH:MM:SS)")

# Entry point when running stand-alone
if __name__ == '__main__':
    main()
