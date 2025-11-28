#!/usr/bin/env python3
"""
Benchmark Comparison Visualization
Compares TensorRT Engine vs PyTorch Backend (HuggingFace) performance
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configuration
BENCH_DIR = Path(__file__).parent
PYTORCH_DIR = BENCH_DIR / "pytorch_hf"
TRT_DIR = BENCH_DIR / "trt_engine"
CONCURRENCY_LEVELS = [16, 64, 128]

# Metrics to extract and plot
METRICS = {
    'ttft': {'name': 'Time to First Token (TTFT)', 'unit': 'ms', 'key': 'mean_ttft_ms'},
    'tpot': {'name': 'Time per Output Token (TPOT)', 'unit': 'ms', 'key': 'mean_tpot_ms'},
    'itl': {'name': 'Inter-Token Latency (ITL)', 'unit': 'ms', 'key': 'mean_itl_ms'},
    'e2el': {'name': 'End-to-End Latency (E2EL)', 'unit': 'ms', 'key': 'mean_e2el_ms'},
    'throughput': {'name': 'Output Token Throughput', 'unit': 'tokens/s', 'key': 'output_throughput'},
}

def load_benchmark_results(directory, concurrency):
    """Load benchmark results from JSON file"""
    file_path = directory / f"concurrency_{concurrency}.json"
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_metrics(data):
    """Extract key metrics from benchmark data"""
    if data is None:
        return None
    
    return {
        'mean_ttft_ms': data.get('mean_ttft_ms', 0),
        'median_ttft_ms': data.get('median_ttft_ms', 0),
        'mean_tpot_ms': data.get('mean_tpot_ms', 0),
        'median_tpot_ms': data.get('median_tpot_ms', 0),
        'mean_itl_ms': data.get('mean_itl_ms', 0),
        'median_itl_ms': data.get('median_itl_ms', 0),
        'mean_e2el_ms': data.get('mean_e2el_ms', 0),
        'median_e2el_ms': data.get('median_e2el_ms', 0),
        'output_throughput': data.get('output_throughput', 0),
        'request_throughput': data.get('request_throughput', 0),
    }

def create_comparison_plots():
    """Create comparison plots for all metrics"""
    # Load all data
    pytorch_data = {}
    trt_data = {}
    
    print("Loading benchmark results...")
    for concurrency in CONCURRENCY_LEVELS:
        pytorch_data[concurrency] = extract_metrics(
            load_benchmark_results(PYTORCH_DIR, concurrency)
        )
        trt_data[concurrency] = extract_metrics(
            load_benchmark_results(TRT_DIR, concurrency)
        )
    
    # Check if we have data
    if not any(pytorch_data.values()) or not any(trt_data.values()):
        print("Error: No valid benchmark data found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TensorRT Engine vs PyTorch Backend (HuggingFace) - Performance Comparison\n' + 
                 'Model: Llama-3-8B-Instruct | ISL: 512 | OSL: 128 | 2x H200 GPUs',
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Plot each metric
    metric_idx = 0
    for metric_key, metric_info in METRICS.items():
        ax = axes[metric_idx]
        metric_idx += 1
        
        # Extract data for this metric
        pytorch_values = [pytorch_data[c][metric_info['key']] 
                          for c in CONCURRENCY_LEVELS if pytorch_data[c]]
        trt_values = [trt_data[c][metric_info['key']] 
                      for c in CONCURRENCY_LEVELS if trt_data[c]]
        
        # Plot
        x = range(len(CONCURRENCY_LEVELS))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], pytorch_values, width, 
                       label='PyTorch (HF)', color='#3498db', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], trt_values, width,
                       label='TensorRT Engine', color='#76b900', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
        
        # Formatting
        ax.set_xlabel('Concurrency Level', fontweight='bold')
        ax.set_ylabel(f'{metric_info["name"]} ({metric_info["unit"]})', fontweight='bold')
        ax.set_title(metric_info['name'], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(CONCURRENCY_LEVELS)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add speedup annotation
        if metric_key in ['ttft', 'tpot', 'itl', 'e2el']:  # Lower is better
            speedups = [(p/t if t > 0 else 0) for p, t in zip(pytorch_values, trt_values)]
        else:  # Higher is better (throughput)
            speedups = [(t/p if p > 0 else 0) for p, t in zip(pytorch_values, trt_values)]
        
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0
        ax.text(0.95, 0.95, f'Avg Speedup: {avg_speedup:.2f}x',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    # Save figure
    output_path = BENCH_DIR / "benchmark_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison graph saved to: {output_path}")
    
    # Generate text summary
    generate_summary(pytorch_data, trt_data)

def generate_summary(pytorch_data, trt_data):
    """Generate text summary of benchmark results"""
    summary_path = BENCH_DIR / "benchmark_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BENCHMARK COMPARISON SUMMARY\n")
        f.write("TensorRT Engine vs PyTorch Backend (HuggingFace)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Model: Llama-3-8B-Instruct\n")
        f.write("Hardware: 2x H200 GPUs\n")
        f.write("Configuration: ISL=512, OSL=128, TP=2\n\n")
        
        for concurrency in CONCURRENCY_LEVELS:
            f.write("-" * 80 + "\n")
            f.write(f"CONCURRENCY LEVEL: {concurrency}\n")
            f.write("-" * 80 + "\n\n")
            
            pytorch = pytorch_data[concurrency]
            trt = trt_data[concurrency]
            
            if not pytorch or not trt:
                f.write("Data not available\n\n")
                continue
            
            f.write(f"{'Metric':<30} {'PyTorch (HF)':<20} {'TensorRT Engine':<20} {'Speedup':<15}\n")
            f.write("-" * 85 + "\n")
            
            # TTFT
            speedup = pytorch['mean_ttft_ms'] / trt['mean_ttft_ms'] if trt['mean_ttft_ms'] > 0 else 0
            f.write(f"{'Mean TTFT (ms)':<30} {pytorch['mean_ttft_ms']:<20.2f} {trt['mean_ttft_ms']:<20.2f} {speedup:<15.2f}x\n")
            
            # TPOT
            speedup = pytorch['mean_tpot_ms'] / trt['mean_tpot_ms'] if trt['mean_tpot_ms'] > 0 else 0
            f.write(f"{'Mean TPOT (ms)':<30} {pytorch['mean_tpot_ms']:<20.2f} {trt['mean_tpot_ms']:<20.2f} {speedup:<15.2f}x\n")
            
            # ITL
            speedup = pytorch['mean_itl_ms'] / trt['mean_itl_ms'] if trt['mean_itl_ms'] > 0 else 0
            f.write(f"{'Mean ITL (ms)':<30} {pytorch['mean_itl_ms']:<20.2f} {trt['mean_itl_ms']:<20.2f} {speedup:<15.2f}x\n")
            
            # E2EL
            speedup = pytorch['mean_e2el_ms'] / trt['mean_e2el_ms'] if trt['mean_e2el_ms'] > 0 else 0
            f.write(f"{'Mean E2EL (ms)':<30} {pytorch['mean_e2el_ms']:<20.2f} {trt['mean_e2el_ms']:<20.2f} {speedup:<15.2f}x\n")
            
            # Throughput
            speedup = trt['output_throughput'] / pytorch['output_throughput'] if pytorch['output_throughput'] > 0 else 0
            f.write(f"{'Output Throughput (tok/s)':<30} {pytorch['output_throughput']:<20.2f} {trt['output_throughput']:<20.2f} {speedup:<15.2f}x\n")
            
            f.write("\n")
    
    print(f"✓ Summary report saved to: {summary_path}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    with open(summary_path, 'r') as f:
        print(f.read())

if __name__ == "__main__":
    print("Starting benchmark visualization...")
    create_comparison_plots()
    print("\n✓ All done!")

