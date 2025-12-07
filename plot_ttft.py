#!/usr/bin/env python3
"""
Script to plot benchmark metrics comparison across different frameworks and concurrency levels.
"""

import json
import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define the frameworks and their directories
FRAMEWORKS = {
    'vLLM': '1_vllm',
    'TRT-LLM PyTorch': '3_trtllm_pytorch',
    'TRT-LLM Engine': '4_trtllm_engine'
}

# Concurrency levels to plot
CONCURRENCIES = [1, 4, 8]

def load_metric(base_dir, framework_dir, concurrency, metric='mean_ttft_ms'):
    """Load a specific metric from the benchmark results JSON file."""
    filepath = os.path.join(base_dir, framework_dir, f'concurrency_{concurrency}.json')
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data.get(metric, None)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def plot_metric_comparison(base_dir, metric='mean_ttft_ms', output_file=None):
    """Create a grouped bar chart comparing a metric across frameworks and concurrency levels."""
    
    # Collect data - organized by framework, then concurrency
    data = {framework_name: [] for framework_name in FRAMEWORKS.keys()}
    
    for framework_name, framework_dir in FRAMEWORKS.items():
        for concurrency in CONCURRENCIES:
            metric_value = load_metric(base_dir, framework_dir, concurrency, metric)
            data[framework_name].append(metric_value if metric_value is not None else 0)
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    framework_names = list(FRAMEWORKS.keys())
    x = np.arange(len(CONCURRENCIES))  # X-axis now represents concurrency levels
    width = 0.25  # Width of each bar
    
    # Create bars for each framework
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, framework_name in enumerate(framework_names):
        offset = (i - len(framework_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, data[framework_name], width, 
                      label=framework_name,
                      color=colors[i], alpha=0.8)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
    
    # Determine y-axis label and title based on metric
    metric_labels = {
        'mean_ttft_ms': ('Mean TTFT (ms)', 'Mean Time To First Token'),
        'mean_tpot_ms': ('Mean TPOT (ms)', 'Mean Time Per Output Token'),
        'mean_itl_ms': ('Mean ITL (ms)', 'Mean Inter-Token Latency'),
        'mean_e2el_ms': ('Mean E2E Latency (ms)', 'Mean End-to-End Latency'),
        'request_throughput': ('Request Throughput (req/s)', 'Request Throughput'),
        'output_throughput': ('Output Throughput (tokens/s)', 'Output Throughput'),
        'total_token_throughput': ('Total Throughput (tokens/s)', 'Total Token Throughput'),
    }
    
    ylabel, title_metric = metric_labels.get(metric, (metric, metric.replace('_', ' ').title()))
    
    # Customize plot
    ax.set_xlabel('Concurrency Level', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'{title_metric} Comparison\nAcross Frameworks and Concurrency Levels', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Concurrency {c}' for c in CONCURRENCIES], fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add some padding to y-axis
    y_max = max([max(values) for values in data.values()])
    ax.set_ylim(0, y_max * 1.15)
    
    plt.tight_layout()
    
    # Save plot
    if output_file is None:
        output_file = f"{metric}_comparison.png"
    output_path = os.path.join(base_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Display summary
    print(f"\nSummary of {metric}:")
    print("-" * 80)
    print(f"{'Framework':<20} {'Concurrency 1':<20} {'Concurrency 4':<20} {'Concurrency 8':<20}")
    print("-" * 80)
    for framework_name in framework_names:
        values = data[framework_name]
        print(f"{framework_name:<20} {values[0]:>19.2f}  {values[1]:>19.2f}  {values[2]:>19.2f}")
    print("-" * 80)
    
    plt.close()  # Close the figure to free memory
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot benchmark metrics comparison across frameworks and concurrency levels.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Plot mean TTFT
  python plot_ttft.py --metric mean_ttft_ms
  
  # Plot multiple metrics
  python plot_ttft.py --metric mean_ttft_ms mean_tpot_ms request_throughput
  
  # Plot all common metrics
  python plot_ttft.py --all
        '''
    )
    parser.add_argument('--metric', nargs='+', default=['mean_ttft_ms'],
                       help='Metric(s) to plot (default: mean_ttft_ms)')
    parser.add_argument('--all', action='store_true',
                       help='Plot all common metrics')
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # Determine which metrics to plot
    if args.all:
        metrics = ['mean_ttft_ms', 'mean_tpot_ms', 'mean_itl_ms', 'mean_e2el_ms',
                   'request_throughput', 'output_throughput', 'total_token_throughput']
    else:
        metrics = args.metric
    
    print(f"Reading benchmark results from: {script_dir}")
    print(f"Plotting {len(metrics)} metric(s): {', '.join(metrics)}\n")
    
    output_files = []
    for metric in metrics:
        print(f"\n{'='*80}")
        print(f"Plotting {metric}...")
        print('='*80)
        try:
            output_file = plot_metric_comparison(script_dir, metric=metric)
            output_files.append(output_file)
        except Exception as e:
            print(f"Error plotting {metric}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Done! Generated {len(output_files)} plot(s):")
    for f in output_files:
        print(f"  - {f}")

