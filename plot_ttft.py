#!/usr/bin/env python3
"""
Script to plot mean TTFT (Time To First Token) comparison across different frameworks and concurrency levels.
"""

import json
import os
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

def plot_ttft_comparison(base_dir, output_file='ttft_comparison.png'):
    """Create a grouped bar chart comparing mean TTFT across frameworks and concurrency levels."""
    
    # Collect data
    data = {concurrency: [] for concurrency in CONCURRENCIES}
    
    for concurrency in CONCURRENCIES:
        for framework_name, framework_dir in FRAMEWORKS.items():
            metric_value = load_metric(base_dir, framework_dir, concurrency)
            data[concurrency].append(metric_value if metric_value is not None else 0)
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    framework_names = list(FRAMEWORKS.keys())
    x = np.arange(len(framework_names))
    width = 0.25  # Width of each bar
    
    # Create bars for each concurrency level
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, concurrency in enumerate(CONCURRENCIES):
        offset = (i - len(CONCURRENCIES) / 2 + 0.5) * width
        bars = ax.bar(x + offset, data[concurrency], width, 
                      label=f'Concurrency {concurrency}',
                      color=colors[i], alpha=0.8)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
    
    # Customize plot
    ax.set_xlabel('Framework', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean TTFT (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Time To First Token Comparison\nAcross Frameworks and Concurrency Levels', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(framework_names, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add some padding to y-axis
    y_max = max([max(data[c]) for c in CONCURRENCIES])
    ax.set_ylim(0, y_max * 1.15)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(base_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Display summary
    print("\nSummary of Mean TTFT (ms):")
    print("-" * 60)
    print(f"{'Framework':<20} {'Concurrency 1':<15} {'Concurrency 4':<15} {'Concurrency 8':<15}")
    print("-" * 60)
    for i, (framework_name, _) in enumerate(FRAMEWORKS.items()):
        values = [data[c][i] for c in CONCURRENCIES]
        print(f"{framework_name:<20} {values[0]:>14.2f}  {values[1]:>14.2f}  {values[2]:>14.2f}")
    print("-" * 60)
    
    return output_path

if __name__ == '__main__':
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    print(f"Reading benchmark results from: {script_dir}")
    output_file = plot_ttft_comparison(script_dir)
    
    print(f"\nDone! View the plot at: {output_file}")

