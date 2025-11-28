# TensorRT-LLM Benchmark Comparison

This directory contains scripts and results for comparing **TensorRT Engine** vs **PyTorch Backend (HuggingFace)** performance.

## Overview

**Model**: Llama-3-8B-Instruct  
**Hardware**: 2x H200 GPUs  
**Configuration**: TP=2, FP16, ISL=512, OSL=128  
**Concurrency Levels Tested**: 16, 64, 128

## Directory Structure

```
bench_results/
├── README.md                          # This file
├── run_comparison_benchmark.sh        # Main benchmark script
├── visualize_results.py               # Visualization script
├── pytorch_hf/                        # PyTorch backend results
│   ├── concurrency_16.json
│   ├── concurrency_64.json
│   └── concurrency_128.json
├── trt_engine/                        # TensorRT engine results
│   ├── concurrency_16.json
│   ├── concurrency_64.json
│   └── concurrency_128.json
├── benchmark_comparison.png           # Comparison graphs
└── benchmark_summary.txt              # Text summary
```

## How to Run

### Step 1: Run the Benchmark

```bash
cd /home/nvidia/TensorRT-LLM/bench_results
chmod +x run_comparison_benchmark.sh
./run_comparison_benchmark.sh
```

**Duration**: ~10-15 minutes total
- PyTorch Backend: 3 concurrency levels × ~2 min each = ~6 min
- TensorRT Engine: 3 concurrency levels × ~2 min each = ~6 min
- Plus server startup/shutdown time

### Step 2: Generate Visualization

```bash
# Install matplotlib if not already installed
pip install matplotlib

# Run visualization script
python3 visualize_results.py
```

This will generate:
- `benchmark_comparison.png` - Comparison graphs for all metrics
- `benchmark_summary.txt` - Detailed text summary

## Metrics Compared

### Latency Metrics (Lower is Better)
- **TTFT (Time to First Token)**: Latency before first token appears
- **TPOT (Time per Output Token)**: Average time per token after the first
- **ITL (Inter-Token Latency)**: Time between consecutive tokens
- **E2EL (End-to-End Latency)**: Total request completion time

### Throughput Metrics (Higher is Better)
- **Output Token Throughput**: Tokens generated per second

## What Gets Compared

### Test 1: PyTorch Backend (HuggingFace)
```bash
trtllm-serve meta-llama/Meta-Llama-3-8B-Instruct --backend pytorch --tp_size 2
```
- Uses HuggingFace model directly
- PyTorch backend with auto-optimization
- JIT compilation on first run

### Test 2: TensorRT Engine (Pre-built)
```bash
trtllm-serve <engine_path> --tokenizer <tokenizer_path> --backend tensorrt --tp_size 2
```
- Uses pre-built TensorRT engine
- Engine optimized during build time
- No runtime compilation overhead

## Expected Results

Typically, TensorRT Engine shows:
- **Lower latency** (faster TTFT, TPOT, ITL, E2EL)
- **Higher throughput** (more tokens/sec)
- **Better scalability** with increasing concurrency

The exact speedup depends on:
- Model architecture
- Hardware capabilities
- Batch size and concurrency level
- Input/output sequence lengths

## Troubleshooting

### Server fails to start
```bash
# Check logs
docker exec trtllm tail -f /tmp/trtllm-serve-pytorch.log
docker exec trtllm tail -f /tmp/trtllm-serve-engine.log
```

### Benchmark fails
```bash
# Verify server is responding
docker exec trtllm curl -s http://localhost:8000/health
```

### Missing results
```bash
# Check if files exist in container
docker exec trtllm ls -la /tmp/bench_pytorch/
docker exec trtllm ls -la /tmp/bench_trt_engine/
```

## Manual Benchmark Commands

If you want to run individual benchmarks:

```bash
# PyTorch Backend - Concurrency 16
docker exec trtllm bash -c "python -m tensorrt_llm.serve.scripts.benchmark_serving \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --backend openai \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --num-prompts 80 \
    --max-concurrency 16 \
    --ignore-eos \
    --random-ids \
    --save-result \
    --result-dir /tmp/bench_pytorch \
    --result-filename concurrency_16.json"
```

## Notes

- The benchmark script automatically stops and restarts the server for each backend
- Results are saved in JSON format for easy parsing
- The visualization script requires matplotlib (`pip install matplotlib`)
- All benchmarks use `--ignore-eos` to ensure consistent output lengths
- The `--random-ids` flag generates synthetic prompts without downloading datasets

