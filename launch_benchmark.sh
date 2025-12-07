#!/bin/bash

for concurrency in 1 4 8; do
  echo "Running benchmark with concurrency: $concurrency"
  
  docker run --rm --gpus all --network host \
    --name bench \
    -v /home/nvidia/TensorRT-LLM/bench_results:/results \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4 \
    bash -c "python -m tensorrt_llm.serve.scripts.benchmark_serving \
      --model meta-llama/Meta-Llama-3-8B-Instruct \
      --backend openai \
      --host 127.0.0.1 \
      --port 8000 \
      --dataset-name random \
      --random-input-len 128 \
      --random-output-len 512 \
      --num-prompts 5 \
      --max-concurrency $concurrency \
      --ignore-eos \
      --random-ids \
      --save-result \
      --result-dir /results/3_trtllm_pytorch \
      --result-filename concurrency_${concurrency}.json \
      --percentile-metrics ttft,tpot,itl,e2el"
  
  echo "Completed benchmark with concurrency: $concurrency"
  echo "---"
done

echo "All benchmarks completed!"