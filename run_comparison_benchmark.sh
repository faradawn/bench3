#!/bin/bash
set -e

# Benchmark Comparison: TensorRT Engine vs PyTorch Backend (HuggingFace)
# Model: Llama-3-8B-Instruct
# Concurrency levels: 16, 64, 128
# Fixed: ISL=512, OSL=128

CONTAINER_NAME="trtllm"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
TOKENIZER_PATH="/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
ENGINE_PATH="/app/tensorrt_llm/trt_output/llama/engine_fp16_2gpu"
ISL=512
OSL=128
CONCURRENCY_LEVELS="16 64 128"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}TensorRT-LLM Benchmark Comparison${NC}"
echo -e "${BLUE}================================${NC}"

# Function to stop server
stop_server() {
    echo -e "${RED}Stopping trtllm-serve...${NC}"
    docker exec $CONTAINER_NAME bash -c "pkill -f trtllm-serve || true"
    sleep 5
}

# Function to wait for server
wait_for_server() {
    echo -e "${GREEN}Waiting for server to be ready...${NC}"
    for i in {1..60}; do
        if docker exec $CONTAINER_NAME bash -c "curl -s http://localhost:8000/health" > /dev/null 2>&1; then
            echo -e "${GREEN}Server is ready!${NC}"
            return 0
        fi
        echo "Checking... ($i/60)"
        sleep 5
    done
    echo -e "${RED}Server failed to start!${NC}"
    exit 1
}

# Function to run benchmark
run_benchmark() {
    local backend=$1
    local concurrency=$2
    local output_dir=$3
    local num_prompts=$((concurrency * 5))
    
    echo -e "${BLUE}Running benchmark: Backend=$backend, Concurrency=$concurrency, Prompts=$num_prompts${NC}"
    
    docker exec $CONTAINER_NAME bash -c "python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model ${MODEL_NAME} \
        --backend openai \
        --dataset-name random \
        --random-input-len ${ISL} \
        --random-output-len ${OSL} \
        --num-prompts ${num_prompts} \
        --max-concurrency ${concurrency} \
        --ignore-eos \
        --random-ids \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --result-dir ${output_dir} \
        --result-filename concurrency_${concurrency}.json \
        --metadata '${backend},Concurrency=${concurrency},ISL=${ISL},OSL=${OSL}'"
}

# ============================================
# Phase 1: PyTorch Backend (HuggingFace Model)
# ============================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Phase 1: PyTorch Backend (HF Model)${NC}"
echo -e "${GREEN}========================================${NC}"


echo -e "${BLUE}Starting trtllm-serve with PyTorch backend...${NC}"
docker exec -d $CONTAINER_NAME bash -c "cd /app/tensorrt_llm && nohup trtllm-serve ${MODEL_NAME} \
    --backend pytorch \
    --tp_size 2 \
    --max_batch_size 256 \
    --max_num_tokens 2048 \
    --max_seq_len 4096 \
    --kv_cache_free_gpu_memory_fraction 0.9 \
    --host 0.0.0.0 \
    --port 8000 > /tmp/trtllm-serve-pytorch.log 2>&1 &"

wait_for_server

# Run benchmarks for each concurrency level
for concurrency in $CONCURRENCY_LEVELS; do
    run_benchmark "PyTorch-HF" $concurrency "/tmp/bench_pytorch"
    echo -e "${GREEN}Completed: PyTorch Backend, Concurrency=$concurrency${NC}\n"
done

# Copy results to host
echo -e "${BLUE}Copying PyTorch results to host...${NC}"
docker cp $CONTAINER_NAME:/tmp/bench_pytorch /tmp/
mkdir -p /home/nvidia/TensorRT-LLM/bench_results/pytorch_hf
cp -r /tmp/bench_pytorch/* /home/nvidia/TensorRT-LLM/bench_results/pytorch_hf/

# ============================================
# Phase 2: TensorRT Engine Backend
# ============================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Phase 2: TensorRT Engine (Pre-built)${NC}"
echo -e "${GREEN}========================================${NC}"

stop_server

echo -e "${BLUE}Starting trtllm-serve with TensorRT engine...${NC}"
docker exec -d $CONTAINER_NAME bash -c "cd /app/tensorrt_llm && nohup trtllm-serve ${ENGINE_PATH} \
    --tokenizer ${TOKENIZER_PATH} \
    --backend tensorrt \
    --tp_size 2 \
    --max_batch_size 256 \
    --max_num_tokens 2048 \
    --max_seq_len 4096 \
    --kv_cache_free_gpu_memory_fraction 0.9 \
    --host 0.0.0.0 \
    --port 8000 > /tmp/trtllm-serve-engine.log 2>&1 &"

wait_for_server

# Run benchmarks for each concurrency level
for concurrency in $CONCURRENCY_LEVELS; do
    run_benchmark "TensorRT-Engine" $concurrency "/tmp/bench_trt_engine"
    echo -e "${GREEN}Completed: TensorRT Engine, Concurrency=$concurrency${NC}\n"
done

# Copy results to host
echo -e "${BLUE}Copying TensorRT Engine results to host...${NC}"
docker cp $CONTAINER_NAME:/tmp/bench_trt_engine /tmp/
mkdir -p /home/nvidia/TensorRT-LLM/bench_results/trt_engine
cp -r /tmp/bench_trt_engine/* /home/nvidia/TensorRT-LLM/bench_results/trt_engine/

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Results saved to: /home/nvidia/TensorRT-LLM/bench_results/"
echo -e "  - pytorch_hf/"
echo -e "  - trt_engine/"
echo -e "\nNext step: Run the visualization script to generate comparison graphs"

