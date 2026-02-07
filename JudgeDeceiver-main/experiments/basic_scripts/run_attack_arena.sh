#!/bin/bash
# JUDGE_MODEL_PATH="/share/disk/llm_cache/Llama-3.1-8B-Instruct"
JUDGE_MODEL_PATH="/share/disk/llm_cache/gemma-3-4b-it/"

export WANDB_MODE=disabled
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export n=2
export m=0
export model=gemma3 # llama2,llama3,openchat35 or msistral2,glm-4-9b-chat-hf
export qid=1 
export device=5
export dataset=arena_hard  # llmbar or mtbench
# # "arena_hard" "code_judge_bench" "alpaca_eval"
# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
fi

if [ ! -d "../results/${model}" ]; then
    mkdir "../results/${model}"
    echo "Folder '../results/${model}' created."
elsess
    echo "Folder '../results/${model}' already exists."
    echo "Folder '../results/${dataset}' already exists."
fi
# --config.train_data="../../dataset/data_for_train/basic/${dataset}/${dataset}_train_${qid}.csv"
    # --config.train_data="../../dataset/data_for_train/basic/${model}/${dataset}_train.json" \
CUDA_VISIBLE_DEVICES=${device} python -u ../main.py \
    --config="../configs/${model}.py" \
    --config.attack=gcg \
    --config.judge_model="${JUDGE_MODEL_PATH}" \
    --config.train_data="../../dataset/data_for_train/basic/ourdata/${dataset}_train.json"  \
    --config.result_prefix="../results/${model}/${model}_${dataset}_${qid}" \
    --config.progressive_goals=True \
    --config.stop_on_success=True \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_test_data=$n \
    --config.n_steps=100 \
    --config.test_steps=1 \
    --config.align_weight=1.0 \
    --config.enhance_weight=1.0 \
    --config.perplexity_weight=0.1 \
    --config.control_weight=0.1 \
    --config.batch_size=16 \
    --config.data_offset=0

