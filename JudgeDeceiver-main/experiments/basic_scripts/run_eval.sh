#!/bin/bash
# gemma31b_alpaca_eval_1_20251125-09:10:57.json
# llama3_arena_hard_1_20251123-16:28:09.json
# gemma31b_arena_hard_1_20251206-11:39:38.json
export model="gemma3_12"  # llama2,llama3,openchat35 or mistral2
export dataset="code_judge_bench"  # llmbar or mtbench or arena_hard，code_judge_bench_train,alpaca_eval
export device=6

if [ ! -d "../eval" ]; then
    mkdir "../eval"
    echo "Folder '../eval' created."
fi

if [ ! -d "../eval/${model}" ]; then
    mkdir "../eval/${model}"
    echo "Folder '../eval/${model}' created."
else
    echo "Folder '../eval/${model}' already exists."
fi

# for i in {1,2,3,4,5,6,7,8,9,10,}; do
#     python -u ../evaluate.py \
#         --config="../configs/${model}.py" \
#         --config.train_data="../../dataset/data_for_eval/basic/${dataset}/${dataset}_test_${i}.csv" \
#         --config.logfile="../results/${model}/${model}_${dataset}_${i}.json" \
#         --config.n_train_data=50 \
#         --config.n_test_data=100 \
#         --config.eval_model="${model}"
# done
# 我们的数据集只有一份，因此不需要上面的循环，直接指定就好，运行命令为bash run_eval.sh qwen3(这里的qwen3指的是配置文件qwen3.py，在qwen3.py中设置好路径就能用到我们的模型)，另外在evaluate.py，也要修改好模型路径
# -config.logfile=../results/qwen3/qwen3_llmbar_1_20251023-19:00:16.json:这个参数取相应文件最后一个时间段的即可
# /home/wzdou/project/JudgeDeiceive1/JudgeDeceiver-main/experiments/results/qwen25/qwen25_arena_hard_0_20251111-00:00:43.json
# gemma3_arena_hard_1_20251113-21:33:05.json
# gemma3_arena_hard_1_20251115-08:55:04
# gemma3_code_judge_bench_1_20251115-08:58:08
# controls[99::100]
# gemma3_alpaca_eval_1_20251115-21:37:44
CUDA_VISIBLE_DEVICES=${device} python -u ../evaluate.py \
    --config="../configs/${model}.py" \
    --config.train_data="../../dataset/data_for_train/basic/ourdata/${dataset}_test.json" \
    --config.test_data="../../dataset/data_for_train/basic/ourdata/${dataset}_test.json" \
    --config.logfile="../results/${model}/${model}_${dataset}_1_20251206-23:07:29.json" \
    --config.n_train_data=9999 \
    --config.n_test_data=9999 \
    --config.eval_model="${model}"