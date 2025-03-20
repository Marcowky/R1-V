#!/bin/bash

# The latest vllm==0.7.2 is required for this script: pip3 install vllm==0.7.2 
cd src/r1-v/

export DEBUG_MODE="true"
export LOG_PATH="./vllm_run.txt"

QWEN_PATH="/home/kaiyu/model/Qwen/Qwen2-VL-2B-Instruct"
HF_DATASET="MMInstruction/Clevr_CoGenT_TrainA_70K_Complex" 
OUTPUT_DIR="/home/kaiyu/Graduation/REF_REPOS/R1-V/output/vllm/Qwen2-VL-2B-Instruct-grpo-vllm-weatherrft" 
RUN_NAME="Qwen2-VL-2B-Instruct-grpo-vllm-weatherrft"

# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node="3" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_weatherrft.py --use_vllm True \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $QWEN_PATH \
    --dataset_name $HF_DATASET \
    --temperature 1.0 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16  \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_steps 13125 \
    --run_name $RUN_NAME \
    --save_steps 1000 \
    --save_only_model true
