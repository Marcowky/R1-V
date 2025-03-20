cd src/r1-v/

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"
export HF_ENDPOINT='https://hf-mirror.com'

torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo.py \
    --output_dir /home/kaiyu/Graduation/REF_REPOS/R1-V/output \
    --model_name_or_path /home/kaiyu/Model/Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name leonardPKU/GEOQA_R1V_Train_8K \
    --deepspeed local_scripts/zero3.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8