cd src/r1-v

# 获取当前时间
current_time=$(date +"%Y-%m-%d-%H-%M-%S")

run_name="Qwen2-VL-2B-GRPO-WeatherRFT-en-4gpu-formate-${current_time}" # to modify

export DEBUG_MODE="true"
export LOG_PATH="/home/kaiyu/Graduation/REF_REPOS/R1-V/log/log_${run_name}.txt" 
export HF_ENDPOINT='https://hf-mirror.com'

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12368" \
    src/open_r1/grpo_weatherrft.py \
    --output_dir /home/kaiyu/Graduation/REF_REPOS/R1-V/output/$run_name \
    --model_name_or_path /home/kaiyu/Model/Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name leonardPKU/GEOQA_R1V_Train_8K \
    --deepspeed local_scripts/zero3.json \
    --reward_funcs accuracy format \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $run_name \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 4 \
    --min_pixels 3136 \
    --max_pixels 401408 \
    --seed 42