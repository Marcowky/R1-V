cd src/r1-v/

# 获取当前时间
current_time=$(date +"%Y-%m-%d-%H-%M-%S")

RUN_NAME="qwen2-vl-2b_grpo_en_length_think_kill-${current_time}"  # to modify
WEATHER_PATH="/home/kaiyu/Graduation/WeatherRFT/data/dataset/WeatherCQ/WeatherCQ_dataset.json"
WEATHER_IMAGE_PATH="/home/kaiyu/Graduation/WeatherRFT/data/dataset/WeatherCQ/image"
DATA_LANGUAGE="en"

QWEN_PATH="/home/kaiyu/Model/Qwen/Qwen2-VL-2B-Instruct"
HF_DATASET="leonardPKU/GEOQA_R1V_Train_8K" 
OUTPUT_DIR="/home/kaiyu/Graduation/REF_REPOS/R1-V/output/${RUN_NAME}"

export DEBUG_MODE="true"
export LOG_PATH="/home/kaiyu/Graduation/REF_REPOS/R1-V/log/log_${RUN_NAME}.txt" 
export HF_ENDPOINT='https://hf-mirror.com'


CUDA_VISIBLE_DEVICES="2,3,4,5" torchrun \
    --nproc_per_node="3" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12368" \
    src/open_r1/grpo_weatherrft.py \
    --use_vllm True \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $QWEN_PATH \
    --dataset_name $HF_DATASET \
    --weather_path $WEATHER_PATH \
    --weather_image_path $WEATHER_IMAGE_PATH \
    --data_language $DATA_LANGUAGE \
    --temperature 1.0 \
    --deepspeed local_scripts/zero2_weatherrft.json \
    --reward_funcs accuracy format length related fluency_logical \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 12 \
    --min_pixels 3136 \
    --max_pixels 401408 \
    --seed 42