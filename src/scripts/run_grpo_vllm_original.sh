cd src/r1-v/

# 获取当前时间
current_time=$(date +"%Y-%m-%d-%H-%M-%S")

WEATHER_PATH="/home/kaiyu/Graduation/WeatherRFT/data/dataset/WeatherCQ/WeatherCQ_dataset_deepseek_v3.json"
WEATHER_IMAGE_PATH="/home/kaiyu/Graduation/WeatherRFT/data/dataset/WeatherCQ/image"
DATA_LANGUAGE="cn"

QWEN_PATH="/home/kaiyu/Model/Qwen/Qwen2-VL-2B-Instruct"
HF_DATASET="leonardPKU/GEOQA_R1V_Train_8K"
BASE_OUTPUT_DIR="/home/kaiyu/Graduation/REF_REPOS/R1-V/new_output/grpo_original"

export DEBUG_MODE="true"
export HF_ENDPOINT='https://hf-mirror.com'

# 定义所有任务类别
CATEGORIES=("500hpa_situation" "850hpa_situation" "land_situation" "rain" "phenomena" "max_temp" "min_temp")

# 遍历每个任务类别
for CATEGORY in "${CATEGORIES[@]}"; do
    # 构建当前任务的运行名称
    RUN_NAME="${current_time}_grpo_original_${CATEGORY}"
    
    # 构建输出目录
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"
    
    # 构建日志路径
    export LOG_PATH="/home/kaiyu/Graduation/REF_REPOS/R1-V/log/grpo_original/log_${RUN_NAME}.txt"
    
    # 修改CATEGORIES数组，跳过500hpa_situation
    if [ "$CATEGORY" == "500hpa_situation" ]; then
        echo "Skipping 500hpa_situation category"
        continue
    fi

    # 构建exclude_category (排除当前类别以外的所有类别)
    EXCLUDE_CATEGORIES=()
    for EXCL in "${CATEGORIES[@]}"; do
        if [ "$EXCL" != "$CATEGORY" ]; then
            EXCLUDE_CATEGORIES+=("$EXCL")
        fi
    done
    EXCLUDE_STR=$(IFS=" " ; echo "${EXCLUDE_CATEGORIES[*]}")
    
    echo "=========================================="
    echo "Starting training for category: $CATEGORY"
    echo "Excluding: $EXCLUDE_STR"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES="0,2,3,4" torchrun \
        --nproc_per_node="3" \
        --nnodes="1" \
        --node_rank="0" \
        --master_addr="127.0.0.1" \
        --master_port="12365" \
        src/open_r1/grpo_original.py \
        --use_vllm True \
        --output_dir $OUTPUT_DIR \
        --model_name_or_path $QWEN_PATH \
        --dataset_name $HF_DATASET \
        --weather_path $WEATHER_PATH \
        --weather_image_path $WEATHER_IMAGE_PATH \
        --data_language $DATA_LANGUAGE \
        --temperature 1.0 \
        --deepspeed local_scripts/zero2_weatherrft.json \
        --reward_funcs accuracy format \
        --exclude_category $EXCLUDE_STR \
        --max_completion_length 512 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --logging_steps 1 \
        --bf16 \
        --report_to wandb \
        --gradient_checkpointing true \
        --attn_implementation flash_attention_2 \
        --num_train_epochs 2 \
        --run_name $RUN_NAME \
        --save_steps 50 \
        --save_only_model true \
        --num_generations 12 \
        --min_pixels 3136 \
        --max_pixels 401408 \
        --seed 42
        
    echo "Finished training for category: $CATEGORY"
    echo ""
done