#!/bin/bash

if [ $# -ne 10 ]; then
    echo "Usage: $0 <VIDEO_DATA_PATH> <VIDEO_PATH> <LLM_VERSION> <VT_VERSION> <VT_VERSION2> <CN_VERSION> <VERSION> <TRAIN_RECIPE> <MODEL_MAX_LENGTH> <NUM_FRAME>"
    exit 1
fi

# Assign the arguments to variables
VIDEO_DATA_PATH="$1"
VIDEO_PATH="$2"
LLM_VERSION="$3"
VT_VERSION="$4"
VT_VERSION2="$5"
CN_VERSION="$6"
VERSION="$7"
TRAIN_RECIPE="${8}"
MODEL_MAX_LENGTH="${9}"
NUM_FRAME="${10}"

VT_VARIANT="${VT_VERSION##*/}"
LLM_VARIANT="${LLM_VERSION##*/}"

deepspeed --include localhost:0,1 --master_port 29501 tinyllava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --video_data_path  $VIDEO_DATA_PATH \
    --video_folder $VIDEO_PATH \
    --is_multimodal True \
    --conv_version pretrain \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --num_frames $NUM_FRAME \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --fp16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm frozen \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --output_dir /data/vlm/zxj/result/llava_video_factory/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-pretrain
