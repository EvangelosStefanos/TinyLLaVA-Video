#!/bin/bash

MODEL_PATH="Zhang199/TinyLLaVA-Video-Qwen2.5-3B-Group-16-512"
MODEL_NAME="TinyLLaVA-Video-Qwen2.5-3B-Group-16-512"
EVAL_DIR="/app/data/dataset/eval/MMVU"

# num_frame=-1 means 1fps
python -m tinyllava.eval.eval_mmvu \
    --model_path $MODEL_PATH \
    --image_folder $EVAL_DIR \
    --question_file $EVAL_DIR/validation.json \
    --answers_file /app/output/eval/mmvu/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv_mode qwen2_base \
    --num_frame 16 \
    --max_frame 16
