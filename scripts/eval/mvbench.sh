#!/bin/bash

MODEL_PATH="Zhang199/TinyLLaVA-Video-Qwen2.5-3B-Group-16-512"
MODEL_NAME="TinyLLaVA-Video-Qwen2.5-3B-Group-16-512"
EVAL_DIR="/app/data/dataset/eval/MVBench"

# num_frame=-1 means 1fps
python -m tinyllava.eval.eval_mvbench \
    --model-path $MODEL_PATH \
    --image-folder $EVAL_DIR/video \
    --question-file $EVAL_DIR/json \
    --answers-file $EVAL_DIR/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode qwen2_base \
    --num_frame 16 \
    --max_frame 16 
