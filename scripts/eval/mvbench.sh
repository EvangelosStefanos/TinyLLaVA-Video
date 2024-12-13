#!/bin/bash

MODEL_PATH="/data/vlm/zxj/result/llava_video_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune"
MODEL_NAME="tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune"
EVAL_DIR="/data/vlm/zxj/data/mm_data/MVBench"

python -m tinyllava.eval.eval_mvbench \
    --model-path $MODEL_PATH \
    --image-folder $EVAL_DIR/video \
    --question-file $EVAL_DIR/json \
    --answers-file $EVAL_DIR/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi