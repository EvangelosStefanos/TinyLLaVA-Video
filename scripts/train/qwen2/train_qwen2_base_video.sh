VIDEO_DATA_PATH=/app/data/dataset/text_files/cleaned_video_caption.json #pretrain annotation file path
FINETUNE_VIDEO_DATA_PATH=/app/data/dataset/text_files/cleaned_video_openqa.json #finetune annotation file path
VIDEO_PATH=/app/data/dataset #pretrain image dir
FINETUNE_VIDEO_PATH=/app/data/dataset #finetune image dir

LLM_VERSION=Qwen/Qwen2-0.5B # llm path
VT_VERSION=google/siglip-so400m-patch14-224 #vision tower path
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=nullresampler #connector type
CONV_VERSION=qwen2_base #chat template
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes
MODEL_MAX_LENGTH=3072 #max model length for llm
NUM_FRAME=16 # -1 means 1fps
NUM_QUERY=512
GROUP=16 # Only applicable to groupresampler

bash scripts/train/qwen2/pretrain_qwen2_video.sh "$VIDEO_DATA_PATH" "$VIDEO_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$NUM_FRAME" "$NUM_QUERY" "$GROUP" &&
bash scripts/train/qwen2/finetune_qwen2_video.sh "$FINETUNE_VIDEO_DATA_PATH" "$FINETUNE_VIDEO_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$NUM_FRAME" "$NUM_QUERY" "$GROUP"

