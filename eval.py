from tinyllava.eval.run_tiny_llava import eval_model

model_path = "/data/vlm/zxj/result/llava_video_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune"
prompt = "What are the people in the video doing?"
video_file = "/data/vlm/zxj/demo.mp4"
conv_mode = "phi" # or llama, gemma, etc

args = type('Args', (), {
    "model_path": model_path,
    "model": None,
    "query": prompt,
    "conv_mode": conv_mode,
    "image_file": None,
    "video_file": video_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)