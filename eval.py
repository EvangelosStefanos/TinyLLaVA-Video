from tinyllava.eval.run_tiny_llava import eval_model

model_path = "/data/vlm/zxj/result/llava_video_factory-11.22/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune"
prompt = "Describe the scene at the beginning of the video."
video_file = "/data/vlm/zxj/others/demo3.mp4"
conv_mode = "phi" # or llama, gemma, etc

args = type('Args', (), {
    "model_path": model_path,
    "model": None,
    "query": prompt,
    "conv_mode": conv_mode,
    "image_file": None, #image_file,
    "video_file": video_file, #video_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "num_frame": 16,
    "max_frame": 16,
    "max_new_tokens": 512
})()

eval_model(args)