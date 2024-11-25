import json
import os
import argparse
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from .lvbenchdataset import LongVideoBenchDataset

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

def extract_question_and_following(inputs):
    question_found = False
    result = []
    
    for item in inputs:
        if isinstance(item, str) and item.startswith("Question:"):
            result.append(item.split("Question: ")[-1])
            question_found = True
        elif question_found and isinstance(item, str):
            result.append(item)
    
    return '\n'.join(result)

def read_frame(images, video_processor):
    images_group = []
    for image in images:
        img = image.convert("RGB")
        img = video_processor(img)
        images_group.append(img)
    torch_imgs = torch.stack(images_group)
    torch_imgs = torch_imgs.unsqueeze(0)
    return torch_imgs

def check_ans(pred, gt):
    flag = False
    
    pred = pred.strip()
    gt = gt.strip()
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt = gt.lower() 
    
    if gt in pred_option.replace('.', ''):
        flag = True
    elif gt in pred_option:
        flag = True
        
    return flag

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    model.to(device="cuda")

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    video_processor = VideoPreprocess(image_processor, data_args)
    
    val_dataset = LongVideoBenchDataset(args.data_folder, "lvb_val.json", max_num_frames=16)
    test_dataset = LongVideoBenchDataset(args.data_folder, "lvb_test_wo_gt.json", max_num_frames=16)

    
    print("start to val!")
    correct_val = 0
    all_val = 0
    for example in tqdm(val_dataset):
        question = extract_question_and_following(example['inputs'])
        images_group = []
        sub = ""
        for item in example['inputs']:
            if isinstance(item, Image.Image):  # 图像处理
                img = item.convert("RGB")
                img = video_processor(img)
                images_group.append(img)
                sub = sub + "[image]"
            elif isinstance(item, str) and not item.startswith("Question:"):  # 字幕处理
                sub = sub + item
            else:
                break
        torch_imgs = torch.stack(images_group)
        video_tensor = torch_imgs.unsqueeze(0)
        sub = "subtitles:" + sub
        #print("sub:",sub)
        
        #images = [item for item in example['inputs'] if isinstance(item, Image.Image)]
        
        correct_answer = example['correct_choice']
        print("correct_answer:", correct_answer)
        
        question = sub + "\n" + "question: <image>" + "\n" + question
        #print("question:",question)
        msg = Message()
        msg.add_message(question)
        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        input_ids = input_ids.unsqueeze(0).cuda()
        
        #video_tensor = read_frame(images, video_processor)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=None,
                video=video_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            print("outputs:",outputs)
        
        all_val += 1
        if check_ans(pred=outputs, gt=correct_answer):
            correct_val += 1
        print(f"val correct: {correct_val/all_val * 100 :.2f}%")
    
    """
    print("start to test!")
    results = {}
    for example in tqdm(test_dataset):
        question = extract_question_and_following(example['inputs'])
        video_id = example['id']
        images = [item for item in example['inputs'] if isinstance(item, Image.Image)]
        
        question = "<image>" + "\n" + question
        msg = Message()
        msg.add_message(question)
        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        input_ids = input_ids.unsqueeze(0).cuda()
        
        video_tensor = read_frame(images, video_processor)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=None,
                video=video_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            #print("outputs:",outputs)
        
        outputs = outputs.strip()
        pred_option = outputs[0].upper()
        #print("pred_option:",pred_option)
        results[video_id] = pred_option
    
    with open(args.answers_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.answers_file}")
    print(f"val correct: {correct_val/all_val * 100 :.2f}%")
    """
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--data-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    args = parser.parse_args()

    eval_model(args)