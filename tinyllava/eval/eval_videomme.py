import argparse
import torch
import os
import re
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import shortuuid

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from PIL import Image
import math
import cv2  # 用于处理视频帧


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

def trans_ans(original_options):
    result = {}
    all_choices = []
    for option in original_options:
        option_letter = option.split(".")[0].strip()
        content = option.replace(option_letter + ". ", "")
        all_choices.append(option_letter)
        result[option_letter] = content
    return all_choices, result

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
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    video_processor = VideoPreprocess(image_processor, data_args)

    # Load .parquet file
    questions_df = pd.read_parquet(os.path.expanduser(args.question_file))
    questions = questions_df.to_dict(orient="records")
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    model.to(device="cuda")

    # Initialize counters
    total = {"short": 0, "medium": 0, "long": 0, "all": 0}
    correct = {"short": 0, "medium": 0, "long": 0, "all": 0}

    for i, line in enumerate(tqdm(questions)):
        idx = line["videoID"]
        question = line["question"]
        answer = line["answer"]
        print("answer:",answer)
        options = line["options"]
        options_text = "\n".join(options)
        video_path = os.path.join(args.image_folder, f"{line['videoID']}.mp4")

        # Process video frames (assuming video exists)
        if os.path.exists(video_path):
            num_frame = 16
            video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
            duration = video.duration
            try:
                video_data = video.get_clip(start_sec=0.0, end_sec=duration)
            except Exception as e:
                print(f"Corrupted video found: {video_path}, Error: {e}")
                continue
            video_data = video_data['video'].permute(1, 0, 2, 3)  # torch.Size([l, 3, W, H])

            total_frames = video_data.shape[0]
            frame_indices = np.linspace(0, total_frames - 1, num_frame, dtype=int)
            video_data = video_data[frame_indices]

            videos = []
            for video in video_data:
                video = video_processor(video)
                videos.append(video)
            video_tensor = torch.stack(videos)
            video_tensor = video_tensor.unsqueeze(dim=0)

            question = "<image>" + "\n" + question + "\n" + options_text
            question = question + "\n" + "Answer with the option's letter from the given choices directly."
        else:
            print(f"Video not found: {video_path}")
            continue

        msg = Message()
        msg.add_message(question)

        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        input_ids = input_ids.unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=None,
                video=video_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            print("outputs:",outputs)

        if check_ans(pred=outputs, gt=answer):
            correct[line["duration"]] += 1
            correct["all"] += 1
            print("correct:",line["duration"])

        total[line["duration"]] += 1
        total["all"] += 1
        
        co_du = correct[line["duration"]] / total[line["duration"]] * 100
        co_all = correct["all"] / total["all"] * 100
        print(f"duration Acc: {co_du :.2f}%")
        print(f"Total Acc: {co_all :.2f}%")

    # Calculate accuracy for each category
    for category in ["short", "medium", "long", "all"]:
        accuracy = (correct[category] / total[category]) * 100 if total[category] > 0 else 0
        print(f"{category.capitalize()} Accuracy: {accuracy:.2f}%")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="videos/")
    parser.add_argument("--question-file", type=str, default="tables/question.parquet")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    eval_model(args)

