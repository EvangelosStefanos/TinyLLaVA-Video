import json
import os
import argparse
import torch
from tqdm import tqdm
import shortuuid
import random
import cv2
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

def read_video(video_path, num_frames, max_num_frames):
    video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
    duration = video.duration
    try:
        video_data = video.get_clip(start_sec=0.0, end_sec=duration)
    except Exception as e:
        print(f"Corrupted video found: {video_path}, Error: {e}")
    video_data = video_data['video'].permute(1, 0, 2, 3) #torch.Size([l, 3, W, H])

    total_frames = video_data.shape[0]
    if num_frames > 0:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        num_frames_to_extract = min(max_num_frames, max(1, int(duration)))
        frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)
    video_data = video_data[frame_indices]
    return video_data

def parse_multi_choice_response(response, all_choices, index2ans):
    response = response.strip()
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    if response[0] in all_choices:
        return response[0]
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

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    model.to(device="cuda")

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    video_processor = VideoPreprocess(image_processor, data_args)

    with open(args.question_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct = 0
    total = 0
    res_list = []
    acc_dict = {}

    for example in tqdm(data):
        task_type = example['question_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        
        question = example["question"]
        if task_type == 'open-ended':
            question = "<image>" + "\n" + question + "\n" + "Answer the question directly without explanation."
        elif task_type == 'multiple-choice':
            question = question + "\n" + "\n".join([f"{key}. {value}" for key, value in example["choices"].items()])
            question = "<image>" + "\n" + question + "\n" + "Answer with the option's letter from the given choices directly."
        print("question:", question)
        
        msg = Message()
        msg.add_message(question)
        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        input_ids = input_ids.unsqueeze(0).cuda()

        video_path = example["video"].replace("https://huggingface.co/datasets/yale-nlp/MMVU/resolve/main", args.image_folder)
        video_data = read_video(video_path, args.num_frame, args.max_frame)
        video_tensor = torch.stack([video_processor(video) for video in video_data])
        video_tensor = video_tensor.unsqueeze(dim=0)
        
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
    
        gt = example['answer']
        print("ground truth:", gt)

        if task_type == 'open-ended':
            pred_ans = outputs
            if len(outputs) > len(gt):
                if gt.lower() in outputs.lower():
                    acc_dict[task_type][0] += 1
                    correct += 1
                    print("correct!")
            else:
                if outputs.lower() in gt.lower():
                    acc_dict[task_type][0] += 1
                    correct += 1
                    print("correct!")  
        elif task_type == 'multiple-choice':
            all_choices = list(example['choices'].keys())
            pred_ans = parse_multi_choice_response(outputs, all_choices, example['choices'])
            if pred_ans==gt:
                acc_dict[task_type][0] += 1
                correct += 1
                print("correct!")
        print("pred_ans:", pred_ans)
        
        res_list.append({
            'outputs': outputs,
            'pred': pred_ans,
            'gt': gt,
            'path': video_path
        })
        
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print(f"Total Acc: {correct / total * 100 :.2f}%")
        print('-' * 30, task_type, '-' * 30)

    final_res = dict()
    correct = 0
    total = 0
    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
        correct += v[0]
        total += v[1]    
    final_res['Avg'] = correct / total * 100

    print(final_res)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, "w") as f:
        json.dump(final_res, f)
        f.write("\n")
        for item in res_list:
            json.dump(item, f)
            f.write("\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question_file", type=str, default="tables/question.json")
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--conv_mode", type=str, default="llama")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_frame", type=int, default=1)
    parser.add_argument("--max_frame", type=int, default=1)
    parser.add_argument("--answer_prompter", action="store_true")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    args = parser.parse_args()

    eval_model(args)
