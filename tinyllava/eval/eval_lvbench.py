import json
import os
import argparse
import torch
import random
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from .lvbenchdataset import LongVideoBenchDataset

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

def extract_question_and_following(example):
    inputs = example['inputs']
    question_found = False
    result = []
    options = []
    
    for item in inputs:
        if isinstance(item, str) and item.startswith("Question:"):
            result.append(item.split("Question: ")[-1])
            question_found = True
        elif question_found and isinstance(item, str):
            result.append(item)
            if item.startswith("Answer"):
                continue
            else:
                options.append(item)
    
    return '\n'.join(result), options

def read_frame(images, video_processor):
    images_group = []
    for image in images:
        img = image.convert("RGB")
        img = video_processor(img)
        images_group.append(img)
    torch_imgs = torch.stack(images_group)
    torch_imgs = torch_imgs.unsqueeze(0)
    return torch_imgs

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    response = response.strip()
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    if len(response) == 0:
        return random.choice(all_choices)
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

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
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
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

def select_from_options(options):
    all_choices = [option.split('.')[0] for option in options]
    index2ans = {option.split('.')[0]: option.split('. ')[1][:-1] for option in options}
    return all_choices, index2ans

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    model.to(device="cuda")

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    video_processor = VideoPreprocess(image_processor, data_args)
    
    val_dataset = LongVideoBenchDataset(args.data_folder, "lvb_val.json", num_frame=args.num_frame, max_num_frames=args.max_frame)
    #test_dataset = LongVideoBenchDataset(args.data_folder, "lvb_test_wo_gt.json", max_num_frames=max_num_frames)
    
    print("start to val!")
    correct_val = 0
    all_val = 0
    for example in tqdm(val_dataset):
        question, options = extract_question_and_following(example)
        all_choices, index2ans = select_from_options(options)
        #images = [item for item in example['inputs'] if isinstance(item, Image.Image)]
        #video_tensor = read_frame(images, video_processor)
        #question = "<image>" + "\n" + question
        #print("question:",question)
        
        images_group = []
        sub = ""
        for item in example['inputs']:
            if isinstance(item, Image.Image):
                img = item.convert("RGB")
                img = video_processor(img)
                images_group.append(img)
                sub = sub + "[image]"
            elif isinstance(item, str) and not item.startswith("Question:"):
                sub = sub + item
            else:
                break
        torch_imgs = torch.stack(images_group)
        video_tensor = torch_imgs.unsqueeze(0)
        sub = "subtitles:" + sub
        question = sub + "\n" + "question: <image>" + "\n" + question
        
        correct_answer = example['correct_choice']
        print("correct_answer:", correct_answer)
        
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
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            print("outputs:",outputs)
        pred_ans = parse_multi_choice_response(outputs, all_choices, index2ans)
        print("pred_ans:",pred_ans)
        
        all_val += 1
        if pred_ans==correct_answer:
            correct_val += 1
        print(f"val correct: {correct_val/all_val * 100 :.2f}%")
    
    # code for testing
    """
    print("start to test!")
    results = {}
    for example in tqdm(test_dataset):
        question, options = extract_question_and_following(example)
        all_choices, index2ans = select_from_options(options)
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
        
        pred_ans = parse_multi_choice_response(outputs, all_choices, index2ans)
        #print("pred_option:",pred_ans)
        results[video_id] = pred_ans
    
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
    parser.add_argument("--num_frame", type=int, default=1)
    parser.add_argument("--max_frame", type=int, default=1)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    args = parser.parse_args()

    eval_model(args)