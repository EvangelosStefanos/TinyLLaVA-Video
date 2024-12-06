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

data_list = {
    "Fine-grained Pose": ("fine_grained_pose.json", "/nturgbd/", "video_avi", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Action Sequence": ("action_sequence.json", "/star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "/star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "/ssv2_video/", "video_webm", False),
    "Fine-grained Action": ("fine_grained_action.json", "/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "/star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "/sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "/perception/videos/", "video", False),
    "Character Order": ("character_order.json", "/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "/vlnqa/", "video", False),
    "Counterfactual Inference": ("counterfactual_inference.json", "/clevrer/video_validation/", "video", False),
}

class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, video_processor, num_segments=16):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'frame': self.read_frame,
            'video_webm': self.read_video_webm,
            'video_avi': self.read_video_avi
        }
        
        self.video_processor = video_processor
        self.num_segments = num_segments

    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        try:
            video_data = video.get_clip(start_sec=0.0, end_sec=duration)
        except Exception as e:
            print(f"Corrupted video found: {video_path}, Error: {e}")
        video_data = video_data['video'].permute(1, 0, 2, 3) #torch.Size([l, 3, W, H])

        total_frames = video_data.shape[0]
        frame_indices = np.linspace(0, total_frames - 1, self.num_segments, dtype=int)
        video_data = video_data[frame_indices]
        
        videos = []
        for video in video_data:
            video = self.video_processor(video)
            videos.append(video)
        video_tensor = torch.stack(videos)
        video_tensor = video_tensor.unsqueeze(dim=0)
    
        return video_tensor
    
    def read_video_avi(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = []
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = torch.tensor(vr[frame_index].asnumpy(), dtype=torch.float16)
            img = self.video_processor(img)
            images_group.append(img)
        video_tensor = torch.stack(images_group)  # Shape: (num_segments, C, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # Shape: (1, num_segments, C, H, W)

        return video_tensor
    
    def read_video_webm(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = []
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = vr[frame_index].to(dtype=torch.float16)
            img = self.video_processor(img)
            images_group.append(img)
        video_tensor = torch.stack(images_group)  # Shape: (num_segments, C, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # Shape: (1, num_segments, C, H, W)

        return video_tensor
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1)
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg")).convert("RGB")
            img = self.video_processor(img)
            images_group.append(img)
        torch_imgs = torch.stack(images_group)
        torch_imgs = torch_imgs.unsqueeze(0)
        return torch_imgs

    def qa_template(self, data):
        question = f"{data['question']}\n"
        answer = data['answer']
        answer_idx = -1
        all_choices = []
        index2ans = {}
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            all_choices.append(chr(ord('A') + idx))
            index2ans[chr(ord('A') + idx)] = c
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"{chr(ord('A') + answer_idx)}"
        return question, answer, all_choices, index2ans

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        question, answer, all_choices, index2ans = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'question': question, 
            'answer': answer, 
            'all_choices': all_choices,
            'index2ans': index2ans,
            'task_type': self.data_list[idx]['task_type']
        }

def check_ans(pred, gt):
    flag = False
    
    pred = pred.strip()
    gt = gt.strip()
    
    pred_list = pred.lower().split(' ')
    #print("pred_list:",pred_list)
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
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

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    model.to(device="cuda")

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    video_processor = VideoPreprocess(image_processor, data_args)
    
    updated_data_list = {
        key: (val[0], args.image_folder + val[1], val[2], val[3]) for key, val in data_list.items()
    }
    
    dataset = MVBench_dataset(args.question_file, updated_data_list, video_processor)
    
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}

    for example in tqdm(dataset):
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        #print("example:",example) # video, question, answer, task_type
        
        question = example["question"]
        question = "<image>" + "\n" + question + "\n" + "Answer with the option's letter from the given choices directly."
        
        msg = Message()
        msg.add_message(question)
        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        input_ids = input_ids.unsqueeze(0).cuda()
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=None,
                video=example["video"],
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
        pred_ans = parse_multi_choice_response(outputs, example['all_choices'], example['index2ans'])
        print("pred_ans:", pred_ans)
        
        res_list.append({
            'pred': pred_ans,
            'gt': gt
        })
        
        if pred_ans==gt:
            acc_dict[task_type][0] += 1
            correct += 1
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
        


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
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
