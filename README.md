<h2 align="center">TinyLLaVA-Video</a><h5 align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-green)]()
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/ZhangXJ199/TinyLLaVA-Video/tree/main)

## 🎉 News

- [2024-12] 🔊 Our [TinyLLaVA-Video-v1](https://github.com/ZhangXJ199/TinyLLaVA-Video/tree/main) repository has been established.

## 📌 About
This is a framework of Small-scale Large Multimodal Models for video understanding based on [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory).

## Installation and Requirements

1. Clone this repository and navigate to the folder
```bash
git clone https://github.com/ZhangXJ199/TinyLLaVA-Video.git
cd TinyLLaVA-Video
```

2. Create a conda environment, activate it and install Packages
```Shell
conda create -n tinyllava_video python=3.10 -y
conda activate tinyllava_video
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages
```Shell
pip install flash-attn --no-build-isolation
```
##### Upgrade to the latest code base

```Shell
git pull
pip install -e .
```

## Get Started

### 1. Data Preparation

We combine partial data from two datasets: [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) and [
Video-LLaVA](https://huggingface.co/datasets/LanguageBind/Video-LLaVA). 

|   Stage  |               Source               |    #Sample    |
|----------| :--------------------------------: | :-----------: |
| Pretrain |   LLaVA-Video-178K + Video-LLaVA   |     397k      |
| Finetune |          LLaVA-Video-178K          |     491k      |

#### Pretrain Data

We use four subsets of [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K): ``0_30_s_academic_v0_1``, ``30_60_s_academic_v0_1``, ``0_30_s_youtube_v0_1``, and ``30_60_s_youtube_v0_1``, supplemented with the filtered [Video-LLaVA](https://huggingface.co/datasets/LanguageBind/Video-LLaVA). The organized pretraining annotations can be downloaded from here.

#### Finetune Data

We use four subsets of [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K): ``0_30_s_academic_v0_1``, ``30_60_s_academic_v0_1``, ``0_30_s_youtube_v0_1``, and ``30_60_s_youtube_v0_1``. The organized finetune annotations can be downloaded from here.

#### Organize Data

Organize the image files and annotation files as follows in ``path/to/your/dataset``:

```Shell
dataset
├── academic_source
├── liwei_youtube_videos
├── valley
├── text_files
│   ├── cleaned_video_caption.json
│   ├── cleaned_video_openqa.json
```
   

### 2. Train

Here's an example for training a LMM using Phi-2.

- Replace data paths with yours in `scripts/train/train_phi.sh`
- Replace `output_dir` with yours in `scripts/train/pretrain.sh`
- Replace `pretrained_model_path` and `output_dir` with yours in `scripts/train/finetune.sh`
- Adjust your GPU ids (localhost) and `per_device_train_batch_size` in `scripts/train/pretrain.sh` and `scripts/train/finetune.sh`

```bash
bash scripts/train/train_phi.sh
```

Important hyperparameters used in pretraining and finetuning are provided below.

| Training Stage | Global Batch Size | Learning rate | conv_version |
| -------------- | :---------------: | :-----------: | :----------: |
| Pretraining    | 256               | 1e-3          | pretrain     |
| Finetuning     | 128               | 2e-5          | phi          |

**Tips:** 

Global Batch Size = num of GPUs * `per_device_train_batch_size` * `gradient_accumulation_steps`, we recommand you always keep global batch size and learning rate as above except for lora tuning your model.

You can refer to [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) to modify components such as "llm," "vision_tower," and "train_recipe."

### 3. Evaluation

We currently provide evaluations on 3 benchmarks, including [Video-MME](https://video-mme.github.io/home_page.html#leaderboard), [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://longvideobench.github.io/).

#### Video-MME

1. Download [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME) and put it under ``path/to/your/dataset/eval/Video-MME``.
2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR``, ``conv-mode`` and ``duration`` in ``scripts/eval/videomme.sh``. There are three types of ``duration`` available for testing: ``short``, ``medium``, and ``long``.
3. Please use the following command for single-gpu inference.
   ```bash
   CUDA_VISIBLE_DEVICES=0 bash scripts/eval/videomme.sh
   ```

#### MVBench

1. Download [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) and put it under ``path/to/your/dataset/eval/MVBench``.
2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR`` and ``conv-mode`` in ``scripts/eval/mvbench.sh``.
3. Please use the following command for single-gpu inference.
   ```bash
   CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mvbench.sh
   ```

#### LongVideoBench

1. Download [LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench) and put it under ``path/to/your/dataset/eval/LongVideoBench``.
2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR`` and ``conv-mode`` in ``scripts/eval/lvbench.sh``.
3. Please use the following command for single-gpu inference.
   ```bash
   CUDA_VISIBLE_DEVICES=0 bash scripts/eval/lvbench.sh
   ```

## Model Zoo

### Trained Models

our trained models will coming soon.

### Model Performance

our trained models will coming soon.

### Quick Inference Scripts

1. Please change ``model_path``, ``prompt``, ``video_file`` and ``conv-mode`` in ``eval.py``.
2.  Please use the following command for single-gpu inference.
   ```bash
   CUDA_VISIBLE_DEVICES=0 python eval.py
   ```

## ❤️ Community efforts
* This repository is based on [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) project.
* Our codebase is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA) project. Great work!
