# Docker version 27.3.1
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo/63377623#63377623
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get -y install ffmpeg

RUN apt-get -y install git

RUN python -m pip install --upgrade pip wheel setuptools opencv-python

WORKDIR /app

COPY pyproject.toml pyproject.toml
RUN pip install -e .
RUN pip install flash-attn==2.5.7 --no-build-isolation

# JEPA requirements
# COPY ./jepa/requirements.txt ./jepa/requirements.txt
# COPY ./jepa/setup.py ./jepa/setup.py
# WORKDIR /app/jepa
# RUN pip install .

RUN mkdir output && pip list --format=freeze > /app/output/requirements.txt

COPY ./ ./

ENV PYTHONPATH="/app:/app/tinyllava:/app/jepa"
# ENV PYTHONWARNINGS="ignore"

CMD [ "bash", "scripts/train/qwen2/train_qwen2_base_video.sh" ]

# ENV CUDA_VISIBLE_DEVICES="0,1"
# CMD [ "bash", "scripts/eval/mmvu.sh" ]
