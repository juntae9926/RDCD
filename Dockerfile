FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Seoul
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get -y install python3 \
    python3-pip \
    python3-dev \
    git ssh vim

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install --upgrade pip
RUN pip3 install setuptools

RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN service ssh restart

EXPOSE 8000


