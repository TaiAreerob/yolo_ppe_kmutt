FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

RUN apt-get update -y && apt-get install -y python3 python3-pip python3-dev libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt
