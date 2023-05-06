FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PATH=/miniconda/bin:${PATH}

WORKDIR /moss

RUN apt-get update && apt-get install -y vim tmux curl g++ cmake

RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh && \
  bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -p /miniconda -b && \
  rm Miniconda3-py310_23.3.1-0-Linux-x86_64.sh && \
  conda update -y conda

COPY . .

RUN python -m pip install --upgrade pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -e .
RUN pip install protobuf==3.20
RUN pip uninstall jax jaxlib -y
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
