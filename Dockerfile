FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt upgrade -y

# ubuntu package install
RUN apt install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        libffi-dev \
        libssl-dev \
        libbz2-dev \
        python3-pip \
        python3-setuptools \
        wget \
        git \
        tzdata \
        libgl1-mesa-dev \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libglib2.0-0

WORKDIR /workspace

COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
