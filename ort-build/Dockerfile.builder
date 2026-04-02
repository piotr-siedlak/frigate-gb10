FROM nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    git ninja-build build-essential patchelf wget unzip && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN pip3 install cmake==3.29.6 numpy==1.26.4 psutil wheel packaging setuptools

WORKDIR /build
RUN git clone --recursive --depth 1 --branch v1.24.4 \
    https://github.com/microsoft/onnxruntime.git .

# Pre-download Eigen from GitHub mirror (SHA1 verified: matches deps.txt for v1.24.4)
COPY eigen.zip /tmp/eigen.zip
RUN cd /tmp && unzip -q eigen.zip && \
    mv eigen-1d8b82b0740839c0de7f1242a3585e3390ff5f33 /tmp/eigen-src

COPY build_ort.sh /build_ort.sh
RUN chmod +x /build_ort.sh && /build_ort.sh
