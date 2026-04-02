#!/bin/bash
set -e
cd /build
# Use pre-downloaded Eigen to bypass FetchContent hash mismatch
./build.sh \
    --use_cuda \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/lib/aarch64-linux-gnu \
    --build_wheel \
    --parallel \
    --config Release \
    --skip_tests \
    --allow_running_as_root \
    --cmake_extra_defines \
        CMAKE_CUDA_ARCHITECTURES=121 \
        FETCHCONTENT_SOURCE_DIR_EIGEN=/tmp/eigen-src
