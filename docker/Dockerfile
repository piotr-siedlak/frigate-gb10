FROM ghcr.io/blakeblackshear/frigate:stable

# Uninstall CPU-only onnxruntime shipped with Frigate
RUN pip3 uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true

# Install custom-built onnxruntime-gpu 1.24.4
# Built from source: CUDA 13.0, SM121 (GB10 Blackwell), Ubuntu 22.04 (GLIBC 2.35)
# Compatible with Frigate container (Debian 12, GLIBC 2.36)
COPY onnxruntime_gpu-1.24.4-cp311-cp311-linux_aarch64.whl /tmp/
RUN pip3 install --break-system-packages --no-deps --force-reinstall \
    /tmp/onnxruntime_gpu-1.24.4-cp311-cp311-linux_aarch64.whl && \
    rm /tmp/onnxruntime_gpu-1.24.4-cp311-cp311-linux_aarch64.whl && \
    rm -f /usr/local/lib/python3.11/dist-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-311-aarch64-linux-gnu.so

# Patch: remove LoraAdapter/AdapterFormat (not compiled without --enable_training)
RUN sed -i '/LoraAdapter/d'  /usr/local/lib/python3.11/dist-packages/onnxruntime/__init__.py 2>/dev/null; \
    sed -i '/AdapterFormat/d' /usr/local/lib/python3.11/dist-packages/onnxruntime/__init__.py 2>/dev/null; \
    true

# Patch: disable CUDA Graphs for yolo-generic (not all ops on CUDAExecutionProvider)
RUN sed -i 's/CudaGraphRunner\.is_model_supported(model_type)/False/' \
    /opt/frigate/frigate/detectors/detection_runners.py

# Patch: force ORT_ENABLE_BASIC — avoids graph fusions that use PTX unsupported on SM121
RUN sed -i 's/ONNXModelRunner\.is_cpu_complex_model(model_type)/True/' \
    /opt/frigate/frigate/detectors/detection_runners.py

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
