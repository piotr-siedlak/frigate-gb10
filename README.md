# Frigate NVR on Dell GB10 / DGX Spark

GPU-accelerated Frigate NVR running on **Dell GB10 / NVIDIA DGX Spark** (arm64, Blackwell SM_121, CUDA 13.0).

This guide covers building a custom `onnxruntime-gpu` wheel from source, patching the Frigate Docker image, and running YOLOv9-based object detection fully on the GPU.

---

## Hardware & Software

| Component | Version |
|---|---|
| Hardware | Dell GB10 / NVIDIA DGX Spark |
| Architecture | aarch64 / arm64 |
| GPU | GB10 Blackwell, SM_121 |
| CUDA | 13.0 |
| Driver | 580.95.05 |
| OS (host) | Ubuntu 24.04 |
| Frigate | 0.17.1 (`ghcr.io/blakeblackshear/frigate:stable`) |
| Frigate base image | Debian 12 (GLIBC 2.36), Python 3.11.2 |
| onnxruntime-gpu | 1.24.4 (built from source) |
| cuDNN | 9.20.0 |

---

## Why a custom build?

The Frigate 0.17.x ONNX detector requires `OrtCompileApiFlags` — a symbol not present in onnxruntime ≤ 1.18.1.  
The Microsoft nightly wheels (`dev20260331005+`) are compiled for CUDA 13.1+, which causes `cudaErrorUnsupportedPtxVersion` on a CUDA 13.0 driver.  
Solution: build onnxruntime 1.24.4 from source targeting SM_121 + CUDA 13.0.

---

## Repository structure

```
.
├── README.md
├── docker-compose.yml
├── docker/
│   ├── Dockerfile                          # Custom Frigate image
│   └── onnxruntime_gpu-1.24.4-cp311-cp311-linux_aarch64.whl   # Pre-built (Git LFS)
├── config/
│   ├── config.yml                          # Frigate config (edit RTSP URLs)
│   └── model_cache/
│       └── yolov9-c-640.onnx              # YOLOv9-c 640px model (Git LFS)
├── mqtt/
│   └── config/
│       └── mosquitto.conf
├── ort-build/
│   ├── Dockerfile.builder                  # Builder image for onnxruntime
│   ├── build_ort.sh                        # Build script
│   └── eigen.zip                           # Eigen source (Git LFS, required by cmake)
└── cuda-libs/
    └── cudnn/                              # cuDNN .so files copied from host (see Step 3)
```

---

## Step 1 — Prerequisites

Verify the host environment:

```bash
nvidia-smi            # should show GB10, Driver 580.x, CUDA 13.0
uname -m              # aarch64
python3 --version     # 3.11.x (host, needed for model build only)
docker --version
docker compose version
```

Verify CUDA toolkit is installed at `/usr/local/cuda-13.0`:

```bash
ls /usr/local/cuda-13.0/targets/sbsa-linux/lib/libcudart*
```

---

## Step 2 — Clone this repository

```bash
git clone https://github.com/piotr-siedlak/frigate-gb10.git
cd frigate-gb10
```

Git LFS is required for the large binary files (`.whl`, `.onnx`, `.zip`):

```bash
# Install git-lfs if not present
curl -sL https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-arm64-v3.6.1.tar.gz \
  | tar xz -C /tmp && sudo cp /tmp/git-lfs-3.6.1/git-lfs /usr/local/bin/
git lfs install
git lfs pull
```

---

## Step 3 — Copy cuDNN libraries from host

The nvidia-container-runtime only injects driver libs (libcuda.so, libnvcuvid), **not** CUDA toolkit or cuDNN. Both must be mounted manually.

Copy only the cuDNN `.so` files (not the full `/usr/lib/aarch64-linux-gnu/` — that causes glibc symbol conflicts):

```bash
mkdir -p cuda-libs/cudnn
cp /usr/lib/aarch64-linux-gnu/libcudnn*.so* cuda-libs/cudnn/
ls cuda-libs/cudnn/
# libcudnn.so  libcudnn.so.9  libcudnn.so.9.20.0  libcudnn_adv.so ...
```

---

## Step 4 — Configure cameras

### All places where you must set your camera IPs

| File | Location | What to replace |
|---|---|---|
| `config/config.yml` | `go2rtc.streams` → every `rtsp://` URL | `YOUR_CAMERA_IP` → IP address of each camera |
| `config/config.yml` | `go2rtc.streams` → every `rtsp://` URL | `YOUR_PASSWORD` → RTSP password (same for all Hikvision cameras if using a shared account) |
| `config/config.yml` | `cameras` → `ffmpeg.inputs` | No change needed — these reference `127.0.0.1:8554` (internal go2rtc), not the cameras directly |

### Stream URL format (Hikvision)

```
rtsp://USERNAME:PASSWORD@CAMERA_IP:554/Streaming/channels/101   ← main stream (high res, record)
rtsp://USERNAME:PASSWORD@CAMERA_IP:554/Streaming/channels/102   ← sub stream  (low res, detect)
```

### How to add a camera

1. Add two entries to `go2rtc.streams` (one `_main`, one `_sub`):

```yaml
go2rtc:
  streams:
    mycam_main:
      - rtsp://admin:YOUR_PASSWORD@192.168.1.100:554/Streaming/channels/101
    mycam_sub:
      - rtsp://admin:YOUR_PASSWORD@192.168.1.100:554/Streaming/channels/102
```

2. Add a matching camera block to `cameras` (name must match the stream prefix):

```yaml
cameras:
  mycam:                          # ← must match "mycam_main" / "mycam_sub" prefix
    enabled: true
    ffmpeg:
      inputs:
        - path: rtsp://127.0.0.1:8554/mycam_main
          input_args: preset-rtsp-restream
          roles: [record]
        - path: rtsp://127.0.0.1:8554/mycam_sub
          input_args: preset-rtsp-restream
          roles: [detect]
    detect:
      enabled: true
      width: 1280
      height: 720
      fps: 10
```

The `_main` stream is used for recording, `_sub` for detection (lower resolution = faster inference).

---

## Step 5 — Build and start

```bash
docker compose build
docker compose up -d
```

Check that Frigate is healthy:

```bash
docker ps               # STATUS should show (healthy)
curl http://localhost:5000/api/version
# 0.17.1-xxxxxxx
```

Open the UI at **http://localhost:5000**

---

## Step 6 — Verify GPU detection

```bash
docker exec frigate python3 -c "
import onnxruntime as ort
print('Version:', ort.__version__)
print('Providers:', ort.get_available_providers())
"
# Version: 1.24.4
# Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Check ONNX model loaded on GPU:

```bash
docker logs frigate 2>&1 | grep -i onnx
# ONNX: loading /config/model_cache/yolov9-c-640.onnx
# ONNX: /config/model_cache/yolov9-c-640.onnx loaded
```

---

## Optional — Build onnxruntime from source

The pre-built `.whl` in this repo is ready to use. Follow this section only if you need to rebuild it (e.g. different CUDA version).

### Requirements

- Host with CUDA 13.0 and Docker
- ~20 GB free disk space
- Build time: ~90 minutes

### Download Eigen (required)

onnxruntime 1.24.4 fetches Eigen via cmake FetchContent. GitLab changed their tarball format, breaking the SHA1 checksum. Use the GitHub mirror instead:

```bash
cd ort-build
# eigen.zip is already in this repo (Git LFS)
# SHA1: 05b19b49e6fbb91246be711d801160528c135e34
sha1sum eigen.zip
```

If you need to re-download:

```bash
curl -L https://github.com/eigen-mirror/eigen/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33.zip \
  -o eigen.zip
```

### Build

```bash
cd ort-build
docker build -f Dockerfile.builder -t ort-builder .
docker run --rm --gpus all \
  -v $(pwd)/output:/output \
  ort-builder \
  bash -c "cp /build/build/Linux/Release/dist/*.whl /output/"
```

Output wheel will be at `ort-build/output/onnxruntime_gpu-1.24.4-cp311-cp311-linux_aarch64.whl`.

Copy it to `docker/`:

```bash
cp ort-build/output/onnxruntime_gpu-1.24.4-cp311-cp311-linux_aarch64.whl docker/
```

---

## Optional — Build YOLOv9 ONNX model

The model `config/model_cache/yolov9-c-640.onnx` is included (Git LFS). To rebuild natively on arm64:

```bash
# Install PyTorch CPU (arm64 wheel)
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Clone and export
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt

python3 export.py \
  --weights yolov9-c.pt \
  --include onnx \
  --imgsz 640 \
  --batch-size 1 \
  --device cpu
# outputs yolov9-c.onnx (~98 MB)
```

> Note: `onnx-simplifier` fails to compile on arm64 — skip it. Frigate works fine with the unsimplified model.

---

## Tuning the number of detectors

Each detector is a separate process with its own ONNX `InferenceSession` running on the GPU.
Adding more detectors increases parallel throughput — up to the point where GPU compute is saturated.

### Benchmark results (GB10, YOLOv9-c 640px)

| Detectors | ms / frame | Total FPS | GPU util |
|:---------:|:----------:|:---------:|:--------:|
| 1         | 32 ms      | 31 FPS    | ~40 %    |
| **2**     | **28 ms**  | **71 FPS**| ~50 %    |
| 4         | 93 ms      | 43 FPS    | ~94 %    |

**2 detectors is the optimal point** for this GPU + model combination.  
At 4 detectors the GPU is saturated and sessions start queuing — total throughput drops below the 2-detector result.

Maximum demand with 10 cameras × 10 FPS = 100 FPS. Two detectors cover ~71 FPS which is sufficient for most real-world scenarios where not all cameras have simultaneous motion.

### How to change

Edit `config/config.yml` — add or remove entries under `detectors`. Each entry must have a unique name:

```yaml
# 1 detector
detectors:
  onnx0:
    type: onnx

# 2 detectors (recommended)
detectors:
  onnx0:
    type: onnx
  onnx1:
    type: onnx

# 4 detectors (GPU saturated on GB10 — not recommended)
detectors:
  onnx0:
    type: onnx
  onnx1:
    type: onnx
  onnx2:
    type: onnx
  onnx3:
    type: onnx
```

Then restart Frigate:

```bash
docker compose restart frigate
```

Verify detectors are running and check inference speed:

```bash
curl -s http://localhost:5000/api/stats | python3 -c "
import sys, json
for name, d in json.load(sys.stdin)['detectors'].items():
    print(name, d['inference_speed'], 'ms')
"
```

Each detector allocates ~619 MiB of GPU memory. Monitor with:

```bash
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader
```

---

## How the patches work

### 1. onnxruntime `.so` filename conflict

Frigate's base image ships `onnxruntime_pybind11_state.cpython-311-aarch64-linux-gnu.so` (v1.18.1).  
The custom wheel installs `onnxruntime_pybind11_state.so` (v1.24.4).  
Python always prefers the ABI-tagged filename, so it was loading the old v1.18.1 `.so` — which lacks `OrtCompileApiFlags`.

**Fix in Dockerfile**: delete the old ABI-tagged file after installing the new wheel.

### 2. CUDA toolkit not injected by nvidia-runtime

The nvidia-container-runtime only injects driver libs. `libcublas`, `libcudart`, `libcufft`, etc. are not available inside the container.

**Fix**: mount `/usr/local/cuda-13.0/targets/sbsa-linux/lib` as `/usr/local/cuda/lib64:ro` and set `LD_LIBRARY_PATH`.

### 3. CUDA Graphs — not all ops on CUDAExecutionProvider

Frigate's `CudaGraphRunner` enables CUDA Graphs for performance. Some ops in YOLOv9 fall back to CPU, which is incompatible with CUDA Graphs.

**Fix**: patch `detection_runners.py` to always return `False` for `CudaGraphRunner.is_model_supported()`.

### 4. Graph fusions — PTX unsupported on SM_121

Graph fusions enabled at `ORT_ENABLE_ALL` use PTX intrinsics not available on SM_121 (Blackwell).

**Fix**: patch to always use `ORT_ENABLE_BASIC` optimization level.

### 5. ffmpeg preset-nvidia — pthread assertion failure

`preset-nvidia` triggers a pthread assertion on the GB10 (CVE-2022-48434 related).

**Fix**: use `-hwaccel cuda -threads 1` directly instead of `preset-nvidia`.

---

## Ports

| Port | Service |
|---|---|
| 5000 | Frigate web UI + API |
| 8971 | Frigate HTTPS UI |
| 8554 | go2rtc RTSP |
| 8555 | go2rtc WebRTC (TCP+UDP) |
| 1883 | MQTT |
| 9001 | MQTT WebSocket |

---

## Tracked objects (COCO-80)

| Label | COCO id | Notes |
|---|---|---|
| person | 0 | |
| car | 2 | also covers bus (5) and truck (7) via labelmap remap |
| bicycle | 1 | |
| motorcycle | 3 | |
| dog | 16 | |
| cat | 15 | |

---

## Build walkthrough — what actually happened

This section documents every problem encountered and how it was solved, in order.

---

### Phase 1 — Environment discovery

```bash
nvidia-smi
# GB10, Driver 580.95.05, CUDA Version: 13.0
uname -m
# aarch64
python3 --version
# Python 3.11.x
```

No QEMU/binfmt is installed on DGX OS — `--platform linux/amd64` emulation is unavailable. Everything must run natively on arm64.

---

### Phase 2 — YOLOv9 ONNX model build

The original guide used `--platform linux/amd64` to export YOLOv9. That fails without QEMU.

**Fix**: Build natively with PyTorch CPU wheels for arm64.

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/WongKinYiu/yolov9.git && cd yolov9
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
python3 export.py --weights yolov9-c.pt --include onnx --imgsz 640 --batch-size 1 --device cpu
```

`onnx-simplifier` was removed from the command — it requires a C++ extension that doesn't compile on arm64. Frigate works fine with the unsimplified model.

Output: `yolov9-c.onnx` (98 MB).

---

### Phase 3 — Initial Frigate setup

Created directory structure:

```
~/frigate/
├── docker/Dockerfile
├── docker-compose.yml
├── config/config.yml
├── config/model_cache/yolov9-c-640.onnx
├── mqtt/config/mosquitto.conf
└── cuda-libs/cudnn/
```

First attempt used the MS nightly onnxruntime wheel (`dev20260331005`). It built but crashed immediately:

```
cudaErrorUnsupportedPtxVersion: the provided PTX was compiled with an unsupported toolchain
```

The nightly wheel was compiled for CUDA 13.1+. Our driver (580.95.05) only supports CUDA 13.0. We must build from source.

---

### Phase 4 — Building onnxruntime from source

#### Problem: cmake too old

The system cmake was 3.22. onnxruntime 1.24.4 requires ≥ 3.26.

```bash
pip3 install cmake==3.29.6
```

#### Problem: `build.py` not found

onnxruntime's entry point is `./build.sh`, not `./build.py`.

Fix in `build_ort.sh`: use `./build.sh` instead of `python3 build.py`.

#### Problem: Eigen FetchContent SHA1 mismatch

onnxruntime downloads Eigen via cmake FetchContent from GitLab. GitLab changed their tarball format, breaking the SHA1 checksum that's hardcoded in onnxruntime's cmake.

Fix: download from the GitHub mirror and use `FETCHCONTENT_SOURCE_DIR_EIGEN`:

```bash
curl -L https://github.com/eigen-mirror/eigen/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33.zip -o eigen.zip
# SHA1: 05b19b49e6fbb91246be711d801160528c135e34  ← matches onnxruntime 1.24.4 cmake
```

In `build_ort.sh`:
```bash
--cmake_extra_defines FETCHCONTENT_SOURCE_DIR_EIGEN=/tmp/eigen-src
```

#### Problem: `longlong4` deprecated in CUDA 13.0

onnxruntime 1.21.1 uses `longlong4` in `attention_impl.h`, which CUDA 13.0 deprecated as an error.

Fix: switch to onnxruntime **1.24.4**, which already removed all uses of `longlong4`.

#### Problem: `packaging` module missing

At the `bdist_wheel` step:
```
ModuleNotFoundError: No module named 'packaging'
```

Fix in `Dockerfile.builder`:
```dockerfile
RUN pip3 install cmake==3.29.6 numpy==1.26.4 psutil wheel packaging setuptools
```

#### Build command (final working)

```bash
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
```

Build time: ~90 minutes. Output: `onnxruntime_gpu-1.24.4-cp311-cp311-linux_aarch64.whl` (53 MB).

---

### Phase 5 — CUDA toolkit and cuDNN mounts

**Problem**: `libcublas`, `libcudart`, `libcufft` etc. are not available inside the Frigate container. The nvidia-container-runtime only injects driver libs (`libcuda.so`, `libnvcuvid`).

**Fix**: mount the CUDA 13.0 toolkit directory:

```yaml
volumes:
  - /usr/local/cuda-13.0/targets/sbsa-linux/lib:/usr/local/cuda/lib64:ro
```

**Problem with cuDNN**: first attempt mounted the full `/usr/lib/aarch64-linux-gnu/` to get `libcudnn`. That caused a `__tunable_is_initialized` symbol conflict with glibc inside the container.

**Fix**: copy only the cuDNN `.so` files to `~/frigate/cuda-libs/cudnn/` and mount that directory:

```bash
cp /usr/lib/aarch64-linux-gnu/libcudnn*.so* ~/frigate/cuda-libs/cudnn/
```

```yaml
volumes:
  - ./cuda-libs/cudnn:/usr/local/cudnn/lib:ro
environment:
  - LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cudnn/lib
```

---

### Phase 6 — `OrtCompileApiFlags` import error

After installing the custom wheel into the Frigate container:

```
ImportError: cannot import name 'OrtCompileApiFlags' from 'onnxruntime.capi._pybind_state'
```

**Root cause**: Two `.so` files coexisted after `pip3 install`:

- `onnxruntime_pybind11_state.cpython-311-aarch64-linux-gnu.so` — old v1.18.1 from Frigate base image  
  (not removed by `pip3 uninstall` because Frigate installs it outside pip's tracking)
- `onnxruntime_pybind11_state.so` — our new v1.24.4

Python always prefers the ABI-tagged `cpython-311` filename over the plain `.so`. It loaded v1.18.1, which lacks `OrtCompileApiFlags`.

Confirmed: `strings onnxruntime_pybind11_state.so | grep OrtCompileApiFlags` → 19 matches. The symbol was there, just not being loaded.

**Fix** in Dockerfile, after `pip3 install`:

```dockerfile
RUN pip3 install --break-system-packages --no-deps --force-reinstall \
    /tmp/onnxruntime_gpu-1.24.4-cp311-cp311-linux_aarch64.whl && \
    rm /tmp/onnxruntime_gpu-1.24.4-cp311-cp311-linux_aarch64.whl && \
    rm -f /usr/local/lib/python3.11/dist-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-311-aarch64-linux-gnu.so
```

---

### Phase 7 — CUDA Graphs and graph fusion patches

Even with the import fixed, inference would crash or silently fall back to CPU without two more patches:

**Patch 1 — Disable CUDA Graphs**

YOLOv9 has ops that fall back to CPU, which is incompatible with CUDA Graphs:

```dockerfile
RUN sed -i 's/CudaGraphRunner\.is_model_supported(model_type)/False/' \
    /opt/frigate/frigate/detectors/detection_runners.py
```

**Patch 2 — Force `ORT_ENABLE_BASIC`**

`ORT_ENABLE_ALL` (default) activates graph fusions that use PTX intrinsics unsupported on SM_121 (Blackwell). The `is_cpu_complex_model` check in Frigate normally only enables basic optimization for large CPU models; patching it to always return True forces basic optimization for all ONNX models:

```dockerfile
RUN sed -i 's/ONNXModelRunner\.is_cpu_complex_model(model_type)/True/' \
    /opt/frigate/frigate/detectors/detection_runners.py
```

---

### Phase 8 — ffmpeg pthread fix

Using `preset-nvidia` in ffmpeg caused a pthread assertion failure on GB10 (related to CVE-2022-48434 — a race condition in FFmpeg's thread pool). Confirmed fix from [Frigate issue #21002](https://github.com/blakeblackshear/frigate/issues/21002):

```yaml
ffmpeg:
  hwaccel_args:
    - -hwaccel
    - cuda
    - -threads
    - "1"
```

`-threads 1` disables multi-threaded demuxing, avoiding the race condition entirely.

---

### Final result

```bash
docker exec frigate python3 -c "
import onnxruntime as ort
print(ort.__version__)          # 1.24.4
print(ort.get_available_providers())  # ['CUDAExecutionProvider', 'CPUExecutionProvider']
"

docker logs frigate 2>&1 | grep onnx
# ONNX: loading /config/model_cache/yolov9-c-640.onnx
# ONNX: /config/model_cache/yolov9-c-640.onnx loaded

docker ps
# STATUS: Up N minutes (healthy)
```

---

## References

- [Frigate issue #21002](https://github.com/blakeblackshear/frigate/issues/21002) — GB10 pthread fix (`-threads 1`)
- [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.4)
- [YOLOv9](https://github.com/WongKinYiu/yolov9)
