<div align="center">

# Unify Robot Actions in Camera Frame

<a href="https://arxiv.org/abs/2511.17001">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2511.17001-red?logo=arxiv" height="20" />
</a>
<a href="https://sii-dannyxsc.github.io/CalibAll/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-A9B5DF" height="20" />
</a>
<a href="LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg" height="20" />
</a>

**Sicheng Xie<sup>1,2,3</sup>, Lingchen Meng<sup>3</sup>, Zijie Diao<sup>1</sup>, Haidong Cao<sup>1</sup>, Zhiying Du<sup>1</sup>, Shuyuan Tu<sup>1</sup>, Jiaqi Leng<sup>1</sup>, Qiuyue Wang<sup>3</sup>, Mingsheng Li<sup>3</sup>, Shuai Bai<sup>3</sup>, [Zuxuan Wu](https://zxwu.azurewebsites.net/)<sup>1,2,&dagger;</sup>, Yu-Gang Jiang<sup>1,&dagger;</sup>**

<sup>1</sup> Institute of Trustworthy Embodied AI, Fudan University &nbsp; <sup>2</sup> Shanghai Innovation Institute &nbsp; <sup>3</sup> Qwen Team, Alibaba Inc.

<sup>&dagger;</sup> Corresponding authors

<video src="assets/demo.mp4" controls width="100%"><video>

</div>

## About

CalibAll is a training-free, robot-independent pipeline for **camera extrinsic calibration** and **automatic annotation** for offline robot datasets. Given video and joint angles, it estimates camera intrinsics and extrinsics via coarse-to-fine optimization (temporal PnP + differentiable rendering), then produces standardized TCP-pose actions, 2D/3D bounding boxes, segmentation masks.

**Current Supported robots:** Franka Panda, UR5e, xArm7, ALOHA, with Robotiq 85, Panda Hand, and xArm Gripper.

**Dataset format:** [LeRobot 2.1](https://github.com/huggingface/lerobot)

## Roadmap

| Status | Feature |
|:------:|---------|
| ✅ | Web-based interactive calibration pipeline |
| ✅ | Franka Panda (+ Robotiq 85 / Panda Hand) |
| ✅ | UR5e (+ Robotiq 85) |
| ✅ | xArm7 (+ xArm Gripper) |
| ✅ | ALOHA / ARX5 dual-arm |
| ⬜ | Automatic annotation pipeline (bbox, mask, keypoints, TCP-pose) |
| ⬜ | Docker support |
| ⬜ | More dataset formats |
| ⬜ | More robot embodiments |

## Prerequisites

Download model checkpoints into the `ckpt/` directory:

```bash
mkdir -p ckpt && cd ckpt

# CoTracker (98 MB)
mkdir -p cotracker && cd cotracker
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
cd ..

# DINOv2 (331 MB)
mkdir -p dinov2 && cd dinov2
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
cd ..

# SAM3 (6.5 GB) — from ModelScope (recommended) or Hugging Face
mkdir -p sam3
modelscope download --model facebook/sam3 --local_dir ./sam3
# Alternative: git clone https://huggingface.co/facebook/sam3

# MoGe (intrinsic estimation, ~1.2 GB)
hf download Ruicheng/moge-2-vitl-normal --local-dir ./moge
```

Expected structure:
```
ckpt/
├── cotracker/
│   └── scaled_offline.pth
├── dinov2/
│   └── dinov2_vitb14_pretrain.pth
├── moge/
│   └── model.pt
└── sam3/
    ├── sam3.pt
    ├── model.safetensors
    ├── config.json
    ├── tokenizer.json
    └── ...
```

### Download Demo Data

Download demo data:
```bash
pip install huggingface_hub
hf download dannyXSC/Caliball_demo --repo-type dataset --local-dir data/demo
```

## Installation

#### 1. Create Environment

```bash
conda create -n caliball python=3.12 -y
conda activate caliball
```

#### 2. Install PyTorch (CUDA 12.8)
```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

#### 3. Install CalibAll

```bash
git clone https://github.com/SII-dannyXSC/CalibAll.git
cd CalibAll
pip install -e .
pip install -r requirements.txt
```

#### 4. Install Special Dependencies

```bash
# nvdiffrast
pip install setuptools wheel ninja
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

# PyTorch3D
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.9+pt2.9.0cu128

# MoGe
pip install git+https://github.com/microsoft/MoGe.git
```

#### 5. Clone Third-Party Repos

```bash
mkdir -p third_party && cd third_party

# Co-Tracker (point tracking)
git clone https://github.com/facebookresearch/co-tracker
cd co-tracker && pip install -e . && cd ..

# SAM3 (segmentation)
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e . && cd ..

# DINOv2 (feature extraction)
git clone https://github.com/facebookresearch/dinov2

cd ..
```

## Quick Start

### Web UI
<video src="assets/usage.mp4" controls width="100%"><video>
Launch the interactive calibration interface:

```bash
python scripts/caliball_web.py
```

Open `http://127.0.0.1:8765` in your browser. The workflow:

1. **Configure** — enter dataset path, scan for cameras, select dataset config and robot type
2. **Annotate** — select frames, set tracking point (or auto-detect with DINOv2), draw masks (or auto-detect with SAM3)
3. **Pipeline** — automatic tracking, coarse PnP, and differentiable refinement
4. **Download** — get intrinsic/extrinsic matrices and calibration YAML

Optional flags:
```bash
python scripts/caliball_web.py --host 0.0.0.0 --port 8765 --device cuda
```

> **Note:** The first pipeline run takes ~10 minutes due to nvdiffrast CUDA kernel compilation. This only happens once — subsequent runs in the same session are fast.

## Project Structure

```
CalibAll/
├── src/caliball/
│   ├── algorithms/      # DINOv2 recognizer, CoTracker, SAM3, PnP solver
│   ├── config/          # Dataset YAML configs and calibration files
│   │   ├── dataset/     # Dataset loading profiles (for Web UI)
│   │   └── calibration/ # Intrinsic/extrinsic calibration results
│   ├── dataset/         # LeRobot dataset reader + state processors
│   ├── labeling/        # Pose calculation, mask rendering, orchestration
│   ├── pipeline/        # CoarseInit + Refinement pipeline
│   ├── rendering/       # NVDiffrast differentiable renderer
│   ├── robots/          # Robot registry (FK, mesh paths, composites)
│   ├── utils/           # Video I/O, visualization, mesh loading
│   └── web/             # Flask web app (routes, services, templates)
├── scripts/
│   ├── caliball_web.py  # Web UI entry point
│   ├── label.py         # Batch labeling pipeline
│   ├── visualize.py     # Result visualization
│   └── check_robot_urdf.py  # Robot mesh verification
├── third_party/         # Co-Tracker, SAM3, DINOv2, nvdiffrast, URDF
├── ckpt/                # Model checkpoints
└── assets/              # Images and demos
```

## Citation

```bibtex
@article{xie2024caliball,
    title={Unify Robot Actions in Camera Frame},
    author={Xie, Sicheng and Meng, Lingchen and Diao, Zijie and Cao, Haidong and Du, Zhiying and Tu, Shuyuan and Leng, Jiaqi and Wang, Qiuyue and Li, Mingsheng and Bai, Shuai and Wu, Zuxuan and Jiang, Yu-Gang},
    journal={arXiv preprint arXiv:2511.17001},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
