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



https://github.com/user-attachments/assets/c7acb2e6-7d7b-4be9-b3d7-724e7c808be2



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
| ✅ | Automatic annotation pipeline (bbox, mask, keypoints, TCP-pose) |
| ✅ | Docker support |
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

### Docker (Recommended)

Requires [Docker](https://docs.docker.com/get-docker/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (GPU passthrough). No host-side CUDA toolkit is needed — the image ships its own CUDA 12.8 runtime — but the host **NVIDIA driver must support CUDA 12.8** (driver ≥ 570; check the `CUDA Version` shown by `nvidia-smi`).

#### Option A — Pull pre-built image (fastest)

Pre-built images are published to GitHub Container Registry:

```bash
git clone https://github.com/SII-dannyXSC/CalibAll.git
cd CalibAll

# Pull from GHCR (requires the package to be public, or `docker login ghcr.io`)
docker compose pull
docker compose up -d --no-build
```

Image: `ghcr.io/sii-dannyxsc/caliball:cuda12.8`

> The image bundles all code and dependencies, but **not** the model
> checkpoints — download them into `ckpt/` as described in
> [Prerequisites](#prerequisites). They are mounted into the container at runtime.
>
> If `docker compose pull` returns `unauthorized`, the package is still private:
> make it public under GitHub → Packages → caliball → Package settings →
> Change visibility, or run `docker login ghcr.io` first.

#### Option B — Build locally (~1 hour first time)

```bash
git clone https://github.com/SII-dannyXSC/CalibAll.git
cd CalibAll
docker compose build
docker compose up -d --no-build
```

#### Usage

Open `http://127.0.0.1:8765` in your browser.

```bash
# View logs
docker compose logs -f

# Shell inside the container
docker compose exec caliball bash

# Health check
docker compose exec caliball bash scripts/docker_setup_env.sh

# Stop
docker compose down
```

Checkpoints (`ckpt/`) and datasets (`data/`) are bind-mounted from the host — download them on the host as described in [Prerequisites](#prerequisites), and the container will see them immediately.

## Quick Start

### Web UI
https://github.com/user-attachments/assets/c25c8711-f980-44c4-ae04-8754114ff275


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


### Writing Dataset Configs

Task configs live in `src/caliball/config/`. A config connects four things: the LeRobot dataset path, the robot FK model, the calibration entry, and the annotation options.

Before labeling, save the calibration YAML downloaded from the Web UI into `src/caliball/config/calibration/`. The task config must set `calib_dataset_name` to the calibration YAML's `dataset` value, or to one of its `aliases`.

Full example:

```yaml
# Calibration lookup key. This must match a calibration YAML under
# src/caliball/config/calibration/ via its `dataset` or `aliases`.
calib_dataset_name: rdt

# Robot FK / mesh model from caliball.robots registry.
robot_type: aloha_v1

# Local LeRobot dataset path = ${base_path}/${dataset_name}.
dataset_name: aloha_lerobot
base_path: /path/to/data/demo

dataset:
  _target_: caliball.dataset.lerobot_dataset.LeRobotDataset
  repo_id: ${base_path}/${dataset_name}

  # Parquet columns used as robot joint states.
  state_keys: observation.state

  # Optional: decode only selected video streams.
  # decode_video_keys:
  #   - observation.images.main

tf:
  # Optional Euler XYZ angle offset, in degrees, applied to output TCP rotation.
  grasp_point_rotation_align: [0.0, 90.0, 0.0]
  # Optional fixed gripper mounting yaw offset around the flange Z axis.
  gripper_mount_yaw_deg: 0.0

label:
  output_dir: ./label_out/aloha/${dataset_name}
  calib_dataset_name: ${calib_dataset_name}

  # Video feature names to annotate. They must exist in meta/info.json.
  camera_names:
    - observation.images.main

  eef_rotation_type: euler_xyz

  # Number of meshes per arm treated as the arm body; remaining meshes are gripper.
  arm_mesh_num: 7

  # Optional labeling range and runtime options.
  episode_start: 0
  max_episodes: 1
  skip_mask: false
  device: cuda
```

Important fields:

- `calib_dataset_name`: key used to look up intrinsics/extrinsics from `src/caliball/config/calibration/*.yaml`. It must match either `dataset` or an entry in `aliases` from the calibration YAML exported by the Web UI.
- `robot_type`: robot FK and mesh model from the registry. Print available names with `python -c "from caliball.robots import list_robots; print(list_robots())"`.
- `base_path` and `dataset_name`: combined into `dataset.repo_id`, the local LeRobot dataset directory.
- `dataset.state_keys`: parquet columns used as joint states. Use names from the dataset `meta/info.json` features, for example `observation.state`.
- `tf.grasp_point_rotation_align`: optional Euler XYZ angle offset, in degrees, right-multiplied to the output grasp/TCP rotation.
- `tf.gripper_mount_yaw_deg`: optional fixed yaw offset for grippers mounted with a Z-axis rotation relative to the arm flange.
- `label.camera_names`: video feature names to annotate. They must exist in `meta/info.json`.
- `label.arm_mesh_num`: number of per-arm meshes treated as arm body; remaining meshes are treated as gripper meshes for separated masks and boxes.

For a new dataset, first check `meta/info.json` for `state_keys` and camera names, then run a small smoke test:

```bash
python scripts/label.py \
  --config src/caliball/config/demo_aloha.yaml \
  --format json \
  --max_episodes 1 \
  --skip_mask
```

If the smoke test passes, remove `--skip_mask` to render full masks.

### Batch Labeling and Visualization

After calibration YAMLs are available under `src/caliball/config/calibration/`, use `scripts/label.py` to generate automatic annotations from a dataset config. The script supports two output formats:

- `json`: episode JSON files plus pickled `LabelData`.
- `lerobot`: a LeRobot-style dataset with annotation columns in parquet, copied meta files, and video symlinks.

Example with the local ALOHA demo dataset:

```bash
# JSON annotations
python scripts/label.py \
  --config src/caliball/config/demo_aloha.yaml \
  --format json \
  --max_episodes 1
```

```bash
# LeRobot annotations
python scripts/label.py \
  --config src/caliball/config/demo_aloha.yaml \
  --format lerobot \
  --max_episodes 1
```

For quick geometry checks without rendering masks, add `--skip_mask`. This only writes TCP pose, keypoint, and bbox-related fields that do not require mesh rasterization.

Visualize JSON annotations:

```bash
python scripts/visualize.py \
  --json_dir ./label_out/aloha/aloha_lerobot \
  --task_path ./data/demo/aloha_lerobot \
  --cameras observation.images.main \
  --output_dir ./label_out/aloha/aloha_lerobot_json_vis \
  --fps 30
```

Visualize LeRobot annotations:

```bash
python scripts/visualize.py \
  --input_format lerobot \
  --dataset_dir ./label_out/aloha/aloha_lerobot \
  --output_dir ./label_out/aloha/aloha_lerobot_vis \
  --episodes 0 \
  --fps 30
```

Useful visualization flags:

- `--first_frame_only`: export still images instead of a full video in JSON mode.
- `--no_mask`, `--no_bbox`, `--no_point`, `--no_axes`: hide selected overlays.
- `--alpha`: adjust mask opacity.

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
│   ├── caliball_web.py      # Web UI entry point
│   ├── docker_setup_env.sh  # Container health check / dependency repair
│   ├── label.py             # Batch labeling pipeline
│   ├── visualize.py         # Result visualization
│   └── check_robot_urdf.py  # Robot mesh verification
├── Dockerfile               # CUDA 12.8 dev image
├── docker-compose.yml       # Web UI service (GPU, ports, volumes)
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
