<div align="center" style="font-family: charter;">
<h1>Stable Offline Hand-Eye Calibration for any Robot with Just One Mark</h1>
<a href="https://arxiv.org/abs/2511.17001" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-CalibAll-red?logo=arxiv" height="20" />
</a>
<a href="https://sii-dannyxsc.github.io/CalibAll/" target="_blank">
    <img alt="Home Page CalibAll" src="https://img.shields.io/badge/📒_HomePage-CalibAll-ffc107?color=A9B5DF&logoColor=white" height="20" />
</a>
<!-- , , , , Jiaqi Leng, Zuxuan Wu, Yu-Gang Jiang -->
<div>
    Sicheng Xie<sup></sup>,</span>
    Lingchen Meng<sup></sup>,</span>
    Zhiying Du<sup></sup>,</span>
    Shuyuan Tu<sup></sup>,</span>
    Haidong Cao<sup></sup>,</span>
    Jiaqi Leng<sup></sup>,<br/>
    </span>
    <a href="https://zxwu.azurewebsites.net/" target="_blank">Zuxuan Wu</a><sup>&dagger;</sup>,</span>
    Yu-Gang Jiang<sup></sup></span>
</div>

<div>
    <sup>&dagger;</sup> Corresponding author&emsp;
</div>
</div>

## Method
[![Method overview](assets/method.jpg)](assets/method.jpg)


## Experiments
[![Demo animation](assets/franka+hand.gif)](assets/franka+hand.gif)

## Installation

```
conda create -n caliball python=3.12
```

```
pip install git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
```

```
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

```

```
pip install setuptools wheel ninja
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

```
cd third_party
git clone https://github.com/facebookresearch/co-tracker
cd co-tracker
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
```

```
cd third_party
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

```
cd third_party
git clone https://github.com/Daniella1/urdf_files_dataset.git
```

```
pip install setuptools wheel ninja
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

```
pip install git+https://github.com/microsoft/MoGe.git
```

```
cd third_party
git clone https://github.com/facebookresearch/dinov2
```

```
roboticstoolbox-python
opencv-python
tensorflow
tensorflow-datasets
loguru
multipledispatch
omegaconf
debugpy
pycocotools
pycollada
imageio[ffmpeg]
decord
numpy==1.26.4
```

```
pytorch3d require torch version <= 2.4.1
https://github.com/facebookresearch/pytorch3d/discussions/1752
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.9+pt2.9.0cu128
```

```
conda install -n caliball ffmpeg -y
```