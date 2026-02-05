```
conda create -n caliball python=3.12
```

python 12 torch 2.9 cuda128


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
https://github.com/Daniella1/urdf_files_dataset.git
```

```
pip install setuptools wheel ninja
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

```
pip install git+https://github.com/microsoft/MoGe.git
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
pycollada
imageio[ffmpeg]
numpy==1.26.4
```

```
cd sam3/
pip install -e .
```

```
pytorch3d require torch version <= 2.4.1
https://github.com/facebookresearch/pytorch3d/discussions/1752
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.9+pt2.9.0cu128
```

```
conda install -n caliball ffmpeg -y
```