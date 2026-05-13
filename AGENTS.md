## 数据
原始数据位置：
1. /cpfs01/user/wenji.zj/dataspace/Data4QwenVLA/RoboMIND_lerobot_v2.1
2. /cpfs01/data/shared/Group-m6/yuanhaoqi.yhq/data/rdt_aloha_lerobot2.1
3. /cpfs01/user/wenji.zj/dataspace/Robointer_updated_v2/Annotation_with_action_lerobotv21/lerobot_droid_anno

本地数据目录：/cpfs02/user/xiesicheng.xsc/project/CalibAll/data/caliball_data
- 数据1、2 已有
- 数据3 尚未同步（数据量较大，需要特殊处理）


## Labeling 数据格式

路径：`src/caliball/pipeline/label_data.py`

### 数据结构

```
LabelData                        # episode 级
  dataset_name: str
  episode_id: str
  arm_names: list[str]           # 如 ["left"] 或 ["left", "right"]
  dataset_root: Optional[str]
  cameras: dict[str, list[FrameLabel]]   # camera_name -> 帧列表

FrameLabel                       # 单帧
  index: int
  arms: dict[str, ArmLabel]      # arm_name -> 单臂数据

ArmLabel                         # 单臂 grip point
  uv: list[int]                  # [u, v]
  xyz_euler_g: list[float]       # [x, y, z, rx, ry, rz, g]       (7,)
  xyz_quat_g: list[float]        # [x, y, z, qw, qx, qy, qz, g]  (8,)
  xyz_mat_g: list[float]         # [x, y, z, r00..r22, g]         (13,)
  uvd: list[float]               # [u, v, d]
  mask_with_gripper: Optional[dict]     # COCO RLE
  mask_without_gripper: Optional[dict]  # COCO RLE
  mask_gripper: Optional[dict]          # COCO RLE（仅夹爪部分）
  bbox_with_gripper: Optional[list[int]]    # [x1, y1, x2, y2]
  bbox_without_gripper: Optional[list[int]] # [x1, y1, x2, y2]
  bbox_gripper: Optional[list[int]]         # [x1, y1, x2, y2]
  is_placeholder: bool
```

### 序列化
- `label_data.save(path)` → pickle
- `LabelData.load(path)` → LabelData

### 工厂方法
- `LabelData.from_label_episode(info_list, dataset_name, episode_id, camera_name, arm_name="left")`
- `LabelData.from_agents_grip_episode(agents_list, dataset_name, episode_id, camera_name)`
- `LabelData.from_dual_label_episode(left_info_list, right_info_list, ..., camera_name)`

### 增量添加相机
- `label_data.add_camera_from_label_episode(info_list, camera_name, arm_name="left")`


## 机器人格式

路径：`src/caliball/robot/`

### 目录结构

```
robot/
  base.py                   # BaseTF（抽象基类）
  rtb.py                    # RoboticsToolBoxTF（RTB 臂基类）
  arm_gripper_composite.py  # ArmGripperCompositeTF（臂+外挂夹爪基类）
  arm/
    franka.py               # FrankaTF
    ur5e.py                 # Ur5eTF
    fr3.py                  # Fr3ArmTF
  gripper/
    robotiq.py              # RobotiqTF
    fr3_hand.py             # Fr3PandaGripperTF
  composite/
    ur5e_robotiq.py         # Ur5eRobotiqTF
    franka_robotiq.py       # FrankaRobotiqTF
    franka_hand.py          # FrankaPandaHandTF
    xarm.py                 # XArm7WithGripperTF
  dual_arm/
    dual_arm_base.py        # DualArmTF（双臂合成基类，已实现）
    aloha.py                # AlohaCobotMagicTF（双臂，基于 DualArmTF）
    arx5.py                 # Arx5RobotwinTF（双臂，基于 DualArmTF）
  urdf/
    aloha.py                # AlohaCobotMagic（双臂整体 URDF）
    arx5_robotwin.py        # Arx5Robotwin（双臂整体 URDF）
    ...
```

### 统一接口

所有类统一实现以下三个方法，输入单帧 `q: (n_joints,)`：

```python
fkine_eef(q)           -> (n_arms, 4, 4)          # 臂末端位姿（不含夹爪偏移）
fkine_gripper(q)       -> (n_arms, 4, 4)          # 夹爪闭合时末端位姿（含 grasp_point_R_align）
_fkine_gripper_raw(q)  -> (n_arms, 4, 4)          # 子类实现：不含旋转对齐
fkine_all(q)           -> (n_arms, n_links, 4, 4) # 所有 link 变换矩阵
gripper_scalar(q)      -> float                    # 单臂：末位关节
gripper_scalars(q)     -> (n_arms,)               # 每臂夹爪标量
```

`fkine` 保留为 `fkine_gripper` 的别名。

**模板方法模式**：`BaseTF.fkine_gripper` 是非抽象模板，自动在 `_fkine_gripper_raw` 结果上右乘 `grasp_point_R_align`（若已设置）。子类实现 `_fkine_gripper_raw`，不需要手动处理对齐。

各类型行为：
- **纯臂**：`_fkine_gripper_raw` = `fkine_eef`
- **臂+外挂夹爪**：`fkine_eef` = 臂末端；`_fkine_gripper_raw` = 夹爪闭合指尖
- **一体 URDF**：`fkine_eef` = 指定 eef link；`_fkine_gripper_raw` = 指尖（强制 q_gripper=0，使位置与夹爪开闭无关）
- **双臂（DualArmTF）**：n_arms=2，顺序 [左, 右]；内部持有两个独立单臂 TF，通过 n_left_joints 切分 q

### grasp_point_rotation_align

所有顶层 TF 类（`Arx5RobotwinTF`、`AlohaCobotMagicTF`、`FrankaPandaHandTF`、`FrankaRobotiqTF`、`Ur5eRobotiqTF`、`XArm7WithGripperTF`）的 `__init__` 均接受：

```python
grasp_point_rotation_align=None   # None | [rx,ry,rz]（度，euler_xyz）| 3×3 嵌套列表
```

设置后，`fkine_gripper` 返回值的旋转部分会右乘该矩阵：`R_out = R_fk @ R_align`。

**约定方向**：Z 轴指向夹爪前进方向，Y 轴为夹爪闭合方向，X = Z×Y（右手系）。

YAML 配置位置：`tf:` 段（不是 `label:` 段）：

```yaml
tf:
  _target_: src.caliball.robot.dual_arm.arx5.Arx5RobotwinTF
  names: ${robot.names}
  eef_name: ${robot.eef_name}
  grasp_point_rotation_align: [0.0, 90.0, 0.0]   # euler_xyz，度
```

### 双臂实现：DualArmTF

路径：`src/caliball/robot/dual_arm/dual_arm_base.py`

```python
class DualArmTF(BaseTF):
    n_arms = 2

    def __init__(self, left: BaseTF, right: BaseTF, n_left_joints: int):
        self.left = left
        self.right = right
        self.n_left_joints = n_left_joints

    def fkine_eef(self, q):           # -> (2, 4, 4)
    def _fkine_gripper_raw(self, q):  # -> (2, 4, 4)
    def fkine_all(self, q):           # -> (2, n_links, 4, 4)
    def gripper_scalars(self, q):     # -> (2,)
```

### ARX5 双臂（Arx5RobotwinTF / Arx5ArmTF）

路径：`src/caliball/robot/dual_arm/arx5.py`

- `Arx5ArmTF`：单臂，`q: (7,)` = 6 臂关节 + 1 夹爪距离 [0, 0.1]
  - 持有双臂整体 URDF robot（左右共享同一实例）
  - 通过 `_expand_q` 将 (7,) 映射到全 robot 关节向量，关节名 `fl_link1`…`fl_link8`（非 `fl_joint*`）
  - `fkine_gripper` 强制 q_gripper=0（闭合），确保 grip point 与夹爪开闭无关
  - `_build_joint_idx_map`：按 `robot.links` 顺序枚举 `link.isjoint` 关节，不用 `link.joint.name`
- `Arx5RobotwinTF`：双臂，继承 DualArmTF，左右各 7 关节
  - `names` 按全部 link 传入（含 fl_* 和 fr_*），内部自动分配给左右臂

URDF 关节约定：
- 前缀 `fl_` = 左臂，`fr_` = 右臂
- `fl_link1`…`fl_link6`：6 个臂关节
- `fl_link7`, `fl_link8`：夹爪两指（同向，各取 gripper/2）
- 每臂 9 个 link（7 臂体 + 2 夹爪）；`arm_mesh_num=7` 表示前 7 个 mesh 为臂体


## Labeling 流程

路径：`src/caliball/label.py`

### Labeler 核心方法

#### label_grip_point_all_repr
```python
def label_grip_point_all_repr(self, joint_angles, tf_model, extrinsic,
                               gripper_state=None, arm_idx=0) -> PoseAllRepr
```
通过 FK 计算 grip point（TCP）位姿，变换到相机坐标系，返回所有旋转表示（PoseAllRepr）。
- `gripper_state=None` 时从 `tf_model.gripper_scalars(q)[arm_idx]` 推断

#### label_mask_and_bbox
```python
def label_mask_and_bbox(self, joint_angles, tf_model, renderer,
                         vertices_list, faces_list, intrinsic, extrinsic,
                         arm_mesh_num=None, device="cuda") -> list[MaskBboxResult]
```
使用 NVDiffrast 渲染所有 link mesh，按臂体/夹爪分离，返回 `list[MaskBboxResult]`（一个元素/臂）。

**深度缓冲渲染逻辑（per-arm）：**
```
n_links_per_arm = nlinks // n_arms
for link_idx in range(nlinks):
    a = link_idx // n_links_per_arm        # 臂索引
    local_idx = link_idx % n_links_per_arm  # 本臂内索引
    valid = (cur_depth > 0) & (cur_depth < final_depth)
    if local_idx < arm_mesh_num:
        arm_masks[a][valid] = ...  # 臂体
    else:
        grip_masks[a][valid] = ... # 夹爪
```

#### label_frame
```python
def label_frame(self, frame_idx, joint_angles, intrinsic, extrinsic,
                tf_model, renderer, vertices_list, faces_list,
                device="cuda", arm_mesh_num=None,
                skip_mask=False, arm_names=None) -> FrameLabel
```
- 自动推断 `arm_names`（n_arms=2 → `["left","right"]`，n_arms=1 → `["left"]`）
- 调用一次 `label_mask_and_bbox`，再逐臂计算 grip point

#### label_episode
```python
def label_episode(self, joint_angles_list, intrinsic, extrinsic,
                  tf_model, renderer, vertices_list, faces_list,
                  device="cuda", arm_mesh_num=None, skip_mask=False,
                  dataset_name="", episode_id="",
                  camera_name="", arm_names=None) -> LabelData
```
- 逐帧调用 `label_frame`，汇总为 `LabelData`
- 不再依赖 eef_pose；所有位姿信息均由 FK 计算


## 标注入口脚本

路径：`scripts/label_from_config.py`

用 Hydra compose 加载任务 YAML，初始化 tf / dataset，按 episode 导出标注 JSON + pickle。

关键参数：
- `--config`：任务 YAML 路径
- `--output_dir`：输出目录（或在 YAML 中设 `label.output_dir`）
- `--skip_mask`：跳过 mesh 加载与 mask 渲染
- `--episode_start / --max_episodes`：episode 范围
- `--camera_names`：逻辑相机名列表
- `--base_path / --dataset_name`：批量任务覆盖 YAML 中的数据路径

### YAML 配置中的 label 段

```yaml
label:
  output_dir: ./label_out/my_task
  camera_names: [image]
  episode_start: 0
  max_episodes: null
  device: cuda
  skip_mask: false
  arm_mesh_num: 7    # 每臂前 N 个 mesh 视为臂体（余下为夹爪）
```

`arm_mesh_num` 默认 7（适配 ARX5：9 link/臂，前 7 为臂体）。

### 输出格式

每个 episode 产生两个文件：
- `episode_{idx:06d}.json`：帧级标注，结构为 `{cam_name: {intrinsic, extrinsic, frames: [FrameLabel]}}`
- `label_data/episode_{idx:06d}.pkl`：pickle LabelData 对象


## 工具模块

### mesh_loader

路径：`src/caliball/utils/mesh_loader.py`

```python
def load_meshes(robot_config, device: str, project_root=None) -> tuple[list, list]:
    """加载 robot mesh，返回 (vertices_list, faces_list)，均为 CUDA Tensor。"""
```

### NVDiffrastRenderer

路径：`src/caliball/utils/nvdiffrast_renderer.py`

- `NVDiffrastRenderer(resolution, device)` 初始化
- `render_all(verts, faces, K, object_pose, mask_color)` 返回 `{depth, mask}`


## 可视化脚本

路径：`scripts/visualize_labels.py`

读取 episode JSON 和原始视频帧，叠加标注后输出图片/视频。

- 双臂支持：左臂（黄/蓝色系）、右臂（青/紫色系）各用独立配色
- 每臂独立绘制 robot bbox、grip point UV、gripper bbox
- 夹爪开合状态显示在左上角（多臂时各占一行）
- **旋转轴可视化**：在 grip point 处绘制 X（红）/ Y（绿）/ Z（蓝）三轴箭头，轴向来自 `xyz_mat_g` 的旋转列，直接在相机坐标系下用内参投影
- `--no_eef`：不绘制 eef 相关信息
- `--no_axes`：不绘制旋转轴
- `--axes_scale`：轴箭头长度（米，默认 0.05）
- `--first_frame_only`：仅导出首帧 JPG（加快调试）

### 快速测试脚本

路径：`scripts/label_from_config_test_visual.sh`

```bash
CONFIG=src/caliball/config/rdt_aloha.yaml bash scripts/label_from_config_test_visual.sh
SKIP_MASK=1 bash scripts/label_from_config_test_visual.sh   # 跳过 mask 渲染
SKIP_LABEL=1 bash scripts/label_from_config_test_visual.sh  # 仅可视化
LABEL_TEST_OUT=/tmp/myout bash scripts/label_from_config_test_visual.sh  # 自定义输出目录
```

### 全数据集批量测试脚本

路径：`tmp/test_all_datasets.sh`

并行跑全部 9 个数据集的标注 + 可视化（episode 0，跳过 mask，输出完整视频）：

```bash
bash tmp/test_all_datasets.sh
```

- 日志：`/tmp/test_all_logs/<dataset_name>.log`
- 输出：`/tmp/test_all/<dataset_name>/`，可视化视频在 `.../vis/episode_000000/`

## Dataset合并
dataset位置/cpfs02/user/xiesicheng.xsc/project/CalibAll/src/caliball/dataset
我想把现在的这些dataset合并一下，你需要
1. check现在数据load的逻辑是不是速度最快的方案
2. 数据集需要指定：
   1. 位置（repo_id）
   2. state key，但这个可能有很多个（aloha的时候）
3. 需要输出
   1. joint angles
   2. gripper，一般情况会在joint angles里，但是有可能单独保存。如果单独保存，就靠后处理和joint angles合起来
   3. 不同视角的 video
4. joint可能需要后处理（比如toto）
5. eef pose要删掉，不需要了

## 数据集汇总

| # | Config | 数据集目录 | Episodes | 已标定相机 | 备注 |
|---|--------|-----------|----------|-----------|------|
| 1 | robomind_ur5e_1rgb.yaml | `RoboMIND_lerobot_v2.1/*/ur_1rgb/*` | 25,002 | 1（camera_top，4个标定集） | UR5e + Robotiq；多任务分组标定 |
| 2 | robomind_aloha.yaml | `RoboMIND_lerobot_v2.1/*/agilex_3rgb/*` | 9,298 | 1（camera_top） | AgileX 双臂 ALOHA；仅顶部相机标定 |
| 3 | robomind_franka.yaml | `RoboMIND_lerobot_v2.1/*/franka_3rgb/*` | 16,879 | 3（camera_top / camera_left / camera_right，4个场景集） | Franka；3视角全标定 |
| 4 | rdt_aloha.yaml | `rdt_aloha_lerobot2.1/*` | 6,061 | 1（camera_top，与 agilex_3rgb 共用标定） | THU RDT-ALOHA；240+ 任务 |
| 5 | berkeley_autolab_ur5.yaml | `berkeley_autolab_ur5` (symlink→OXE) | 896 | 1（observation.images.image） | UR5e + Robotiq；gripper 来自 state[13] |
| 6 | nonprehensile.yaml | `kaist_nonprehensile_converted_externally_to_rlds` (symlink→OXE) | 201 | 1（observation.images.image） | Franka；交错关节 [q,w,q,w,...] |
| 7 | nyu_franka.yaml | `nyu_franka_play_dataset_converted_externally_to_rlds` (symlink→OXE) | 365 | 2（image + image_additional_view） | Franka；双视角均已标定 |
| 8 | toto.yaml | `toto` (symlink→OXE) | 902 | 1（observation.images.image） | Franka；关节偏置 [0,0,0,0,0,π/2,π/4] |
| 9 | ucsd_kitchen.yaml | `ucsd_kitchen_dataset_converted_externally_to_rlds` (symlink→OXE) | 150 | 1（observation.images.image） | XArm7；gripper 来自 action[6] |

**合计：59,754 episodes，覆盖 UR5e / XArm7 / Franka / AgileX-ALOHA 四种机型**

数据根目录：`/cpfs02/user/xiesicheng.xsc/project/CalibAll/data/caliball_data/`
OXE symlink 源：`/cpfs02/user/xiesicheng.xsc/convert/oxe/lerobot_2.1/`
标定参数：`/cpfs02/user/xiesicheng.xsc/project/CalibAll/src/caliball/config/camera_intrinsic_extrinsic.py`


## scripts/ 目录文件说明

### 标注流水线

| 文件 | 作用 |
|------|------|
| `label_from_config.py` | **核心标注入口**。读 YAML config，初始化 TF/dataset，逐 episode 生成 `episode_*.json` + `label_data/*.pkl` |
| `create_lerobot_with_anno.py` | **生成带标注的新 lerobot 数据集**。读原始 lerobot dataset，运行标注流水线，输出含 `annotation.*` 列的新 parquet dataset（不修改原始数据）|
| `label_visual_whole.py` | **全数据集批量标注 + 可视化**。并行跑 9 个数据集的 episode 0 标注与可视化，支持 GPU 轮询分配，断点续跑 |
| `process_dataset.py` | 早期批量标注脚本，遍历 RoboMIND franka_3rgb，生成 JSON。已被 `label_from_config.py` 取代 |

### 标注写回

| 文件 | 作用 |
|------|------|
| `write_labels_to_lerobot.py` | 从 `label_out/visual_whole` 读取 pkl，将 `annotation.*` 列 in-place 写回原始 lerobot parquet。幂等，已有列则跳过 |

### 可视化

| 文件 | 作用 |
|------|------|
| `visualize_labels.py` | 读标注 JSON + 原始视频，叠加 mask/bbox/UV/坐标轴输出 MP4/JPG。支持单帧、单 episode、批量三种模式 |
| `visualize_anno_dataset.py` | 从 `create_lerobot_with_anno.py` 输出的 parquet dataset 直接读取标注列，生成可视化 MP4 |
| `review_calibration.py` | 启动本地 HTTP 服务，在浏览器中展示各 task 标注预览（视频中间帧），逐一判断标定是否 OK，结果保存为 JSON |
| `visual_eef.py` | 早期调试脚本，可视化 EEF 位姿投影 |
| `robomind_ur1rgb_ep0_first_frame_gallery.py` | 截取 RoboMIND ur_1rgb 各任务 episode 0 首帧，生成 index.html 图库 |

### 外参标定

| 文件 | 作用 |
|------|------|
| `extrinsic_detection.py` | 外参检测流程：解码视频帧 → 浏览器选追踪点 → SAM3 mask → 写 manual_label。分两趟运行 |

### 数据集工具

| 文件 | 作用 |
|------|------|
| `convert_lerobot_30_to_21.py` | 将 LeRobot v3.0 格式数据集转换为 v2.1 格式 |
| `resize_lerobot_dataset.py` | 将 lerobot 数据集的视频缩放到指定分辨率（默认 224×224），同步更新 meta/info.json 和 parquet 中的标定数值 |
| `find_valid_tasks.py` | 枚举 RoboMIND franka_3rgb 可用 task 路径，输出 `valid_tasks.json` 供批量处理脚本使用 |

### 调试 / 检查

| 文件 | 作用 |
|------|------|
| `check_robot.py` | 硬编码机器人类型，交互式测试 FK 结果（调试用）|
| `check_robot_from_config.py` | 从 YAML config 实例化 TF 模型并打印 FK 结果 |
| `check_qpos.py` | 检查 qpos 关节角范围/格式 |
| `test_lerobot_anno.py` | 用官方 `LeRobotDataset` 加载 `create_lerobot_with_anno.py` 输出，验证 annotation 列可正常读取 |
| `infer.py` | 标注结果推断/测试脚本（开发调试用）|
| `infer_manual.py` | 手动标注调试 |
| `infer_manual_aloha.py` | ALOHA 双臂手动标注调试 |
| `infer_manual_aloha_refine.py` | ALOHA 标注结果精化调试 |
| `label_pos.py` | 早期调试脚本，测试 grip point 标注位置 |
| `mask_robot.py` | 早期调试脚本，测试 robot mask 渲染 |
| `mask_aloha.py` | 早期调试脚本，测试 ALOHA mask 渲染 |
| `test.py` | 临时测试脚本 |

### Shell 启动脚本

| 文件 | 作用 |
|------|------|
| `label_from_config.sh` | 快速启动 `label_from_config.py`（berkeley_autolab_ur5 默认配置）|
| `label_from_config_test.sh` | 只跑 episode 0 的快速测试，验证管线是否正常 |
| `label_from_config_test_visual.sh` | episode 0 标注 + 可视化一体化测试。支持 `CONFIG=`/`SKIP_LABEL=1`/`SKIP_MASK=1` 等环境变量 |
| `run_visualize_all.sh` | 批量可视化 `label_result/` 和 `label_out/` 下所有 task 的标注 JSON，支持 `--episode` 指定单 episode |
| `review_calibration.sh` | 启动 `review_calibration.py`（旧路径 `label_out/robomind_ur`，端口 8765）|
| `extrinsic_detection.sh` | 启动外参检测流程（需设 `TASK_PATH` 环境变量）|
| `run_process.sh` | 批量运行 `process_dataset.py`（早期批量标注，已基本废弃）|
| `convert.sh` | OXE 数据集格式转换批量脚本 |


## label数据写入
现在我想把我