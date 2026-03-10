# 激活 conda
source /inspire/hdd/global_user/xiesicheng-253108120120/env/miniconda3/bin/activate
conda activate lerobot

# 要处理的 repo_id 列表
repo_id_list=(
  taco_play
  toto
  columbia_cairlab_pusht_real
  stanford_kuka_multimodal_dataset
  nyu_rot_dataset
  stanford_hydra_dataset
  austin_buds_dataset
  nyu_franka_play_dataset
)

ROOT=/inspire/hdd/project/autoregressive-video-generation/public/xsc/data
SCRIPT=/inspire/hdd/global_user/xiesicheng-253108120120/project/CalibAll/scripts/convert_lerobot_30_to_21.py

# 逐个转换
for repo_id in "${repo_id_list[@]}"; do
  echo "=== Converting ${repo_id} ==="
  python "${SCRIPT}" \
    --repo-id "${repo_id}" \
    --root "${ROOT}"
done
