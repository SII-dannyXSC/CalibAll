cd /cpfs02/user/xiesicheng.xsc/project/CalibAll
source .venv/bin/activate

PYTHONPATH=. python scripts/label_from_config.py \
  --config src/caliball/config/berkeley_autolab_ur5.yaml \
  --output_dir ./label_out/berkeley_autolab_ur5