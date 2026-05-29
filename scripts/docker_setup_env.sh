#!/usr/bin/env bash
# Sanity-check the CalibAll runtime inside the Docker container, and (re)install
# any of the README §4/§5 dependencies that happen to be missing. The image
# already bundles everything, so on a fresh container this script should be a
# no-op and finish in a few seconds.
set -euo pipefail

cd /workspace/CalibAll

echo "=== Python / Torch / CUDA ==="
python - <<'PY'
import torch
print("python    :", __import__("sys").version.split()[0])
print("torch     :", torch.__version__)
print("cuda ok   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device    :", torch.cuda.get_device_name(0))
    print("cuda ver  :", torch.version.cuda)
PY

echo
echo "=== CalibAll package ==="
python -c "import caliball; print('caliball :', getattr(caliball, '__file__', None) or list(caliball.__path__))"

ensure_pkg() {
    local mod="$1"; shift
    if python -c "import $mod" 2>/dev/null; then
        printf '  %-12s OK\n' "$mod"
    else
        printf '  %-12s MISSING -> installing...\n' "$mod"
        "$@"
    fi
}

echo
echo "=== Special dependencies (nvdiffrast / pytorch3d / MoGe) ==="
ensure_pkg nvdiffrast \
    pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
ensure_pkg pytorch3d \
    pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
        pytorch3d==0.7.9+pt2.9.0cu128
ensure_pkg moge \
    pip install git+https://github.com/microsoft/MoGe.git

echo
echo "=== Third-party repos (co-tracker / sam3 / dinov2) ==="
mkdir -p third_party
clone_if_missing() {
    local dir="$1" url="$2"
    if [ ! -d "third_party/$dir/.git" ] && [ ! -f "third_party/$dir/setup.py" ] && [ ! -f "third_party/$dir/pyproject.toml" ]; then
        echo "  third_party/$dir missing -> cloning"
        git clone --depth 1 "$url" "third_party/$dir"
    else
        echo "  third_party/$dir OK"
    fi
}
clone_if_missing co-tracker https://github.com/facebookresearch/co-tracker
clone_if_missing sam3        https://github.com/facebookresearch/sam3.git
clone_if_missing dinov2      https://github.com/facebookresearch/dinov2

ensure_pkg cotracker pip install -e third_party/co-tracker
ensure_pkg sam3      pip install -e third_party/sam3

echo
echo "=== Environment ready ==="
echo "Launch the Web UI with:"
echo "    python scripts/caliball_web.py --host 0.0.0.0 --port 8765"
