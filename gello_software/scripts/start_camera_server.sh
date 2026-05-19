#!/usr/bin/env bash
# Start the long-lived YAM camera server.
#
# Usage:
#     scripts/start_camera_server.sh [config_path] [extra args...]
#
# Defaults config_path to configs/yam_left.yaml. Extra args are forwarded to
# `python -m gello.cameras.camera_server` (see --help).
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GELLO_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
CONFIG="${1:-${GELLO_DIR}/configs/yam_left.yaml}"
shift || true

if [[ "${CONDA_DEFAULT_ENV:-}" != "ai2_yam" ]]; then
    echo "[start_camera_server] Activating conda env ai2_yam..."
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ai2_yam
fi

cd "${GELLO_DIR}"
exec python -m gello.cameras.camera_server --config "${CONFIG}" "$@"
