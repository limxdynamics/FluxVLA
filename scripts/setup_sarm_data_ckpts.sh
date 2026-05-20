#!/usr/bin/env bash
# Download SARM example datasets and pretrained checkpoints used by SARM configs.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

DATA_DIR="${DATA_DIR:-./datasets}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
HF_ENDPOINT="${HF_ENDPOINT:-}"
DRY_RUN=0
INSTALL_HF_CLI=0

SARM_DATA_REPO="limxdynamics/FluxVLAData"
SARM_DATA_URL="https://huggingface.co/datasets/${SARM_DATA_REPO}"
CLIP_REPO_ID="openai/clip-vit-base-patch32"
CLIP_URL="https://huggingface.co/${CLIP_REPO_ID}"
QWEN3_VL_REPO_ID="Qwen/Qwen3-VL-30B-A3B-Instruct"
QWEN3_VL_URL="https://huggingface.co/${QWEN3_VL_REPO_ID}"

SARM_VERSION="v3"
INCLUDE_VLM_DATA=0
INCLUDE_QWEN3_VL=0
SKIP_DATA=0
SKIP_CKPTS=0
HF_DOWNLOAD_CMD=()
DATASET_FOLDERS=()
CHECKPOINT_KEYS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_sarm_data_ckpts.sh [options]

Default:
  Download the SARM manual v3.0 example dataset and CLIP checkpoint.
  This matches the built-in configs/sarm/*.py defaults.

Options:
  --version v2|v3|both     SARM dataset version to download. Default: v3.
  --with-vlm               Also download VLM annotation assets:
                            SARM_vlm dataset for the selected version and Qwen3-VL.
  --include-vlm-data       Download SARM_vlm dataset for the selected version.
  --include-qwen3-vl       Download Qwen3-VL checkpoint for VLM auto-annotation.
  --all                    Download all released SARM example datasets plus CLIP and Qwen3-VL.
  --skip-data              Do not download datasets.
  --skip-ckpts             Do not download checkpoints.
  --data-dir PATH          Dataset output directory. Default: ./datasets
  --ckpt-dir PATH          Checkpoint output directory. Default: ./checkpoints
  --mirror                 Use https://hf-mirror.com as HF_ENDPOINT.
  --hf-endpoint URL        Use a custom Hugging Face endpoint.
  --install-hf-cli         Install huggingface_hub[cli] if the CLI is missing.
  --dry-run                Print commands without downloading.
  -h, --help               Show this help.

Examples:
  bash scripts/setup_sarm_data_ckpts.sh
  bash scripts/setup_sarm_data_ckpts.sh --version v2
  bash scripts/setup_sarm_data_ckpts.sh --with-vlm --mirror
  bash scripts/setup_sarm_data_ckpts.sh --all --data-dir /mnt/data/datasets --ckpt-dir /mnt/data/checkpoints

Official sources:
  SARM data: https://huggingface.co/datasets/limxdynamics/FluxVLAData
  CLIP:      https://huggingface.co/openai/clip-vit-base-patch32
  Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

contains_item() {
  local needle="$1"
  shift || true

  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done

  return 1
}

add_dataset_folder() {
  local folder="$1"

  if ! contains_item "${folder}" "${DATASET_FOLDERS[@]+"${DATASET_FOLDERS[@]}"}"; then
    DATASET_FOLDERS+=("${folder}")
  fi
}

add_checkpoint_key() {
  local key="$1"

  if ! contains_item "${key}" "${CHECKPOINT_KEYS[@]+"${CHECKPOINT_KEYS[@]}"}"; then
    CHECKPOINT_KEYS+=("${key}")
  fi
}

select_dataset_folders() {
  if [[ "${SKIP_DATA}" == "1" ]]; then
    return
  fi

  case "${SARM_VERSION}" in
    v2)
      add_dataset_folder SARM_manual_test_10Episodes_lerobotv2.1
      if [[ "${INCLUDE_VLM_DATA}" == "1" ]]; then
        add_dataset_folder SARM_vlm_test_10Episodes_lerobotv2.1
      fi
      ;;
    v3)
      add_dataset_folder SARM_manual_test_10Episodes_lerobotv3.0
      if [[ "${INCLUDE_VLM_DATA}" == "1" ]]; then
        add_dataset_folder SARM_vlm_test_10Episodes_lerobotv3.0
      fi
      ;;
    both)
      add_dataset_folder SARM_manual_test_10Episodes_lerobotv3.0
      add_dataset_folder SARM_manual_test_10Episodes_lerobotv2.1
      if [[ "${INCLUDE_VLM_DATA}" == "1" ]]; then
        add_dataset_folder SARM_vlm_test_10Episodes_lerobotv3.0
        add_dataset_folder SARM_vlm_test_10Episodes_lerobotv2.1
      fi
      ;;
    *)
      die "unknown SARM version: ${SARM_VERSION}. Use v2, v3, or both."
      ;;
  esac
}

select_checkpoint_keys() {
  if [[ "${SKIP_CKPTS}" == "1" ]]; then
    return
  fi

  add_checkpoint_key clip
  if [[ "${INCLUDE_QWEN3_VL}" == "1" ]]; then
    add_checkpoint_key qwen3-vl
  fi
}

detect_hf_cli() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    HF_DOWNLOAD_CMD=(huggingface-cli download)
    return
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_DOWNLOAD_CMD=(huggingface-cli download)
    return
  fi

  if command -v hf >/dev/null 2>&1; then
    HF_DOWNLOAD_CMD=(hf download)
    return
  fi

  if [[ "${INSTALL_HF_CLI}" == "1" ]]; then
    run_cmd python -m pip install -U 'huggingface_hub[cli]'
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_DOWNLOAD_CMD=(huggingface-cli download)
    return
  fi

  if command -v hf >/dev/null 2>&1; then
    HF_DOWNLOAD_CMD=(hf download)
    return
  fi

  die "huggingface-cli or hf was not found. Install with: pip install -U 'huggingface_hub[cli]'"
}

hf_download() {
  if [[ "${#HF_DOWNLOAD_CMD[@]}" -eq 0 ]]; then
    detect_hf_cli
  fi

  run_cmd "${HF_DOWNLOAD_CMD[@]}" "$@"
}

dataset_exists() {
  local folder="$1"
  local target="${DATA_DIR}/${folder}"

  [[ -f "${target}/meta/info.json" ]] || return 1
  [[ -d "${target}/data" || -d "${target}/videos" ]] || return 1
}

checkpoint_target_dir() {
  local key="$1"

  case "${key}" in
    clip)
      printf '%s\n' "${CKPT_DIR}/clip-vit-base-patch32"
      ;;
    qwen3-vl)
      printf '%s\n' "${CKPT_DIR}/Qwen3-VL-30B-A3B-Instruct"
      ;;
    *)
      die "unknown checkpoint key: ${key}"
      ;;
  esac
}

checkpoint_exists() {
  local key="$1"
  local target

  target="$(checkpoint_target_dir "${key}")"
  [[ -e "${target}" || -L "${target}" ]] || return 1
  [[ -f "${target}/config.json" || -d "${target}/snapshots" ]] || return 1
}

download_sarm_dataset() {
  local folder="$1"

  if dataset_exists "${folder}"; then
    echo "Dataset exists, skipping: ${DATA_DIR}/${folder}"
    return
  fi

  echo "Downloading ${folder} from ${SARM_DATA_URL}"
  hf_download "${SARM_DATA_REPO}" \
    --repo-type dataset \
    --include "${folder}/*" \
    --local-dir "${DATA_DIR}"
}

download_checkpoint() {
  local key="$1"
  local target

  case "${key}" in
    clip)
      target="$(checkpoint_target_dir "${key}")"
      if checkpoint_exists "${key}"; then
        echo "Checkpoint exists, skipping: ${target}"
        return
      fi
      echo "Downloading CLIP from ${CLIP_URL}"
      hf_download "${CLIP_REPO_ID}" \
        --local-dir "${target}"
      ;;
    qwen3-vl)
      target="$(checkpoint_target_dir "${key}")"
      if checkpoint_exists "${key}"; then
        echo "Checkpoint exists, skipping: ${target}"
        return
      fi
      echo "Downloading Qwen3-VL from ${QWEN3_VL_URL}"
      hf_download "${QWEN3_VL_REPO_ID}" \
        --local-dir "${target}"
      ;;
    *)
      die "unknown checkpoint key: ${key}"
      ;;
  esac
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --version)
      [[ "$#" -ge 2 ]] || die "--version requires a value"
      SARM_VERSION="$2"
      shift 2
      ;;
    --with-vlm)
      INCLUDE_VLM_DATA=1
      INCLUDE_QWEN3_VL=1
      shift
      ;;
    --include-vlm-data)
      INCLUDE_VLM_DATA=1
      shift
      ;;
    --include-qwen3-vl)
      INCLUDE_QWEN3_VL=1
      shift
      ;;
    --all)
      SARM_VERSION="both"
      INCLUDE_VLM_DATA=1
      INCLUDE_QWEN3_VL=1
      shift
      ;;
    --skip-data)
      SKIP_DATA=1
      shift
      ;;
    --skip-ckpts)
      SKIP_CKPTS=1
      shift
      ;;
    --data-dir)
      [[ "$#" -ge 2 ]] || die "--data-dir requires a value"
      DATA_DIR="$2"
      shift 2
      ;;
    --ckpt-dir)
      [[ "$#" -ge 2 ]] || die "--ckpt-dir requires a value"
      CKPT_DIR="$2"
      shift 2
      ;;
    --mirror)
      HF_ENDPOINT="https://hf-mirror.com"
      shift
      ;;
    --hf-endpoint)
      [[ "$#" -ge 2 ]] || die "--hf-endpoint requires a value"
      HF_ENDPOINT="$2"
      shift 2
      ;;
    --install-hf-cli)
      INSTALL_HF_CLI=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1. Use --help for usage."
      ;;
  esac
done

select_dataset_folders
select_checkpoint_keys

if [[ -n "${HF_ENDPOINT}" ]]; then
  export HF_ENDPOINT
  echo "Using HF_ENDPOINT=${HF_ENDPOINT}"
fi

run_cmd mkdir -p "${DATA_DIR}" "${CKPT_DIR}"

if [[ "${#DATASET_FOLDERS[@]}" -eq 0 && "${#CHECKPOINT_KEYS[@]}" -eq 0 ]]; then
  echo "No SARM datasets or checkpoints selected."
  exit 0
fi

if [[ "${#DATASET_FOLDERS[@]}" -gt 0 ]]; then
  echo "SARM datasets: ${DATASET_FOLDERS[*]}"
  for folder in "${DATASET_FOLDERS[@]}"; do
    download_sarm_dataset "${folder}"
  done
fi

if [[ "${#CHECKPOINT_KEYS[@]}" -gt 0 ]]; then
  echo "SARM checkpoints: ${CHECKPOINT_KEYS[*]}"
  for key in "${CHECKPOINT_KEYS[@]}"; do
    download_checkpoint "${key}"
  done
fi

echo "Done."
