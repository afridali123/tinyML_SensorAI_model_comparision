#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUNDLE_DIR="${PACKAGE_DIR}/bundle"
ENGINE_DIR="${PACKAGE_DIR}/engines"
PRECISION="fp16"
WORKSPACE="1024"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir)
      BUNDLE_DIR="$2"
      shift 2
      ;;
    --engine-dir)
      ENGINE_DIR="$2"
      shift 2
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 2
      ;;
  esac
done

TRTEXEC_PATH="$(command -v trtexec || true)"
if [[ -z "${TRTEXEC_PATH}" ]]; then
  TRTEXEC_PATH="$(find /usr -name trtexec 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${TRTEXEC_PATH}" ]]; then
  echo "Unable to locate trtexec. Confirm TensorRT is installed on Jetson."
  exit 4
fi

TRTEXEC_HELP="$("${TRTEXEC_PATH}" --help 2>&1 || true)"
if grep -q -- "--memPoolSize" <<< "${TRTEXEC_HELP}"; then
  WORKSPACE_OPTION="--memPoolSize=workspace:${WORKSPACE}"
else
  WORKSPACE_OPTION="--workspace=${WORKSPACE}"
fi

mkdir -p "${ENGINE_DIR}"

build_engine() {
  local name="$1"
  local shape="$2"
  local onnx_path="${BUNDLE_DIR}/${name}.onnx"
  local engine_path="${ENGINE_DIR}/${name}_${PRECISION}.engine"

  if [[ ! -f "${onnx_path}" ]]; then
    echo "Missing ONNX file: ${onnx_path}"
    exit 3
  fi

  local cmd=("${TRTEXEC_PATH}" "--onnx=${onnx_path}" "--saveEngine=${engine_path}" "${WORKSPACE_OPTION}" "--shapes=input:${shape}")
  if [[ "${PRECISION}" == "fp16" ]]; then
    cmd+=("--fp16")
  fi

  echo "Building ${engine_path}"
  echo "Command: ${cmd[*]}"
  "${cmd[@]}"
}

build_engine "fashion_mnist_cnn" "1x28x28x1"
build_engine "cats_dogs_transfer" "1x96x96x3"

echo "TensorRT engines written to ${ENGINE_DIR}"
