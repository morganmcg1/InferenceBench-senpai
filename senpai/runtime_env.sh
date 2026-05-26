#!/usr/bin/env bash
# Source before launching inference servers in SENPAI pods.
#
# This only adjusts process-local runtime paths and cache locations. It does
# not change the official evaluator or benchmark scoring.

set -euo pipefail

pythonpath_entries="$(python - <<'PY'
import site
import sys
from pathlib import Path

roots = []
for raw in site.getsitepackages() + [site.getusersitepackages()]:
    path = Path(raw)
    if path.exists():
        roots.append(path)

includes = []
libs = []
for root in roots:
    nvidia = root / "nvidia"
    if not nvidia.exists():
        continue
    includes.extend(str(p) for p in nvidia.glob("*/include") if p.is_dir())
    libs.extend(str(p) for p in nvidia.glob("*/lib") if p.is_dir())

print(":".join(includes))
print(":".join(libs))
PY
)"

nvidia_includes="$(printf '%s\n' "$pythonpath_entries" | sed -n '1p')"
nvidia_libs="$(printf '%s\n' "$pythonpath_entries" | sed -n '2p')"

if [ -n "$nvidia_includes" ]; then
  export CPATH="${nvidia_includes}${CPATH:+:${CPATH}}"
  export C_INCLUDE_PATH="${nvidia_includes}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
  export CPLUS_INCLUDE_PATH="${nvidia_includes}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
  nvcc_include_flags="$(printf '%s' "$nvidia_includes" | tr ':' '\n' | sed 's#^#-I#' | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  export NVCC_PREPEND_FLAGS="${nvcc_include_flags}${NVCC_PREPEND_FLAGS:+ ${NVCC_PREPEND_FLAGS}}"
fi

if [ -n "$nvidia_libs" ]; then
  export LD_LIBRARY_PATH="${nvidia_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/senpai-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${XDG_CACHE_HOME}/cuda}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${XDG_CACHE_HOME}/vllm}"
export INFERENCE_BENCH_MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"
export PIP_REQUIRE_VIRTUALENV="${PIP_REQUIRE_VIRTUALENV:-true}"

# RTX PRO 6000 / Blackwell shakedown default: vLLM can auto-use FlashInfer
# sampling/prefill when the package is installed, even if a launcher did not
# explicitly request the FlashInfer attention backend. Leave opt-in possible.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_DISABLE_FLASHINFER_PREFILL="${VLLM_DISABLE_FLASHINFER_PREFILL:-1}"

mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$VLLM_CACHE_ROOT"
