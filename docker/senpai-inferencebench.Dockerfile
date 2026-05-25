ARG BASE_IMAGE=ghcr.io/coreweave/ml-containers/torch-extras:c0f5966-base-cuda13.2.0-ubuntu22.04-torch2.11.0-vision0.26.0-audio2.11.0-abi1
FROM ${BASE_IMAGE}

ARG VLLM_VERSION=0.11.0

ENV DEBIAN_FRONTEND=noninteractive \
    UV_HTTP_TIMEOUT=600 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/opt/uv_cache \
    PIP_CACHE_DIR=/opt/pip_cache \
    PATH=/root/.local/bin:${PATH} \
    NO_PROXY=localhost,127.0.0.1 \
    no_proxy=localhost,127.0.0.1 \
    SOFT_FILELOCK=1 \
    PYTHONPATH=/opt/filelock_workarounds/soft_file_locks:${PYTHONPATH}

RUN chmod 1777 /tmp && \
    mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        gettext-base \
        git \
        iproute2 \
        lsof \
        net-tools \
        netcat-openbsd \
        ninja-build \
        pkg-config \
        procps \
        ripgrep \
        swig \
        wget \
        && rm -rf /var/lib/apt/lists/*

RUN cat >/usr/local/bin/nvidia-smi <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
for candidate in \
  /usr/bin/nvidia-smi \
  /usr/local/nvidia/bin/nvidia-smi \
  /usr/lib/nvidia/current/nvidia-smi
do
  if [ -x "${candidate}" ]; then
    exec "${candidate}" "$@"
  fi
done
echo "nvidia-smi: not found in container; check Kubernetes GPU passthrough." >&2
exit 127
EOF
RUN chmod +x /usr/local/bin/nvidia-smi
RUN test -s /usr/local/bin/nvidia-smi && grep -q '/usr/bin/nvidia-smi' /usr/local/bin/nvidia-smi

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli-stable.list && \
    apt-get update && apt-get install -y --no-install-recommends nodejs gh && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 \
        -o /usr/local/bin/yq && \
    chmod +x /usr/local/bin/yq && \
    curl -fsSL "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
        -o /usr/local/bin/kubectl && \
    chmod +x /usr/local/bin/kubectl

RUN python -m pip install --no-cache-dir --upgrade pip uv

RUN curl -fsSL https://claude.ai/install.sh | bash || true && \
    npm install -g \
        @anthropic-ai/claude-code \
        @openai/codex \
        @google/gemini-cli \
        opencode-ai \
        weave-claude-plugin@latest && \
    (sed -i "s/const INACTIVITY_TIMEOUT_MS = 10 \\* 60 \\* 1_000;/const INACTIVITY_TIMEOUT_MS = 12 * 60 * 60 * 1_000;/" \
      "$(npm root -g)/weave-claude-plugin/dist/daemon.js" || true) && \
    mkdir -p /root/.weave_claude_plugin/logs && \
    printf '%s\n' \
      '{' \
      '  "log_file": "/root/.weave_claude_plugin/logs/daemon.log",' \
      '  "weave_project": null,' \
      '  "wandb_api_key": null,' \
      '  "debug": false,' \
      '  "version": "0.1.0",' \
      '  "daemon_socket": "/root/.weave_claude_plugin/daemon.sock"' \
      '}' \
      > /root/.weave_claude_plugin/settings.json

RUN uv pip install --system --no-cache \
        accelerate \
        aiohttp \
        bitsandbytes \
        boto3 \
        certifi \
        ConfigSpace \
        datasets \
        evaluate \
        inspect-ai \
        lm-eval \
        matplotlib \
        numpy \
        openai \
        optuna \
        packaging \
        pandas \
        peft \
        PyYAML \
        requests \
        scikit-learn \
        sentencepiece \
        setuptools \
        shortuuid \
        smac \
        tiktoken \
        tokenizers \
        "transformers>=4.55.2,<4.58" \
        trl \
        wandb \
        wheel

RUN uv pip install --system --no-cache \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 \
        torchvision==0.23.0 \
        torchaudio==2.8.0 \
        xformers==0.0.32.post1 && \
    uv pip install --system --no-cache \
        "vllm==${VLLM_VERSION}" \
        nvidia-curand-cu12 \
        nvidia-cuda-nvcc-cu12 \
        prometheus-client \
        python-multipart \
        uvloop \
        "transformers>=4.55.2,<4.58" && \
    uv pip install --system --no-cache --prerelease=allow flashinfer-python || true

RUN export LD_LIBRARY_PATH="$(python -c 'import site; from pathlib import Path; libs=[]; [libs.extend(str(path) for path in (Path(root) / "nvidia").glob("*/lib") if path.is_dir()) for root in site.getsitepackages() + [site.getusersitepackages()] if (Path(root) / "nvidia").exists()]; print(":".join(dict.fromkeys(libs)))')${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" && \
    python -c 'import torch, transformers, vllm; assert torch.__version__.startswith("2.8.0"), torch.__version__; assert vllm.__version__ == "0.11.0", vllm.__version__; print("torch", torch.__version__); print("transformers", transformers.__version__); print("vllm", vllm.__version__)' && \
    python -c 'from vllm.entrypoints.openai import api_server; print("vllm_openai_api_server", api_server.__name__)'

RUN cd /opt && \
    git clone --depth=1 https://github.com/rank-and-file/filelock_workarounds.git

WORKDIR /workspaces
