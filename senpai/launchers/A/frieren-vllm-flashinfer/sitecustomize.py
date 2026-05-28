"""sitecustomize hook for PR #148: install FlashInfer compat shim in every
Python process (including vLLM's spawned engine workers).

This file is placed first on PYTHONPATH by the patched launcher so Python
imports it during interpreter startup. The actual patch lives in
vllm_flashinfer_compat_patch.py to keep concerns separated.
"""

from __future__ import annotations

import os
import sys

try:
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import vllm_flashinfer_compat_patch  # noqa: F401
except Exception as exc:  # noqa: BLE001
    import traceback

    sys.stderr.write(
        f"[sitecustomize] FlashInfer compat patch failed to load: {exc!r}\n"
    )
    traceback.print_exc()
