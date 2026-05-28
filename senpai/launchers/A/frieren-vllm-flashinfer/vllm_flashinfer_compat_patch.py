"""vLLM 0.11 <-> FlashInfer 0.6+ compat shim used by PR #148 FlashInfer arms.

vLLM 0.11 pins `flashinfer-python==0.3.1` and calls
`BatchDecodeWithPagedKVCacheWrapper.plan(...)` with positional args matching the
0.3.x signature. FlashInfer 0.4+ inserted `o_data_type` between `kv_data_type`
and `data_type`, which shifts `sm_scale` one slot down. The result is that
`decode_wrapper._sm_scale` ends up holding what was meant for `rope_scale`
(`None`), and the next forward pass trips
`assert decode_wrapper._sm_scale == self.scale` in `flashinfer.py:972`.

This shim wraps `BatchDecodeWithPagedKVCacheWrapper.plan` so it accepts the
old vLLM-0.3.x positional ordering and re-dispatches via kwargs.

Importing this module installs the patch. The launcher should `import` it
before launching `vllm.entrypoints.openai.api_server`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vllm_flashinfer_compat_patch")


def _install_decode_plan_compat() -> bool:
    try:
        import flashinfer  # noqa: F401
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    except ImportError:
        logger.warning("FlashInfer not importable; compat shim not installed.")
        return False

    plan_fn = BatchDecodeWithPagedKVCacheWrapper.plan
    if getattr(plan_fn, "__vllm_flashinfer_compat__", False):
        return True

    def _compat_plan(
        self,
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        window_left=-1,
        logits_soft_cap=None,
        q_data_type="float16",
        kv_data_type=None,
        data_type=None,
        sm_scale=None,
        rope_scale=None,
        rope_theta=None,
        non_blocking=True,
        **kwargs,
    ):
        return plan_fn(
            self,
            indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            non_blocking=non_blocking,
            **kwargs,
        )

    _compat_plan.__vllm_flashinfer_compat__ = True  # type: ignore[attr-defined]
    BatchDecodeWithPagedKVCacheWrapper.plan = _compat_plan
    logger.info(
        "Installed FlashInfer BatchDecodeWithPagedKVCacheWrapper.plan compat shim "
        "(vLLM 0.11 positional args -> kwargs)."
    )
    return True


_install_decode_plan_compat()
