#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""clipped_swiglu 的 TTK 多路径 TestSpec golden。

参考docs/aclnnClippedSwiglu.md：
    1. 按 dim 合轴 -> [pre, cut]
    2. group = min(sum(group_index), pre); 仅前 group 行参与计算
    3. interleaved: A=x[:,::2], B=x[:,1::2]; else 前后: A=x[:,:h], B=x[:,h:]  (h=cut//2)
    4. clamp_mode 0: A=clamp(A,None,limit), B=clamp(B,-limit,limit), y=A*sigmoid(alpha*A)*(B+bias)
       clamp_mode 1: B=clamp(B,-limit,limit), y=clamp(A*sigmoid(A),None,limit)*B
"""

__spec__ = {
    "clipped_swiglu": "ClippedSwigluKernelSpec",
    "aclnnClippedSwiglu": "AclnnClippedSwigluSpec",
    "aclnnClippedSwigluV2": "AclnnClippedSwigluV2Spec",
    "cann_ops_nn.clipped_swiglu": "ClippedSwigluE2ESpec",
}

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:  # pragma: no cover
    _bf16 = None


def _prod(seq):
    p = 1
    for v in seq:
        p *= int(v)
    return p


def _to_np(t):
    """Normalize numpy / torch (CPU or NPU) input to a numpy ndarray (or None)."""
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    import torch

    if isinstance(t, torch.Tensor):
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        # torch CPU half/bfloat16 have no stable numpy() view; lift to fp32.
        if t.dtype in (torch.float16, torch.bfloat16):
            return t.float().numpy()
        return t.numpy()
    return np.asarray(t)


def _acl_output_target(dtype_str):
    """Map the input dtype string to the output cast target."""
    dt = str(dtype_str)
    if "bfloat16" in dt:
        return "bfloat16"
    if "float16" in dt:
        return "float16"
    return None  # float32 -> keep fp32


def _ref_clipped_swiglu(
    x,
    group_index,
    dim,
    alpha,
    limit,
    bias,
    interleaved,
    clamp_mode,
    target=None,
    bg_value=1.0,
):
    """float32 CPU reference shared by all paths.

    bg_value is what the op leaves in the rows beyond ``group`` (the rows the
    kernel never writes). ACLNN harness pre-fills pure outputs with dtype(1), so
    there bg_value=1.0; E2E / GEIR output buffers read back 0, so there
    bg_value=0.0. (Confirmed on Ascend950DT.)
    """
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32)

    group_np = _to_np(group_index)
    orig_shape = list(x.shape)
    ndim = len(orig_shape)
    dim_pos = dim % ndim

    pre = _prod(orig_shape[:dim_pos]) if dim_pos > 0 else 1
    cut = _prod(orig_shape[dim_pos:])

    xf = x.reshape(pre, cut)

    group = pre
    if group_np is not None:
        group = min(int(np.asarray(group_np).sum()), pre)

    xt = xf[:group]
    if interleaved:
        a = xt[:, 0::2]
        b = xt[:, 1::2]
    else:
        h = cut // 2
        a = xt[:, :h]
        b = xt[:, h:]

    if int(clamp_mode) == 0:
        a = np.clip(a, None, limit)
        b = np.clip(b, -limit, limit)
        with np.errstate(over="ignore", invalid="ignore"):
            sig = 1.0 / (1.0 + np.exp(-alpha * a))
            res = a * sig * (b + bias)
    elif int(clamp_mode) == 1:
        b = np.clip(b, -limit, limit)
        with np.errstate(over="ignore", invalid="ignore"):
            sig = 1.0 / (1.0 + np.exp(-a))
            a = a * sig
            a = np.clip(a, None, limit)
            res = a * b

    y = np.full((pre, cut // 2), bg_value, dtype=np.float32)
    y[:group] = res.astype(np.float32)

    out_shape = list(orig_shape)
    out_shape[dim_pos] = out_shape[dim_pos] // 2
    y = y.reshape(out_shape)

    if target is not None:
        if target == "bfloat16":
            y = y.astype(_bf16) if _bf16 is not None else y
        else:
            y = y.astype(target)
    return [y]


# ---- Kernel / GEIR（numpy.ndarray）-------------------------------------------------
# GEIR 复用 Kernel 的注册（op_name=clipped_swiglu）；TTK 中 GEIR golden 也走此 golden。
class ClippedSwigluKernelSpec:
    """ClippedSwiglu 的 Kernel / GEIR 流程 golden（输入为 numpy.ndarray）。"""

    def golden(*input_arrays, **kwargs):
        x = np.asarray(input_arrays[0])
        group_index = None
        if len(input_arrays) > 1 and input_arrays[1] is not None:
            group_index = np.asarray(input_arrays[1])

        dim = int(kwargs.get("dim", -1))
        alpha = float(kwargs.get("alpha", 1.702))
        limit = float(kwargs.get("limit", 7.0))
        bias = float(kwargs.get("bias", 1.0))
        interleaved = bool(kwargs.get("interleaved", True))
        clamp_mode = int(kwargs.get("clamp_mode", 0))

        output_dtypes = kwargs.get("output_dtypes")
        if output_dtypes is not None and len(output_dtypes) > 0:
            target = str(output_dtypes[0])
        else:
            target = str(x.dtype)

        return _ref_clipped_swiglu(
            x,
            group_index,
            dim,
            alpha,
            limit,
            bias,
            interleaved,
            clamp_mode,
            target=target,
            bg_value=0.0,
        )


# ---- ACLNN V1 : aclnnClippedSwiglu(x, groupIndexOptional, dim, alpha, limit, bias, interleaved, out) ----
class AclnnClippedSwigluSpec:
    """aclnnClippedSwiglu 的 ACLNN 流程 golden（输入为 torch.Tensor）。"""

    def golden(
        x,
        group_index=None,
        dim=-1,
        alpha=1.702,
        limit=7.0,
        bias=1.0,
        interleaved=True,
        out=None,
        **kwargs,
    ):
        x_np = _to_np(x)
        target = None
        import torch

        if isinstance(x, torch.Tensor):
            target = _acl_output_target(x.dtype)
        return _ref_clipped_swiglu(
            x_np,
            group_index,
            int(dim),
            float(alpha),
            float(limit),
            float(bias),
            bool(interleaved),
            0,
            target=target,
            bg_value=1.0,
        )


# ---- ACLNN V2 : aclnnClippedSwigluV2(..., interleaved, clampMode, out) ----
class AclnnClippedSwigluV2Spec:
    """aclnnClippedSwigluV2 的 ACLNN 流程 golden（输入为 torch.Tensor）。"""

    def golden(
        x,
        group_index=None,
        dim=-1,
        alpha=1.702,
        limit=7.0,
        bias=1.0,
        interleaved=True,
        clampMode=0,
        out=None,
        **kwargs,
    ):
        x_np = _to_np(x)
        target = None
        import torch

        if isinstance(x, torch.Tensor):
            target = _acl_output_target(x.dtype)
        return _ref_clipped_swiglu(
            x_np,
            group_index,
            int(dim),
            float(alpha),
            float(limit),
            float(bias),
            bool(interleaved),
            int(clampMode),
            target=target,
            bg_value=1.0,
        )


# ---- E2E : cann_ops_nn.clipped_swiglu(x, *, group_index=None, dim=-1, alpha=1.702, ...) ----
class ClippedSwigluE2ESpec:
    """cann_ops_nn.clipped_swiglu 的 E2E 流程 golden（输入为 torch.Tensor）。"""

    def golden(x, **kwargs):
        group_index = kwargs.get("group_index")
        clamp_mode = int(kwargs.get("clamp_mode", 0))
        x_np = _to_np(x)
        target = None
        import torch

        if isinstance(x, torch.Tensor):
            target = _acl_output_target(x.dtype)
        return _ref_clipped_swiglu(
            x_np,
            group_index,
            int(kwargs.get("dim", -1)),
            float(kwargs.get("alpha", 1.702)),
            float(kwargs.get("limit", 7.0)),
            float(kwargs.get("bias", 1.0)),
            bool(kwargs.get("interleaved", True)),
            clamp_mode,
            target=target,
            bg_value=0.0,
        )
