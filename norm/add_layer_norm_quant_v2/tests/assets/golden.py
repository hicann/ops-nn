#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""AddLayerNormQuantV2 多通路 golden（TestSpec 范式）。

通路支持面（见 01_requirement.md §3.3）：
  kernel ✅ / geir ❌ / aclnn ✅ / e2e ❌（torch_npu 二进制未引用 aclnnAddLayerNormQuantV2，
  对照 aclnnAddRmsNormQuant 有命中，判定方法有效）。
"""

__spec__ = {
    # kernel
    "add_layer_norm_quant_v2": "AddLayerNormQuantV2KernelSpec",
    # aclnn
    "aclnnAddLayerNormQuantV2": "AddLayerNormQuantV2AclnnSpec",
}

import numpy as np
import torch

# 【不存在】e2e/torch 通路：torch_npu 未引用 aclnnAddLayerNormQuantV2（strings 查 0 命中）

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int8": {"standard": "quant"},  # 量化输出判据：绝对误差 <= 1 LSB
}


def _attr(kwargs, name, default, *aliases):
    for n in (name,) + aliases:
        if n in kwargs:
            v0 = kwargs[n]
            if isinstance(v0, (list, tuple)) and v0:
                v0 = v0[0]
            if isinstance(v0, str):
                s = v0.strip().strip('"').strip("'").lower()
                if s in ("true", "false", "yes", "no", "1", "0"):
                    return s in ("true", "yes", "1")
                try:
                    return type(default)(s)
                except Exception:
                    return v0.strip().strip('"').strip("'")
            return v0
    return _attr_old(kwargs, name, default)


def _attr_old(kwargs, name, default):
    v = kwargs.get(name, default)
    if isinstance(v, (list, tuple)) and v:
        v = v[0]
    if isinstance(v, str):
        s = v.strip().strip('"').strip("'").lower()
        if s in ("true", "false", "yes", "no", "1", "0"):
            return s in ("true", "yes", "1")
        try:
            return type(default)(s)
        except Exception:
            return default
    return v


def _as_tensor(a):
    """numpy/torch -> torch.Tensor；缺失可选输入(None/object 占位) -> None；fp16/bf16 抬到 fp32。

    抬精度的理由：内核按 fp32 累加、最后才 round 回 fp16/bf16；golden 若留在 fp16 逐位算，
    会因每步 round 而比内核糙，边缘元素差几 ULP → 判据在容差边缘随机翻红（仅向上兜底，不向下 cast）。
    """
    if a is None:
        return None
    if isinstance(a, torch.Tensor):
        return a.to(torch.float32) if a.dtype in (torch.bfloat16, torch.float16) else a
    arr = np.asarray(a)
    if arr.dtype == object:
        flat = [x for x in arr.reshape(-1).tolist() if x is not None]
        if not flat:
            return None
        arr = np.asarray(flat[0])
    if str(arr.dtype) in ("bfloat16", "float16"):
        arr = arr.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _compute(*tensors, **kwargs):
    """x1,x2,gamma,beta,bias?,scales1?,scales2?,zero_points1?,zero_points2?
    -> y1,y2,x,layernorm_res,out_scales1,out_scales2（顺序照 def.cpp）。"""
    t = list(tensors) + [None] * 9
    x1, x2, gamma, beta, bias, scales1, scales2, zp1, zp2 = t[:9]

    quant_mode = (
        str(_attr(kwargs, "quant_mode", "dynamic", "quantMode"))
        .strip()
        .strip('"')
        .strip("'")
    )
    epsilon = float(_attr(kwargs, "epsilon", 1e-5))
    div_mode = bool(_attr(kwargs, "div_mode", True, "divMode"))

    x = x1 + x2 if bias is None else x1 + x2 + bias

    row = int(gamma.reshape(-1).numel())
    xf = x.reshape(-1, row)
    x_mean = xf.mean(dim=-1, keepdim=True)
    x_var = xf.var(dim=-1, unbiased=False, keepdim=True)
    g = gamma.reshape(1, -1).to(xf.dtype)
    b = beta.reshape(1, -1).to(xf.dtype)
    y = (xf - x_mean) / torch.sqrt(x_var + epsilon) * g + b
    y = y.reshape(x.shape)

    layernorm_res = y

    def _round_i8(v):
        return torch.clamp(torch.round(v), -128, 127).to(torch.int8)

    dev = x1.device

    if quant_mode == "static":
        s1 = scales1.reshape(-1)[0] if scales1.numel() == 1 else scales1
        y1 = (
            _round_i8(y / s1 + (0.0 if zp1 is None else zp1))
            if div_mode
            else _round_i8(y * s1 + (0.0 if zp1 is None else zp1))
        )
        if scales2 is not None:
            s2 = scales2.reshape(-1)[0] if scales2.numel() == 1 else scales2
            y2 = (
                _round_i8(y / s2 + (0.0 if zp2 is None else zp2))
                if div_mode
                else _round_i8(y * s2 + (0.0 if zp2 is None else zp2))
            )
        else:
            y2 = torch.zeros((1,), dtype=torch.int8, device=dev)
        out_scale1 = torch.ones((1,), dtype=torch.float32, device=dev)
        out_scale2 = torch.ones((1,), dtype=torch.float32, device=dev)
    else:
        yf = y.reshape(-1, row)

        def _dyn(sc):
            tmp = (
                yf
                if sc is None
                else yf * sc.reshape(-1).to(yf.dtype)
                if sc.numel() == 1
                else yf * sc
            )
            m = torch.abs(tmp).amax(dim=-1)
            os_ = (m / 127.0).to(torch.float32)
            out = _round_i8(tmp / os_.reshape(-1, 1)).reshape(x.shape)
            return out, os_

        y1, out_scale1 = _dyn(scales1)
        if scales2 is not None:
            y2, out_scale2 = _dyn(scales2)
        else:
            y2 = torch.zeros((1,), dtype=torch.int8, device=dev)
            out_scale2 = torch.zeros((1,), dtype=torch.float32, device=dev)

    return [y1, y2, x, layernorm_res, out_scale1, out_scale2]


_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def _od(kwargs):
    od = kwargs.get("output_dtypes") or []
    return [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]


def _cast_out(outs, od):
    """KernelSpec 出口：按【下发】的 output_dtypes 窄回，返 numpy。"""
    res = []
    for i, o in enumerate(outs):
        arr = o.detach().cpu().numpy()
        if i < len(od) and od[i]:
            arr = arr.astype(od[i])
        res.append(arr)
    return res


def _cast_out_tensor(outs, od):
    """AclnnSpec 出口：按【下发】的 output_dtypes 窄回，返 torch.Tensor（模板要求返 Tensor）。

    Promote 下 od 已是抬高档（fp16→fp32、fp32→fp64）→ 本步退化为 no-op，等价规范的「零 cast」；
    两方（无 Promote）下 od 是算子声明 dtype → 与 NPU 出口窄回行为一致。
    """
    res = []
    for i, o in enumerate(outs):
        if not isinstance(o, torch.Tensor):
            o = torch.as_tensor(o)
        if i < len(od) and od[i] and od[i] in _TORCH_DTYPE:
            o = o.to(_TORCH_DTYPE[od[i]])
        res.append(o)
    return res


class AddLayerNormQuantV2KernelSpec:
    """kernel；收 numpy.ndarray，返 numpy.ndarray。"""

    def golden(*inputs, **kwargs):
        od = _od(kwargs)
        t = [_as_tensor(a) for a in inputs]
        outs = _compute(*t, **kwargs)
        return _cast_out(outs, od)

    tolerance = _TOL


class AddLayerNormQuantV2AclnnSpec:
    """aclnn；golden 收已 H2D 的 torch.Tensor，【直接返 Tensor】（规范 §6 / 模板要求）。"""

    def golden(*inputs, **kwargs):
        od = _od(kwargs)
        # fp16/bf16 输入抬到 fp32 再算（与 KernelSpec 的 _as_tensor 一致）：仅向上兜底，不向下 cast；
        # 镜像 NPU 内核的加宽行为（VF 全程 fp32、出口才窄回）。Promote 开启时输入已被 TTK 抬过 → no-op。
        t = [
            a.to(torch.float32)
            if isinstance(a, torch.Tensor)
            and a.dtype in (torch.bfloat16, torch.float16)
            else a
            for a in inputs
        ]
        outs = _compute(*t, **kwargs)
        return _cast_out_tensor(outs, od)

    tolerance = _TOL
