#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""ThnnFusedGruCellGrad TTK TestSpec（golden）。

数学语义:

    hsz = storage.shape[1] // 5              # storage (B,5H) 行内 [r,z,n,hx,hn] 五平面
    go  = grad_hy[:, :hsz]                   # 上游梯度 ∂L/∂hy
    gin = go * (1 - z) * (1 - n * n)         # tanh_backward
    gig = go * (hx - n) * (1 - z) * z        # sigmoid_backward（尾序 ·(1-z)·z）
    grg = gin * hn * (1 - r) * r             # sigmoid_backward（尾序 ·(1-r)·r，乘 hn 非 n）
    ghn = gin * r
    ghx = go * z                             # h' 中 z⊙h 通路直连项

    grad_input_gates  = [grg | gig | gin]    # (B,3H) 行内三段
    grad_hidden_gates = [grg | gig | ghn]    # (B,3H) 行内三段（r/z 段与 input 路逐位相同）
    grad_hx           = ghx                  # (B,H)
    grad_input_bias   = grad_input_gates .sum(axis=0)   # (3H,)；has_bias=false → golden 返回 None（比对短路跳过）；张量形态契约 (0,) 由算子定义（OpDef/InferShape）持有
    grad_hidden_bias  = grad_hidden_gates.sum(axis=0)   # (3H,)；has_bias=false → golden 返回 None（比对短路跳过）；张量形态契约 (0,) 由算子定义（OpDef/InferShape）持有

数值口径：入算子升 fp32（H2F 语义），五条梯度链乘加
与两路 bias 归约全 fp32 域完成；门梯度输出前 F2H 单次收窄（round-to-nearest-even），
bias 归约完成后一次性收窄。

双路径支持（同一核心实现，数学逐位一致）：
  - kernel / GEIR 模式：numpy.ndarray 入 → numpy.ndarray 出（bf16 走 int16 位桥保真）；
  - aclnn / e2e 模式：torch.Tensor 入 → torch.Tensor 出（bf16 原生支持）。

独立性声明：golden 独立于被测 kernel，不 import / 复用 kernel 任何产物；
竞品标杆 ``torch.ops.aten._thnn_fused_gru_cell_backward``
仅通过 ``third_party`` 属性声明，由 TTK 框架在独立子进程隔离执行——
**切勿在本文件内直接 import / 调用该竞品 API**（aclnn 进程已初始化 aclrt，同进程调用会 SIGSEGV）。

判读：三 dtype 均声明 cross_check 最严档 L2（mare/mere/rmse 比值 ≤ 2/1.2/1.2，
NPU 与 third_party 各自对 golden 的误差比值判据，见 TTK resolve.py LEVEL_PRESETS）。
"""

import numpy as np
import torch

# TestSpec 注册：
#   kernel / GEIR 模式按 CSV op_name 查找；aclnn 模式按 CSV api_name 查找。
__spec__ = {
    "thnn_fused_gru_cell_grad": "ThnnFusedGruCellGradTestSpec",
    "aclnnThnnFusedGruCellBackward": "ThnnFusedGruCellGradTestSpec",
}

# numpy dtype → torch dtype（bf16 走位桥，不进此表）
_NP_TO_TORCH = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float64): torch.float64,
}


def _is_bf16(dtype) -> bool:
    return "bfloat16" in str(dtype)


def _as_bool(value) -> bool:
    """has_bias 属性归一（CSV / attr 可能给 str / np.bool_ / int）。"""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _to_fp32_torch(x):
    """半精度（fp16/bf16）升 fp32 计算，fp32/fp64 保持原始精度。

    返回 ``(tensor, 原始dtype, 是否torch输入)``。
    """
    if isinstance(x, torch.Tensor):
        if x.dtype in (torch.float16, torch.bfloat16):
            return x.detach().to(torch.float32).contiguous(), x.dtype, True
        return x.detach().contiguous(), x.dtype, True
    arr = np.asarray(x)
    dtype = arr.dtype
    if _is_bf16(dtype):
        bits = np.ascontiguousarray(arr).view(np.int16)
        t = torch.from_numpy(bits).view(torch.bfloat16)
        return t.to(torch.float32), dtype, False
    np_dtype = np.dtype(dtype)
    if np_dtype not in _NP_TO_TORCH:
        raise ValueError(
            f"ThnnFusedGruCellGrad golden 仅支持 float32/float16/bfloat16 输入，got {np_dtype}"
        )
    if np_dtype == np.dtype(np.float16):
        arr = arr.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr)), np_dtype, False


def _cast_back(t, dtype, torch_out):
    """fp32 计算结果 → 用例 dtype 输出（F2H 单次收窄，round-to-nearest-even）。

    ``torch_out=True`` 返回 torch.Tensor（aclnn/e2e 模式）；
    否则返回 numpy.ndarray（kernel/GEIR 模式；bf16 经 uint16 位桥保真往返）。
    """
    if torch_out:
        return t.to(dtype)
    if _is_bf16(dtype):
        bits = t.to(torch.bfloat16).contiguous().view(torch.uint16)
        return bits.numpy().view(dtype)
    return t.to(_NP_TO_TORCH[np.dtype(dtype)]).contiguous().numpy()


class TorchRefImpl:
    """third_party 类形式适配器：我方输入名 storage → aten 参数名 workspace。

    bind_params 按名绑定（未知名响亮报 400），参数名必须用我方算子定义名；
    **kwargs 吞掉框架注入的 input_formats 等伪属性。
    """

    def __call__(self, grad_hy, storage, has_bias=None, **kwargs):
        return torch.ops.aten._thnn_fused_gru_cell_backward(
            grad_hy, workspace=storage, has_bias=_as_bool(has_bias)
        )


class ThnnFusedGruCellGradTestSpec:
    """ThnnFusedGruCellGrad 测试规范。

    golden 参数序 = 算子契约序（inputs grad_hy, storage → attr has_bias → 纯输出张量）；
    kernel/GEIR 模式仅传输入（属性走 kwargs），aclnn 模式按 C header 序全位置传入，
    尾随纯输出张量由 ``*pure_outputs`` 吸收（golden 不消费、不回写）。
    """

    def golden(grad_hy, storage, has_bias=False, *pure_outputs, **kwargs):
        """五路梯度融合反向，返回 [grad_input_gates, grad_hidden_gates, grad_hx,
        grad_input_bias, grad_hidden_bias]（顺序与 spec outputs 一致）。

        **kwargs 接收框架元信息（tensor_dtypes / shapes / soc_version 等）。
        """
        go32, out_dtype, torch_out = _to_fp32_torch(grad_hy)
        st32, _, _ = _to_fp32_torch(storage)

        # 形状合法性（spec shape_constraints；非法输入让 golden 显式报错，
        # 而非靠 torch 切片静默截断 / 广播出错值）
        if go32.dim() != 2 or st32.dim() != 2:
            raise ValueError(
                f"ThnnFusedGruCellGrad 仅支持 2D 输入，got rank "
                f"{go32.dim()}/{st32.dim()}"
            )
        if st32.shape[0] != go32.shape[0] or st32.shape[1] != 5 * go32.shape[1]:
            raise ValueError(
                f"shape_mismatch: grad_hy {tuple(go32.shape)} vs storage "
                f"{tuple(st32.shape)}（须满足 storage.shape == (B, 5H)）"
            )
        has_bias = _as_bool(has_bias)

        hsz = st32.shape[1] // 5
        go = go32[:, 0:hsz]
        r = st32[:, 0 * hsz : 1 * hsz]
        z = st32[:, 1 * hsz : 2 * hsz]
        n = st32[:, 2 * hsz : 3 * hsz]
        hx = st32[:, 3 * hsz : 4 * hsz]
        hn = st32[:, 4 * hsz : 5 * hsz]

        # 五链乘序逐字照抄 spec formula / CUDA kernel（左结合，尾序 ·(1-z)·z / ·(1-r)·r）
        gin = go * (1 - z) * (1 - n * n)
        gig = go * (hx - n) * (1 - z) * z
        grg = gin * hn * (1 - r) * r
        ghn = gin * r
        grad_hx32 = go * z

        input_gates32 = torch.cat((grg, gig, gin), dim=1)
        hidden_gates32 = torch.cat((grg, gig, ghn), dim=1)

        if has_bias:
            input_bias32 = input_gates32.sum(dim=0)
            hidden_bias32 = hidden_gates32.sum(dim=0)

        # has_bias=false：两路 bias 输出缺席——golden 返回 None，TTK 比对层
        # golden-None 短路跳过比对（框架 comparison.py："if golden is None →
        # SUPPRESSED"，比对层语义）。
        return [
            _cast_back(input_gates32, out_dtype, torch_out),
            _cast_back(hidden_gates32, out_dtype, torch_out),
            _cast_back(grad_hx32, out_dtype, torch_out),
            _cast_back(input_bias32, out_dtype, torch_out) if has_bias else None,
            _cast_back(hidden_bias32, out_dtype, torch_out) if has_bias else None,
        ]

    third_party = {"torch": TorchRefImpl}

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L2"},
        "float16": {"standard": "cross_check", "level": "L2"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
    }
