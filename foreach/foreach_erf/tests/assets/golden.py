#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""foreach_erf 三方 Golden — 误差函数。

误差函数 y[i] = erf(x[i])，对输入张量列表 (tensor list) 中每个张量
逐元素计算。参考实现为 torch._foreach_erf，以字符串 API 形式声明于
third_party，由测试框架在隔离的第三方进程中解析执行；各通路 golden
的计算核直接调用同一参考实现。

通路:
  - Kernel  : list[np.ndarray]  -> list[np.ndarray]
  - ACLNN   : list[torch.Tensor] -> list[torch.Tensor] (out 仅签名占位, 零读写; 结果经 return 返回)
  - E2E     : list[torch.Tensor] -> list[torch.Tensor]

Import discipline: torch 只在函数体内 lazy import — 本文件会在可能已
初始化 aclrt 的进程内被加载，模块级框架 import 有 SIGSEGV 风险;
numpy 可全局。

浮点槽到达契约 (仅 kernel/geir/aclnn 通路: 由测试框架在比对需要时对 golden
收到的浮点输入槽统一抬升一档, fp16/bf16 -> fp32, fp32 -> fp64; E2E/torch
通路的插件 golden 输入不经该抬升, 以原始 dtype 直达):
  - f64 到达槽 = 原生 fp32 的可逆抬升, 或原生 f64 输入 — 按到达精度 f64
    原生计算, 输出 f64;
  - f32 到达槽 = 原生 fp32，或原生 fp16/bf16 的提升 (二者不可区分) —
    输出 fp32 高精度真值，量化残差由输出 dtype 键控容差吸收;
  - fp16 到达槽 = 原生 fp16 (抬升后不会以 fp16 到达) — 按原生 dtype
    计算，输出 fp16;
  - E2E 通路无抬升, 一切槽以原生 dtype 到达, 按原生 dtype 计算。

标量形态: 输入张量可为任意 rank (含 rank-0 标量); 转换链路保持
输入 shape 原样 (ascontiguousarray 会把 0-d 升为 (1,)，此处以
reshape 落回原 shape)。
"""

__spec__ = {
    "foreach_erf": "ForeachErfKernelSpec",
    "aclnnForeachErf": "AclnnForeachErfSpec",
    "torch._foreach_erf": "TorchForeachErfSpec",
    "ForeachErf": "ForeachErfKernelSpec",  # GE 算子类型名变体 (OP_ADD(ForeachErf))
}

import numpy as np

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_THIRD_PARTY = {"torch": "torch._foreach_erf"}


def _to_torch_bf16_safe(t):
    """numpy → torch,兼容 bfloat16 裸到达形态。

    torch.from_numpy 不支持 ml_dtypes.bfloat16 数组;裸 bf16 槽(比对判据
    不触发框架浮点抬升时)经 uint16 位视图转入 torch.bfloat16,值逐位保持。
    其余 dtype 直通。"""
    import torch

    if "bfloat16" in str(t.dtype):
        return (
            torch.from_numpy(np.ascontiguousarray(t).view(np.uint16))
            .view(torch.bfloat16)
            .reshape(t.shape)
        )
    return torch.from_numpy(np.ascontiguousarray(t)).reshape(t.shape)


def _to_numpy_bf16_safe(t, like):
    """torch → numpy,bfloat16 结果经 uint16 位视图回投为 like.dtype。

    like=同 dtype 的输入数组(bf16 结果仅在裸 bf16 输入下产生,其 dtype 对象
    即正确目标);其余 dtype 走 t.numpy()。"""
    import torch

    if t.dtype == torch.bfloat16 and "bfloat16" in str(getattr(like, "dtype", "")):
        return t.contiguous().view(torch.uint16).numpy().view(like.dtype)
    return t.numpy()


class ForeachErfKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。"""

    def golden(x, **kwargs):
        import torch

        x_list = [_to_torch_bf16_safe(t) for t in x]
        result = torch._foreach_erf(x_list)
        return [_to_numpy_bf16_safe(r, o) for r, o in zip(result, x)]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachErfSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: AclnnForeachErfGetWorkspaceSize(x, out, workspaceSize, executor)
    out 参数仅为调用签名兼容保留 (零读写); 比对真值一律经 return 返回 —
    golden 不向 out 写入, 以免输出缓冲哨兵被覆盖 (NPU 未写/漏写行为将
    因此不可检出)。
    """

    def golden(x, out=None, **kwargs):
        import torch

        return list(torch._foreach_erf(x))

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachErfSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x, **kwargs):
        import torch

        return list(torch._foreach_erf(x))

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE
