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
"""ArgMaxGradD 多通路 golden(TestSpec 范式)。

通路支持表(照抄 01_requirement.md §3.3):
  | 通路   | 支持 | 依据                                                        |
  |--------|------|-------------------------------------------------------------|
  | kernel | ✅   | op_kernel/arch35/ 有实现                                     |
  | geir   | ✅   | op_graph/ 有 REG_OP(ArgMaxGradD) + IMPL_OP_INFERSHAPE        |
  | aclnn  | ❌   | canndev op_api/inc/level2 无 aclnn_arg_max_grad*.h            |
  | e2e    | ❌   | 无 aclnn 接口, torch 无从下发                                 |

语义: y[o,k,i] = (assist[o,k,i] == indices[o,0,i]) ? updates[o,0,i] : var[o,k,i]
(assist 是沿 dimension 轴的序号矩阵, 由 ArgMaxGradFusionPass 生成; 见 01 §6.1/§6.2)
"""

import numpy as np
import torch

__spec__ = {
    # kernel + geir 共用同一个注册键(算子蛇形名), geir 不另写
    "arg_max_grad_d": "ArgMaxGradDKernelSpec",
}

# 本算子只做条件选择、无任何算术: var/updates 的值原样搬运, 浮点输出也必须与 golden **逐位相等**,
# 因此浮点档同样声明 binary_equal(比 cross_check 更严, 且不依赖三方腿);
# 整数输出本就由 TTK 硬路由到逐位相等。三方对比另由 --compare cross_check 在精度档单独跑。
_TOL = {
    "float32": {"standard": "binary_equal"},
    "float16": {"standard": "binary_equal"},
}


def _attr(kwargs, name, default):
    """attributes 可能平铺在 kwargs, 也可能收在 kwargs['attributes'] dict。"""
    v = kwargs.get(name)
    if v is None:
        attrs = kwargs.get("attributes")
        if isinstance(attrs, dict):
            v = attrs.get(name)
    return default if v is None else v


def _compute(var, indices, updates, assist, dimension):
    """torch.Tensor 进 / 出。用 torch.where 显式表达 A2 的 vcmp+vsel 语义。

    不用 torch.scatter_ 作 golden 的原因: indices 越界时 scatter_ 直接抛异常,
    而本算子契约是"越界即不命中、整轴保留 var"(01 §2), where 形态能覆盖该行为。
    """
    dim = dimension if dimension >= 0 else dimension + var.dim()
    # indices/updates 沿 dim 轴长度为 1(允许该轴被 squeeze 掉, 这里统一还原成 keepdim 形态)
    keep_shape = list(var.shape)
    keep_shape[dim] = 1
    idx = indices.reshape(keep_shape)
    upd = updates.reshape(keep_shape)

    hit = torch.eq(assist, idx.expand_as(assist))
    return [torch.where(hit, upd.expand_as(var).to(var.dtype), var)]


class _ArgMaxGradDCompose:
    """三方标杆(A100 上执行): 用 torch 的融合散射接口 Tensor.scatter。

    形态与真实使用一致 —— 单个融合算子, 不用 "eq + where" 的分解表达式。
    参数名与 def.cpp 逐字一致(var / indices / updates / assist)。
    """

    def __init__(self, *, dimension=0, **_):
        self.dimension = int(dimension)

    def __call__(self, var, indices, updates, assist, **_):
        dim = self.dimension if self.dimension >= 0 else self.dimension + var.dim()
        keep_shape = list(var.shape)
        keep_shape[dim] = 1
        idx = indices.reshape(keep_shape).to(torch.int64)
        upd = updates.reshape(keep_shape).to(var.dtype)
        return [var.scatter(dim, idx, upd)]


class ArgMaxGradDKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray, 返 numpy.ndarray。

    参数名取自 op_host/arg_max_grad_d_def.cpp: var / indices / updates / assist。
    """

    def golden(var, indices, updates, assist, **kwargs):
        dimension = int(_attr(kwargs, "dimension", 0))
        outs = _compute(
            torch.from_numpy(np.ascontiguousarray(var)),
            torch.from_numpy(np.ascontiguousarray(indices)),
            torch.from_numpy(np.ascontiguousarray(updates)),
            torch.from_numpy(np.ascontiguousarray(assist)),
            dimension,
        )
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        return [
            o.numpy().astype(od[i]) if i < len(od) else o.numpy()
            for i, o in enumerate(outs)
        ]

    def customize_inputs(var, indices, updates, assist, **kwargs):
        """assist 与 indices 不能用随机数据, 必须按语义构造(参数名与 def.cpp 逐字一致)。

        - assist: 沿 dimension 轴的序号矩阵, 与 var 同形 —— 图上它由 ArgMaxGradFusionPass
          生成为常量(01 §6.1), 随机值会让整个用例失去意义。
        - indices: 收敛到 [0, D) —— 随机 int32 几乎必然越界, 越界时全轴保留 var,
          用例就退化成"恒等拷贝", 覆盖不到命中路径。
        """
        dimension = int(_attr(kwargs, "dimension", 0))
        dim = dimension if dimension >= 0 else dimension + var.ndim
        # DFX 负向用例会传越界的 dimension: 这里不能抛异常(否则连输入都生成不出来,
        # 拿不到"算子按预期拒收"的证据), 原样返回让 host 校验去拒收。
        if dim < 0 or dim >= var.ndim:
            return (var, indices, updates, assist)
        axis_len = var.shape[dim]
        if axis_len > 0:
            seq_shape = [1] * var.ndim
            seq_shape[dim] = axis_len
            assist[...] = np.arange(axis_len, dtype=assist.dtype).reshape(seq_shape)
            indices[...] = np.abs(indices) % axis_len
        return (var, indices, updates, assist)

    third_party = {"torch": _ArgMaxGradDCompose}
    tolerance = _TOL


def arg_max_grad_d_golden(var, indices, updates, assist, **kwargs):
    """保留 __golden__ 约定入口(上库件, 签名照 def.cpp), 与 Spec 共用同一实现。"""
    return ArgMaxGradDKernelSpec.golden(var, indices, updates, assist, **kwargs)[0]


__golden__ = {"kernel": {"arg_max_grad_d": "arg_max_grad_d_golden"}}

# 【不存在】aclnn 通路: canndev op_api/inc/level2 下无 aclnn_arg_max_grad*.h(01 §3.3)。
# 【不存在】e2e(torch) 通路: 本算子无 aclnn 接口, torch dispatcher 无从下发。
# 【不存在】tf / caffe 通路: canndev ops/built-in/framework/ 下无本算子 adapter;
#   onnx 通路存在但落点是 NPUScatter→ArgMaxGrad 的图映射, 不是单算子调用, 不在 TTK 覆盖范围。
