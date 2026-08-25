#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""BN3DTrainingReduce 多通路 golden（TestSpec 范式）。

语义: BatchNorm3D 训练前向 reduce 阶段。跨批次轴 N 与全部空间轴归约，只保留通道轴。
  sum        = Σ x
  square_sum = Σ x^2
输出为原始和，不做 1/R 缩放；两个输出恒 FLOAT32，与输入 dtype 无关。

## 通路支持表（= 01 §3.3 的 ✅ 集合，`__spec__` 注册项与之逐条相等）

| 通路   | 支持 | 依据（实测） |
|--------|------|-------------|
| kernel | ✅   | `op_kernel/arch35/bn3d_training_reduce_dense_channel.h` 有 arch35 实现 |
| geir   | ✅   | `op_graph/bn3d_training_reduce_proto.h:44` 有 `REG_OP(BN3DTrainingReduce)` |
| aclnn  | ❌   | 无 `op_host/op_api/`、无 `docs/aclnn*.md`；`op_host/CMakeLists.txt:13`
|        |      | 显式 `ACLNNTYPE aclnn_exclude`（本算子图模式专用，不对外提供 aclnn） |
| e2e    | ❌   | 见文件末尾【不存在】说明 |

kernel 与 geir 共用一个注册键（算子蛇形名），故 `__spec__` 只有一条。

## 两个注册面，一份计算核

本文件同时挂 `__spec__` 与 `__golden__`，二者都指向同一个 `_golden_impl`：

| 注册面 | 用途 | 消费方 |
|--------|------|--------|
| `__spec__` | 多通路 TestSpec（带 tolerance / third_party） | 通路验证、三方精度与性能 |
| `__golden__` | 上库件约定入口，签名照 def.cpp | 仓内 CI 与静态检查 |

`get_plugin_function` 的优先级是 TestSpec > 自定义 plugin，故实际生效的始终是
`__spec__`；`__golden__` 是约定入口兼回退，不会造成两套真值。
写法照仓内既有先例 `index/non_zero_with_value/tests/assets/golden.py`。
注意 `__golden__` 指向的函数名必须在**模块层**解析——类体内引用双下划线名会被
Python 私有改写（`__x` → `_Cls__x`），故实现单列为 `_golden_impl`，类里调它。

## R3 说明（golden 只能竞品接口/拼接）

`_compute` 用 `torch.sum` / `torch.square` 竞品接口实现，不写 numpy 纯公式。

## 精度档位：本算子 golden 整体 fp64

BN3DTrainingReduce 属大规约 / 对消敏感家族（同 in_training_update_grad、
l2_normalize_grad、instance_norm_grad），单个输出槽位可归约 1e5 量级的元素。
当前内核使用 FP32 分块 Kahan / TwoSum 补偿，golden 不能复刻其切分和归约顺序，
故用独立的 CPU FP64 原生规约作高精度真值。
这是 golden 的**档位选择**，不是对框架 Promote 的 cast 干预：框架在
cross_check 下已按 DTYPE_PROMOTE_MAP 把输入抬一档（fp16/bf16→fp32、fp32→fp64）
并同步抬了 `output_dtypes`，本文件只再向上兜到 fp64，绝不向下砍——
末尾 astype 用的是 kwargs 里**已抬过的** `output_dtypes`，不是硬编码 float32。

## 随机三方回归与近零定向用例必须并存

随机 cross-check 主回归使用 `(-2, 6)` 一类非零均值区间，避免 `sum` 的真值随机
落到 0 附近后由误差比值主导统计稳定性。这不是规避正式精度门槛：抵消、近零和
特殊值必须另用固定输入定向覆盖，并按正式 FP32 混合容差验收。
"""

__spec__ = {
    # kernel + geir 同一个注册键（算子蛇形名），共用一个类，geir 不另写
    "bn3d_training_reduce": "BN3DTrainingReduceKernelSpec",
}

import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════════
# 1. 判据 tolerance —— 逐 dtype 配
# ══════════════════════════════════════════════════════════════════════
# TTK 按输出 dtype 解析判据；本算子两个输出恒为 fp32，与输入 dtype 无关。
# 因此这里只声明 float32。若保留 float16/bfloat16，它们不会被任何输出命中，
# 反而容易让维护者误以为判据随输入 dtype 变化。
# cross_check 是 NPU 相对独立 GPU 实现的补充竞争力门槛；正式精度按当前 TTK 的
# --compare close 验收，使用 FP32 的 rtol、atol 与 ptol 混合容差。
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
}


# ══════════════════════════════════════════════════════════════════════
# 2. 计算核 —— 全算子只写这一份
# ══════════════════════════════════════════════════════════════════════
# 归约轴由当前输入 format 决定：
#   NCDHW / NCHW（rank 2~5）：通道轴 = dim1，归约其余全部轴          → [C]
#   NDHWC（GEIR 外部 Data）   ：通道轴 = dim-1，归约 N/D/H/W             → [C]
#   NDC1HWC0    （rank 6）  ：[N,D,C1,H,W,C0]，归约 N/D/H/W，保留 C1、C0
#                                                                    → [1,1,C1,1,1,C0]
_NDC1HWC0_RANK = 6
_NDC1HWC0_REDUCE_DIMS = (0, 1, 3, 4)  # N, D, H, W
_CHANNEL_DIM = 1


def _input_format(kwargs):
    """取当前输入 format。

    kernel 直调收到的是已选定的 storage format；GEIR 的公开 NDHWC
    用例收到的是外部 Data format。当 third_party 使用逻辑原布局数组时，
    TTK 会把 input_ori_formats 作为 input_formats 传入。因此该字段始终与
    当前计算实际收到的数组布局一致。
    """
    fmts = kwargs.get("input_formats") or ()
    if not fmts:
        return ""
    fmt = fmts[0]
    fmt = fmt[0] if isinstance(fmt, (list, tuple)) else fmt
    return str(fmt).upper()


def _is_ndc1hwc0(x, kwargs):
    """判布局。优先信 storage format，缺失时回退到 rank。

    两者本不会打架：channel-first 只支持 rank 2~5、NDC1HWC0 固定 rank 6，
    但 format 是 tiling 的实际依据，能拿到就不靠 rank 推。
    """
    fmt = _input_format(kwargs)
    if fmt:
        return fmt == "NDC1HWC0"
    return x.ndim == _NDC1HWC0_RANK


def _is_ndhwc(kwargs):
    """GEIR 外部 Data 可直接是 NDHWC，golden 应与其物理排列一致。"""
    return _input_format(kwargs) == "NDHWC"


def _compute(x, **kwargs):
    """返回 [sum, square_sum]，顺序照 bn3d_training_reduce_def.cpp 的输出序。

    fp64 真值（见文件头「精度档位」）。禁融合算子：先 square 再 sum，两步拼接。
    """
    x64 = x.to(torch.float64)

    if _is_ndc1hwc0(x, kwargs):
        dims = list(_NDC1HWC0_REDUCE_DIMS)
        keepdim = True  # 保留 C1、C0 的同时把 N/D/H/W 压成 1 → [1,1,C1,1,1,C0]
    elif _is_ndhwc(kwargs):
        dims = list(range(x64.ndim - 1))
        keepdim = False
    else:
        dims = [d for d in range(x64.ndim) if d != _CHANNEL_DIM]
        keepdim = False  # channel-first 输出压成一维 [C]

    s = torch.sum(x64, dim=dims, keepdim=keepdim)
    sq = torch.sum(torch.square(x64), dim=dims, keepdim=keepdim)
    return [s, sq]


# ══════════════════════════════════════════════════════════════════════
# 3. third_party —— 三方精度 + 三方性能共用
# ══════════════════════════════════════════════════════════════════════
class _Compose:
    """竞品标杆：远端 GPU server 上用 torch 原生归约算出同语义结果。

    当前 TTK 的 cross_check 与 xpu_perf 共用这一个 third_party，因此这里使用
    竞品 GPU 上真实、原生且可计时的 `torch.sum` / `torch.square`。数学语义
    与算子对齐，但不复制 NPU 内部的 Kahan/TwoSum、分块大小或归约树；
    那些是 NPU 实现细节，复制进竞品腿会改变真实性能标杆。独立高精度真值
    由 CPU fp64 `_compute` 承担，本类仅承担 GPU fp32 竞品腿。

    浮点输出必须 cast 回 NPU 的输出 dtype，否则竞品对 golden 的误差远小于 NPU，
    ratio 会凭空爆表。本算子两个输出恒 fp32。
    """

    def __init__(self, **kwargs):
        # 权威 REG_OP 无属性，无可绑定项。
        self._kwargs = kwargs

    def __call__(self, x, **kwargs):
        x32 = x.to(torch.float32)

        if _is_ndc1hwc0(x, kwargs):
            dims = list(_NDC1HWC0_REDUCE_DIMS)
            keepdim = True
        elif _is_ndhwc(kwargs):
            dims = list(range(x32.ndim - 1))
            keepdim = False
        else:
            dims = [d for d in range(x32.ndim) if d != _CHANNEL_DIM]
            keepdim = False

        s = torch.sum(x32, dim=dims, keepdim=keepdim)
        sq = torch.sum(torch.square(x32), dim=dims, keepdim=keepdim)
        # 两个输出恒 fp32 —— 与 def.cpp 的 output DataType 一致
        return [s.to(torch.float32), sq.to(torch.float32)]


# ══════════════════════════════════════════════════════════════════════
# 4. numpy 容器壳 —— 两个注册面共用这一份，不做计算
# ══════════════════════════════════════════════════════════════════════
def _golden_impl(x, **kwargs):
    """收 numpy.ndarray，返 numpy.ndarray。kernel 与 geir 通路同一形态。

    参数名取自 op_host/bn3d_training_reduce_def.cpp：输入只有 x，无属性。
    """
    # bfloat16 的 numpy 表示（ml_dtypes）torch 不认，先过一次 float32 再进 fp64。
    arr = np.ascontiguousarray(x)
    if "bfloat16" in str(arr.dtype):
        arr = arr.astype(np.float32)
    t = torch.from_numpy(arr)

    outs = _compute(t, **kwargs)

    # 照 kwargs 里的 output_dtypes 转回。cross_check 下框架已把它抬过一档，
    # 这里跟着走即可；**不要**硬编码 float32，那会把 fp64 真值砍回去。
    od = kwargs.get("output_dtypes") or []
    od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
    return [
        o.numpy().astype(od[i]) if i < len(od) else o.numpy()
        for i, o in enumerate(outs)
    ]


def __golden_bn3d_training_reduce(x, **kwargs):
    """`__golden__` 的约定入口。实现在 `_golden_impl`，此处只转发。"""
    return _golden_impl(x, **kwargs)


# ══════════════════════════════════════════════════════════════════════
# 5. 通路壳 —— 只做容器转换
# ══════════════════════════════════════════════════════════════════════
class BN3DTrainingReduceKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray，返 numpy.ndarray。"""

    def golden(x, **kwargs):
        # 调 _golden_impl 而非 __golden_bn3d_training_reduce：后者在类体内会被
        # 私有改写成 _BN3DTrainingReduceKernelSpec__golden_bn3d_training_reduce。
        return _golden_impl(x, **kwargs)

    third_party = {"torch": _Compose}
    tolerance = _TOL


# 上库件约定入口。与 `__spec__` 指向同一个 `_golden_impl`，不存在两套真值。
__golden__ = {"kernel": {"bn3d_training_reduce": "__golden_bn3d_training_reduce"}}


# ---------------------------------------------------------------------------
# 【不存在】aclnn 通路：本算子图模式专用，不对外提供 aclnn 接口。
#   实测依据：无 op_host/op_api/ 目录、无 docs/aclnn*.md；
#             op_host/CMakeLists.txt:13 显式 `ACLNNTYPE aclnn_exclude`。
#   与 canndev 一致 —— Ascend 950PR/DT 上的 aclnnBatchNorm 仍走 BatchNormV3 单算子，
#   不经过本算子。
#
# 【不存在】e2e 通路：torch_npu 不会执行本算子（aclnn 都没有，无从绑定）。
#   实测依据（查二进制符号，非 dir(torch.ops.npu) 前缀推断）：
#     SO=.../torch_npu/lib/libtorch_npu.so
#     strings $SO | grep -c 'aclnnBN3DTrainingReduce'   → 0
#   防伪 ①（方法有效性对照组）：
#     strings $SO | grep -o 'aclnnBatchNorm[A-Za-z0-9]*' → aclnnBatchNorm /
#     aclnnBatchNormBackward / aclnnBatchNormElemt ... 有命中，证明查法能查到符号。
#   防伪 ②（排除 torch_npu 版本滞后）：本算子 aclnn 接口自始不存在（aclnn_exclude），
#     不是"torch_npu 还没跟上"，故 0 命中是结论而非时序假象；
#     同族已入仓的 INTrainingReduceV2 同样 0 命中，与之一致。
# ---------------------------------------------------------------------------
