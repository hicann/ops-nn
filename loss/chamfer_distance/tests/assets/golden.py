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
"""ChamferDistance 多通路 golden(TestSpec 范式)。

通路支持表(照抄 01_requirement.md §3.3):
  | 通路   | 支持 | 依据                                                      |
  |--------|------|-----------------------------------------------------------|
  | kernel | ✅   | op_kernel/arch35/ 有实现                                   |
  | geir   | ✅   | op_graph/ 有 REG_OP(ChamferDistance) + IMPL_OP_INFERSHAPE  |
  | aclnn  | ❌   | canndev 老树只有 aclnn_chamfer_distance_backward.h(反向)    |
  | e2e    | ❌   | torch_npu 二进制无 aclnnChamferDistance 符号                |

输入布局 (2, B, N): xyz[0] 为全部 x 坐标、xyz[1] 为全部 y 坐标(见 01 §6.1)。
"""

import numpy as np
import torch

try:  # bfloat16 的 numpy 载体来自 ml_dtypes(TTK 同款), 缺失时只影响 bf16 用例
    import ml_dtypes  # noqa: F401
except ImportError:
    ml_dtypes = None

__spec__ = {
    # kernel + geir 共用同一个注册键(算子蛇形名), geir 不另写
    "chamfer_distance": "ChamferDistanceKernelSpec",
}

# Spec.tolerance 只认官方四标准：stat_rel_err / binary_equal / cross_check / quant
# （close、requant 是 CLI 专用别名，写进 Spec 会 InvalidSpecError）。
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    # idx1/idx2 是最近点下标, 差 1 就是另一个点, 不套容差
    "int32": {"standard": "binary_equal"},
}

# (B, N, N) 的全对比矩阵在大 N 上会撑爆内存, 按查询点分块算
_CHUNK = 512


def _widen(t):
    """只把 fp16/bf16 抬到 fp32, fp32/fp64 一律不动。

    内核计算类型 U = fp32(TIK/regbase 均在 fp32 上算距离与比较), 故窄类型要补齐
    "跨算子不落回"这一点; 但**不能写成"所有浮点一律 to(float32)"** ——
    三方活动下 TTK 开 Promote, golden 收到的是抬高一档的 dtype(fp16/bf16→fp32,
    fp32→fp64), 砍回 fp32 就撤销了 Promote, 双标杆塌成单标杆, 比值失去判别力
    (ttk_golden_logic.md §三/§五 四格 + §八 第一条误区; 实测 fp32 档两腿逐位相同)。
    这一条同时满足四格: 三方档收到 fp32/fp64 都不命中窄类型、原样不动(零 cast);
    两方档收到 fp16/bf16 则抬到 fp32、与 NPU 的加宽行为一致。
    """
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


def _min_with_index(x1, y1, x2, y2):
    """对每个查询点求到另一组的最小平方距离与最小下标。

    返回 (dist, idx): dist 形状 (B, N) fp32, idx 形状 (B, N) int32。
    并列时 torch.min 返回首个命中位置, 与内核"取最小下标"一致。
    """
    b, n = x1.shape
    # 累加器跟随入参的计算精度(Promote 后可能是 fp64), 写死 fp32 会把真值砍回三方档精度
    dist = torch.empty((b, n), dtype=x1.dtype)
    idx = torch.empty((b, n), dtype=torch.int32)
    for beg in range(0, n, _CHUNK):
        end = min(beg + _CHUNK, n)
        dx = x1[:, beg:end].unsqueeze(2) - x2.unsqueeze(1)  # (B, chunk, N)
        dy = y1[:, beg:end].unsqueeze(2) - y2.unsqueeze(1)
        d = torch.add(torch.mul(dx, dx), torch.mul(dy, dy))
        vals, pos = torch.min(d, dim=2)
        dist[:, beg:end] = vals
        idx[:, beg:end] = pos.to(torch.int32)
    return dist, idx


def _compute(xyz1, xyz2, **kwargs):
    """全程 torch.Tensor 进出, 返回 list[Tensor], 顺序照 def.cpp 的输出序。

    计算语义对齐 ascend910b 的 tbe(TIK)实现:
        d(b, i, j) = (x1[b][i] - x2[b][j])^2 + (y1[b][i] - y2[b][j])^2
        dist1/idx1 = min/argmin over j;  dist2/idx2 = min/argmin over i

    精度决策按 ttk_golden_logic.md §三的四格规则, 浮点输出与整型输出**各按各的规则**:

    - 浮点输出 dist1/dist2(判据含 cross_check): 零 cast —— 入参是 TTK Promote 抬上来的
      高精度真值, 动它就撤销了 Promote。窄类型(fp16/bf16)由 _widen 抬到 fp32, 与内核
      计算类型 U = fp32 一致。
    - 整型输出 idx1/idx2(判据 binary_equal): §三 "整型没有三方一说", 它比的是
      NPU vs golden, 规则是**按 NPU 的加宽行为决定是否 cast, NPU 不宽 -> 也不宽**。
      本算子 U = fp32, fp32 输入时内核不加宽, 故下标必须在 **fp32** 上求 argmin。
      跟着 Promote 在 fp64 上求会选出不同的最近点: 坐标差的平方在 fp32 下会下溢成 0
      (非规格化输入)或上溢成 +inf(极值输入), 判别信息被抹掉, 而 fp64 仍可区分 ——
      实测非规格化档 82/96、极值档 81/96 的下标不同, 正常值域对照档 0 不同。
      该现象**只在 fp32 声明档出现**(fp16/bf16 被 Promote 抬到 fp32 恰好等于 U)。

    出口按 §八 取**下发**的 var.dtype(即入参 xyz1.dtype), 不按算子声明的 dtype。
    """
    dt = xyz1.dtype
    p1 = _widen(xyz1)
    p2 = _widen(xyz2)
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    dist1, _ = _min_with_index(x1, y1, x2, y2)
    dist2, _ = _min_with_index(x2, y2, x1, y1)

    # 下标另在内核的计算类型 U(fp32)上求; 入参已是 fp32 时 .float() 是恒等操作,
    # 两方活动(不开 Promote)下本行不改变任何结果。
    q1 = p1 if p1.dtype == torch.float32 else p1.float()
    q2 = p2 if p2.dtype == torch.float32 else p2.float()
    _, idx1 = _min_with_index(q1[0], q1[1], q2[0], q2[1])
    _, idx2 = _min_with_index(q2[0], q2[1], q1[0], q1[1])

    return [
        dist1.to(dt).contiguous(),
        dist2.to(dt).contiguous(),
        idx1.contiguous(),
        idx2.contiguous(),
    ]


class _Compose:
    """竞品标杆(A100 上执行): 用点云库的融合最近邻算子。

    形态必须与真实使用一致 —— 用 pytorch3d 的 `knn_points`(CUDA 融合 kNN, K=1),
    而不是"广播 (B,N,N) 平方距离 + min"的分解表达式: 后者多 4 个 kernel 与一个
    O(B*N*N) 中间张量, A100 实测慢 5.8x(B=8/N=4096: 4.91ms vs 0.85ms), 拿它当基准
    会让 G/N 系统性虚高。两者结果实测一致(dist 最大差 5.8e-11, idx 零不一致)。

    pytorch3d 缺失时回退到广播实现: 只用于本机无该库时的调试, 此时测得的 G/N 不可用于交付结论。
    """

    @staticmethod
    def _pairs(xyz):
        # (2, B, N) → (B, N, 2): xyz[0]=x 平面、xyz[1]=y 平面
        return torch.stack([xyz[0], xyz[1]], dim=-1).contiguous()

    def _fallback(self, p1, p2):
        chunk = 512
        b, n, _ = p1.shape
        dist = torch.empty((b, n), dtype=torch.float32, device=p1.device)
        idx = torch.empty((b, n), dtype=torch.int32, device=p1.device)
        for beg in range(0, n, chunk):
            end = min(beg + chunk, n)
            d = p1[:, beg:end].unsqueeze(2) - p2.unsqueeze(1)
            vals, pos = torch.min((d * d).sum(-1), dim=2)
            dist[:, beg:end] = vals
            idx[:, beg:end] = pos.to(torch.int32)
        return dist, idx

    def _knn(self, p1, p2):
        try:
            from pytorch3d.ops import knn_points
        except ImportError:
            return self._fallback(p1, p2)
        out = knn_points(p1, p2, K=1, return_sorted=False)
        return out.dists[..., 0], out.idx[..., 0].to(torch.int32)

    def __call__(self, xyz1, xyz2, **kwargs):
        out_dtype = xyz1.dtype
        p1 = self._pairs(xyz1.float())
        p2 = self._pairs(xyz2.float())
        # 空点集(B==0 或 N==0): 融合算子不接空 batch/空点集, 直接空进空出
        if p1.shape[0] == 0 or p1.shape[1] == 0 or p2.shape[1] == 0:
            empty_f = p1.new_empty(p1.shape[:2])
            empty_i = torch.empty(p1.shape[:2], dtype=torch.int32, device=p1.device)
            return [empty_f.to(out_dtype), empty_f.to(out_dtype), empty_i, empty_i]
        dist1, idx1 = self._knn(p1, p2)
        dist2, idx2 = self._knn(p2, p1)
        # 浮点输出必须 cast 回 NPU 输出 dtype, 否则竞品天然更准, ratio 失真
        return [dist1.to(out_dtype), dist2.to(out_dtype), idx1, idx2]


def _as_torch(a):
    """numpy → torch。cross_check 走 golden_mode=Promote 时 bf16/fp16 输入已抬成 fp32,
    非 Promote 场景 bf16 仍是 ml_dtypes 载体、torch 不认, 这里兜底抬一档。
    """
    if a is None:
        return None
    if a.dtype.name == "bfloat16":
        a = a.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(a))


class ChamferDistanceKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray, 返 numpy.ndarray。

    参数名取自 op_host/chamfer_distance_def.cpp: xyz1 / xyz2。
    """

    def golden(*inputs, **kwargs):
        # 出口 dtype 取**下发**的 var.dtype, 不取 kwargs["output_dtypes"](算子声明的 dtype)。
        # 三方活动下 TTK 开 Promote, 下发的是抬高一档的 dtype(fp16/bf16→fp32, fp32→fp64);
        # 按声明 dtype 砍回去就撤销了 Promote, 双标杆塌成单标杆 —— 实测 fp16 声明档两腿
        # 逐位相同、判别力归零(ttk_golden_logic.md §八 第一条误区 + V1/V4)。
        # 两方活动下不开 Promote, 下发即声明 dtype, 这条规则自然退化成"等于声明 dtype"。
        sent = inputs[0].dtype
        t = [_as_torch(a) for a in inputs]
        outs = _compute(*t, **kwargs)
        # _compute 已按下发 dtype 收口, 这里唯一要做的是 bf16 的**载体**还原:
        # torch 读不了 ml_dtypes.bfloat16, _as_torch 把它抬成了 fp32, 出口落回 bf16。
        # 这是载体转换, 不是精度决策 —— 其余各档一律不动, 保证三方活动下 golden 零 cast。
        need_bf16 = sent.name == "bfloat16"
        return [
            o.numpy().astype(sent)
            if (need_bf16 and o.dtype.is_floating_point)
            else o.numpy()
            for o in outs
        ]

    third_party = {"torch": _Compose}
    tolerance = _TOL


# 【不存在】aclnn 通路: canndev 老树 op_api 只有 aclnn_chamfer_distance_backward.h(反向),
#   新树 ops/loss/ 下也只有 chamfer_distance_grad, 无前向目录(01 §3.3)。
# 【不存在】e2e(torch) 通路: torch_npu 2.10.0 的 libtorch_npu.so 无 aclnnChamferDistance 符号。
# 【不存在】tf / onnx / caffe 通路: canndev ops/built-in/framework/ 下无本算子 adapter。
