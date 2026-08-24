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

# dist 是浮点输出走 cross_check; idx 是整数输出, TTK 自动路由到逐位相等
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

# (B, N, N) 的全对比矩阵在大 N 上会撑爆内存, 按查询点分块算
_CHUNK = 512


def _min_with_index(x1, y1, x2, y2):
    """对每个查询点求到另一组的最小平方距离与最小下标。

    返回 (dist, idx): dist 形状 (B, N) fp32, idx 形状 (B, N) int32。
    并列时 torch.min 返回首个命中位置, 与内核"取最小下标"一致。
    """
    b, n = x1.shape
    dist = torch.empty((b, n), dtype=torch.float32)
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

    精度决策契约(来自 01 §6.3 的算法规格, 不是照抄被测内核): 距离与比较统一 fp32,
    最后按输出 dtype 转回; fp16 / bf16 输入同走这一条。该契约的依据是 ascend910b 的
    tbe(TIK)参考实现——它是本算子的行为基准, A5 要对齐的就是它。
    """
    dt = xyz1.dtype
    p1 = xyz1.to(torch.float32)
    p2 = xyz2.to(torch.float32)
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    dist1, idx1 = _min_with_index(x1, y1, x2, y2)
    dist2, idx2 = _min_with_index(x2, y2, x1, y1)
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
        t = [_as_torch(a) for a in inputs]
        outs = _compute(*t, **kwargs)
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        return [
            o.numpy().astype(od[i]) if i < len(od) else o.numpy()
            for i, o in enumerate(outs)
        ]

    third_party = {"torch": _Compose}
    tolerance = _TOL


# 【不存在】aclnn 通路: canndev 老树 op_api 只有 aclnn_chamfer_distance_backward.h(反向),
#   新树 ops/loss/ 下也只有 chamfer_distance_grad, 无前向目录(01 §3.3)。
# 【不存在】e2e(torch) 通路: torch_npu 2.10.0 的 libtorch_npu.so 无 aclnnChamferDistance 符号。
# 【不存在】tf / onnx / caffe 通路: canndev ops/built-in/framework/ 下无本算子 adapter。
