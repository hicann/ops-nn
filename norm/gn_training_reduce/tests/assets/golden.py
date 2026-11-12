#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""TTK TestSpec golden for gn_training_reduce (GroupNorm training statistical reduction)."""

__spec__ = {
    "gn_training_reduce": "GNTrainingReduceTestSpec",  # legacy official IR op name
}

import numpy as np
import torch

_NCHW = "NCHW"
_NHWC = "NHWC"


def _flatten_tokens(value):
    """Yield every scalar token inside a (possibly nested) format structure."""
    if value is None:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_tokens(item)
    elif hasattr(value, "tolist"):
        yield from _flatten_tokens(value.tolist())
    else:
        yield value


def _resolve_format(kwargs):
    """Layout of x -- purely format-driven, never guessed.

    TTK always supplies the input format (``input_formats`` is carried inside the
    X-Input-Schema and handed to the golden/compose), and for this rank-4 op an
    ambiguous shape (space dim equal to C) is resolved by the format alone.

    Only NCHW / NHWC are meaningful here, so the first such token wins.  When no
    format is supplied the golden must fail loudly instead of assuming NCHW:
    assuming NCHW for an NHWC input reads the channel axis off the wrong dim,
    which either crashes the competitor outright or -- worse -- lands on a
    different-but-legal grouping and silently reports a bogus comparison.
    """
    for key in (
        "input_formats",
        "input_ori_formats",
        "tensor_formats",
        "tensor_ori_formats",
    ):
        for token in _flatten_tokens(kwargs.get(key)):
            token = str(token).upper()
            if token in (_NCHW, _NHWC):
                return token
    raise ValueError(
        "no input format for layout resolution (need input_formats in kwargs): "
        "format-ish keys present=%s" % sorted(k for k in kwargs if "format" in k)
    )


# 归约分块宽度:每块内部 fp32 求和,块结果再 fp32 成对归约(same dtype end to end)。
# 见 `_chunked_row_sum` 的注释 —— 这是纯规约顺序调整,不改变任何语义或 dtype 档位。
_REDUCE_CHUNK = 8192


def _grouped_rows(x, num_groups, fmt):
    """View x as ``[outer=N*G, R]`` — one row per output, reduction on the last axis.

    NCHW ``(N,C,H,W)``: the grouped view ``(N,G,D,H,W)`` already keeps D,H,W as its
    trailing dims, so flattening is a pure reshape (no data movement).

    NHWC ``(N,H,W,C)``: the grouped view ``(N,H,W,G,D)`` interleaves the group axis
    inside the reduced dims, so ``torch.sum(..., dim=(1,2,4))`` runs over strided
    axes.  Moving G next to N first makes the reduction one contiguous last axis.
    That layout change alone is worth ~30x on the fp32 golden: on the dumped
    L1_309/L1_496 inputs the strided multi-axis sum sits 33-99 ULP away from the
    fp64 truth, while the same elements reduced as a contiguous axis sit at 1-2 ULP
    (the strided path is the slow scalar-accumulate iterator, not the blocked one).
    """
    g = int(num_groups)
    if fmt == _NHWC:
        n, h, w, c = (int(s) for s in x.shape)
        d = c // g
        xg = torch.reshape(x, (n, h, w, g, d))
        xg = torch.permute(xg, (0, 3, 1, 2, 4))  # (N,G,H,W,D)
        return torch.reshape(xg, (n * g, h * w * d))
    n, c, h, w = (int(s) for s in x.shape)
    d = c // g
    return torch.reshape(x, (n * g, d * h * w))


def _pairwise_sum_last(x):
    """Balanced pairwise sum along the last (contiguous) axis of a 2D tensor."""
    while int(x.shape[1]) > 1:
        width = int(x.shape[1])
        half = width // 2
        head = torch.sum(torch.reshape(x[:, : 2 * half], (-1, half, 2)), dim=2)
        if width % 2:
            x = torch.cat((head, x[:, 2 * half :]), dim=1)
        else:
            x = head
    return x[:, 0]


def _chunked_row_sum(rows, square=False):
    """fp32 sum of every row: blockwise fp32 sums, then a pairwise fp32 merge.

    A single long ``torch.sum`` over R = 4e5 .. 1.7e8 elements accumulates in one
    long chain, so its own fp32 rounding grows with R: on the dumped long-reduce
    cases the current golden is 33.9 (L1_160, R=9.2e6) and 41.0 (L1_309, R=4.2e6)
    fp32 ULP away from the fp64 truth — larger than the 32 ULP judgement cap, i.e.
    the reference noise alone exhausts the two-party budget.  Splitting the axis
    into ``_REDUCE_CHUNK`` blocks bounds the chain, and merging the block results
    as a balanced tree makes the total depth ~log2(R) instead of R.  Same
    dtype throughout: fp32 blocks -> fp32 merge (fp64 only ever arrives as the
    Promote-tier input dtype and is then reduced in fp64 as before).
    """
    width = int(rows.shape[1])
    if width == 0:  # empty reduce: torch.sum over an empty dim is 0
        return torch.zeros((int(rows.shape[0]),), dtype=rows.dtype, device=rows.device)
    parts = []
    for start in range(0, width, _REDUCE_CHUNK):
        block = rows[:, start : start + _REDUCE_CHUNK]
        if square:
            block = block * block
        parts.append(torch.sum(block, dim=1))
    if len(parts) == 1:
        return parts[0]
    return _pairwise_sum_last(torch.stack(parts, dim=1))


def _torch_group_reduce(x, num_groups, fmt):
    """tier-3 golden body: torch.reshape + torch.square + torch.sum, at the NPU's width.

    加宽档**跟随 NPU**(golden_standard.md §三/§五「浮点 + 两方泛化」):arch35 内核的
    累加寄存器与输出恒为 fp32 —— fp16/bf16 入口升 fp32、fp32 入口原样,两级归约与
    cache 缓冲全是 `<float>`,出口直接写 fp32 不窄回。
    故此处只对 fp16/bf16 抬 fp32;**不无条件抬 fp64**(NPU 从不宽到 fp64)。
    Promote(三方档)下入参与 output_dtypes 均已被 TTK 抬到 fp64,原样保留 = 零 cast
    (§三/§四「浮点 + 三方」格),故抬档判断只看入参 dtype,不写死任何档位。
    出口**不写死 fp32**:dtype 由 `_group_reduce` 按 TTK **下发**的 output_dtypes 决定 ——
    Promote(三方档)下下发值是抬档后的 fp64,写死 fp32 会把真值砍回(golden_standard.md §八#2)。

    归约**只在同一个 dtype 内做**,唯一变化是规约顺序:`_grouped_rows` 把每个输出的
    被归约元素摊到最后一根连续轴上,再按 `_REDUCE_CHUNK` 分块 + 块间成对合并。
    语义(哪些元素进同一个输出、NCHW/NHWC 轴、num_groups、D=C/G)与逐元素加法完全相同,
    只是把一条 O(R) 的累加链换成 O(log R) 的树 —— 见 `_chunked_row_sum`。
    """
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)
    rows = _grouped_rows(x, num_groups, fmt)
    sum_out = _chunked_row_sum(rows)
    square_sum_out = _chunked_row_sum(rows, square=True)
    g = int(num_groups)
    if fmt == _NHWC:
        out_shape = (int(x.shape[0]), 1, 1, g, 1)
    else:
        out_shape = (int(x.shape[0]), g, 1, 1, 1)
    return torch.reshape(sum_out, out_shape), torch.reshape(square_sum_out, out_shape)


def _torch_oracle_group_reduce(x, num_groups, fmt):
    """Independent competitor (third_party): fp32 ones-vector multiply-accumulate.

    The competitor mirrors the arch35 kernel's float32 arithmetic, which is the
    operator contract (``spec.yaml accumulator_dtype: float32``): fp16/bf16 lanes
    are promoted to fp32, the reduction accumulates in fp32, and both outputs are
    stored as fp32.  An fp64 competitor would be effectively exact, collapsing the
    cross_check ratio denominator onto the ``small_value`` floor and turning a
    peer-to-peer comparison into an absolute-error-vs-floor test.

    Same value semantics as the formula but a different computation form from the
    golden's torch.sum: the group axis is flattened and reduced with torch.matmul
    against a ones vector, so the two references do not share a summation code path.
    """
    g = int(num_groups)
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.ascontiguousarray(np.asarray(x)))
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)
    if fmt == _NHWC:
        n, h, w, c = (int(s) for s in x.shape)
        d = c // g
        xg = torch.reshape(x, (n, h, w, g, d))
        xg = torch.reshape(torch.permute(xg, (0, 3, 1, 2, 4)), (n, g, h * w * d))
        out_shape = (n, g, 1, 1, 1)
        permute_to_nhwc = True
    else:
        n, c, h, w = (int(s) for s in x.shape)
        d = c // g
        xg = torch.reshape(x, (n, g, d * h * w))
        out_shape = (n, g, 1, 1, 1)
        permute_to_nhwc = False

    ones = torch.ones((xg.shape[2], 1), dtype=xg.dtype, device=xg.device)
    sum_out = torch.reshape(torch.matmul(xg, ones), out_shape)
    square_sum_out = torch.reshape(torch.matmul(torch.square(xg), ones), out_shape)
    if permute_to_nhwc:  # (N,G,1,1,1) -> (N,1,1,G,1)
        sum_out = torch.permute(sum_out, (0, 2, 3, 1, 4))
        square_sum_out = torch.permute(square_sum_out, (0, 2, 3, 1, 4))
    return sum_out.to(torch.float32), square_sum_out.to(torch.float32)


# The competitor is a small-operator composition (reshape/permute/matmul); for a
# fair XPU performance baseline it must be compiled (kernel fusion) rather than
# paid as eager launch overhead.  Fail-open: if torch.compile is unavailable the
# eager implementation is kept, so golden/cross_check semantics are unaffected.
try:
    _torch_oracle_group_reduce = torch.compile(
        _torch_oracle_group_reduce, dynamic=False
    )
except Exception:  # pragma: no cover - torch.compile optional
    pass


_STAT_DTYPE_ALIAS = {
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}


def _normalize_dtype_name(dtype):
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0] if dtype else None
    if dtype is None:
        return None
    return str(dtype).lower().replace("torch.", "").replace("numpy.", "")


def _dtype_from_token(token):
    return _STAT_DTYPE_ALIAS.get(_normalize_dtype_name(token))


def _group_reduce(x, num_groups, fmt, output_dtypes=None):
    """Dispatch on the input container: torch (ACLNN) vs numpy (Kernel/GEIR).

    出口 dtype 取 TTK **下发**的 `output_dtypes`;
    下发缺省时回落到算子契约 dtype fp32(两方泛化档跟 NPU 输出 dtype)。
    不读算子"声明"的 dtype —— 那会把 Promote 抬上去的精度砍回来。
    """
    is_torch = isinstance(x, torch.Tensor)
    x_t = x if is_torch else torch.from_numpy(np.ascontiguousarray(np.asarray(x)))
    outs = list(_torch_group_reduce(x_t, num_groups, fmt))
    names = [_normalize_dtype_name(d) for d in (output_dtypes or ())]
    result = []
    for index, out in enumerate(outs):
        target = _dtype_from_token(names[index]) if index < len(names) else None
        if target is None:
            target = torch.float32
        out = out.to(target)
        result.append(out if is_torch else out.numpy())
    return result


class GNTrainingReduceTorchOracle:
    """reference_oracle competitor, bound by name to the op's x / num_groups.

    The xpu-server binds parameters by name, so ``num_groups`` must stay verbatim
    (spec attributes[].name); a renamed parameter would silently fall back to the
    default and the competitor would run the wrong grouping.
    """

    def __init__(self, num_groups=2, **kwargs):
        self.num_groups = int(num_groups)

    def __call__(self, x, **kwargs):
        fmt = _resolve_format(
            kwargs
        )  # no layout attr exists; defaults to canonical NCHW
        return list(_torch_oracle_group_reduce(x, self.num_groups, fmt))


class GNTrainingReduceTestSpec:
    """gn_training_reduce TestSpec: golden + third_party + tolerance.

    The same class serves both flows: Kernel/GEIR pass numpy.ndarray (converted to
    torch inside and back), ACLNN/E2E pass torch.Tensor.  sumOut / squareSumOut are
    the ACLNN output slots (positional in the C header); the reference math returns
    freshly computed tensors regardless.
    """

    def golden(x, num_groups=2, sumOut=None, squareSumOut=None, **kwargs):
        fmt = _resolve_format(kwargs)
        return _group_reduce(x, num_groups, fmt, kwargs.get("output_dtypes"))

    third_party = {"torch": GNTrainingReduceTorchOracle}

    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }
