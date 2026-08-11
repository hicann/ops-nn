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
"""
TTK TestSpec for non_zero_with_value (kernel / GEIR 通路, arch35 / Ascend950).

三份资产各司其职：
    golden       —— 真值，torch 算子拼接（nonzero 出坐标 + 布尔索引出值 + count，
                   非 numpy 纯公式，红线 R3）
    third_party  —— 三方标杆，同一 torch 拼接在远端 GPU 上执行
    tolerance    —— 三输出全是"精确搬运"语义（gather + 计数，无任何算术），
                   NPU 正确即逐位相等，统一 binary_equal；不走 cross_check
                   （真值与 NPU 误差恒为 0，比值 0/0 无意义）。三方精度腿因此
                   天然不涉及，三方性能由 --xpu-perf 正常产出。

Golden 参照 = torch.nonzero（竞品接口，红线 R3：禁 numpy 纯公式 golden）。
torch 裁定非零集合与行主序次序，本文件只把结果搬进算子的静态 max-size 布局。
torch.nonzero 未实现 uint16/uint32/uint64 → 用同宽有符号视图定位（零值判定按位等价），
value 仍从原数组取，dtype 不变。

Semantics (transpose=true 坐标主序):
    N     = torch.nonzero(x) 的行数    # nan 与 +-inf 计非零；-0.0 视为零，行主序
    count = [N]
    value = 静态 max-size [numel]，前 N 段 = x[mask]（行主序），其余为无效预留区(0 填充)
    index = 静态 max-size [2*numel]，坐标主序展平：
                index[0:N]           = 非零元素行号(行主序)
                index[numel:numel+N] = 非零元素列号(行主序)
            其余为无效预留区(0 填充)

三输出均为静态 max-size；有效长度由 count 给出，尾部无效预留区在 NPU 上未定义。
TTK 默认整块比对 buffer，会把这段未定义尾部算进判定（通过率恰好退化成非零密度），
故本文件同时提供 NonZeroWithValueSpec.compare：只比有效前缀 [0:N]，由 TTK 调用并
据其返回值输出该 case 的最终 PASS/FAIL。golden 仍对尾部补 0，仅为形状对齐。

Canonical IO order (non_zero_with_value_def.cpp):
    input : x（严格 2D，dtype-generic）
    output: value(同 x dtype), index(int32), count(int32)
    attrs : transpose(OPTIONAL bool=false), dtype(OPTIONAL int=DT_INT32)
"""

import numpy as np
import torch

# Spec.tolerance 只认官方四标准：stat_rel_err / binary_equal / cross_check / quant
# （close、requant 是 CLI 专用别名，写进 Spec 会 InvalidSpecError）。
# 本算子三输出均为精确搬运语义（gather 输入位 + 坐标 + 计数），正确即逐位相等，
# 故连浮点 value 也配 binary_equal（cross_check 比值 0/0 无意义）。
_TOL = {
    "float32": {"standard": "binary_equal"},
    "float16": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
}

# torch.nonzero 未实现的无符号类型 → 同宽有符号视图。零值判定等价于「全 bit 为 0」,
# 有/无符号解释不改变该判定,故只用于定位非零位置;value 仍从原数组取,dtype 不变。
_UNSIGNED_VIEW = {
    np.dtype("uint16"): "i2",
    np.dtype("uint32"): "i4",
    np.dtype("uint64"): "i8",
}


def _golden_impl(x, **kwargs):
    """golden 参照 = torch.nonzero(竞品接口),不用 numpy 公式复刻内核逻辑(红线 R3)。

    torch 裁定「哪些元素非零」及其行主序次序(nan/±inf 非零、-0.0 为零,与内核语义一致);
    本函数只做竞品结果 → 算子静态 max-size 布局的搬运(取值 + 补零对齐),不重新实现语义。
    """
    xa = np.ascontiguousarray(x)
    numel = xa.size

    probe = xa.view(_UNSIGNED_VIEW[xa.dtype]) if xa.dtype in _UNSIGNED_VIEW else xa
    coord = torch.nonzero(torch.from_numpy(probe))  # [N, 2] 行主序 (row, col)
    n = int(coord.shape[0])
    rows = coord[:, 0].numpy().astype(np.int32)
    cols = coord[:, 1].numpy().astype(np.int32)

    value = np.zeros((numel,), dtype=xa.dtype)
    value[:n] = xa[rows, cols]  # 值取自原数组,保持原 dtype

    index = np.zeros((2 * numel,), dtype=np.int32)
    index[:n] = rows
    index[numel : numel + n] = cols

    count = np.array([n], dtype=np.int32)
    return [value, index, count]


class _NzvCompose:
    """三方标杆:同一 torch 拼接在远端 GPU 上执行(nonzero 出坐标 + 取值 + 补零预留区)。

    参数名与 def.cpp 逐字一致(x);三输出为精确搬运,dtype 随输入/约定,无需 cast。
    """

    def __init__(self, **_):
        pass

    def __call__(self, x, **_):
        numel = x.numel()
        coord = torch.nonzero(x)  # [N, 2] 行主序,与 golden 同一 torch 接口裁定
        n = int(coord.shape[0])
        rows, cols = coord[:, 0], coord[:, 1]

        value = torch.zeros((numel,), dtype=x.dtype, device=x.device)
        index = torch.zeros((2 * numel,), dtype=torch.int32, device=x.device)
        if n:
            value[:n] = x[rows, cols]
            index[:n] = rows.to(torch.int32)
            index[numel : numel + n] = cols.to(torch.int32)

        count = torch.tensor([n], dtype=torch.int32, device=x.device)
        return [value, index, count]


def _bits(a):
    """按位视图,保持逐元素粒度:NaN 逐位相等即视为一致(np.equal 下 NaN != NaN,不能用)。

    视成同宽无符号整型而非 uint8,否则精度百分比会退化成按字节统计,与 index 的按元素
    统计口径不一致(同一份 CSV 里两个数含义不同)。
    """
    arr = np.ascontiguousarray(a)
    return (
        arr.view(f"u{arr.dtype.itemsize}")
        if arr.dtype.itemsize in (1, 2, 4, 8)
        else arr.view(np.uint8)
    )


def _verdict(matched, total):
    """total==0(空前缀)时判 PASS,精度记 100%。"""
    if total == 0:
        return True, 100.0
    return matched == total, matched / total * 100.0


class NonZeroWithValueSpec:
    """kernel / GEIR 通路 spec:golden + 有效前缀比对。

    compare 的判定结果由 TTK 汇总成 case 的 precision_status(见 ttk
    core_modules/comparison/custom.py::try_custom_compare),不是本文件自行下结论。
    """

    def golden(x, **kwargs):
        return _golden_impl(x, **kwargs)

    def compare(*outputs, **kwargs):
        """只比有效前缀 [0:N];尾部预留区在 NPU 上未定义,不参与判定。

        outputs = (npu_value, npu_index, npu_count, gold_value, gold_index, gold_count)
        """
        npu_value, npu_index, npu_count, gold_value, gold_index, gold_count = outputs[
            :6
        ]
        n = int(np.asarray(gold_count).reshape(-1)[0])
        npu_n = int(np.asarray(npu_count).reshape(-1)[0])
        numel = np.asarray(npu_value).size

        # count:非零元素个数,整块比(本身就只有 1 个元素)
        count_ok = npu_n == n

        # value:前 N 个非零值逐位比
        v_npu, v_gold = (
            np.asarray(npu_value).reshape(-1),
            np.asarray(gold_value).reshape(-1),
        )
        v_matched = (
            int(np.count_nonzero(_bits(v_npu[:n]) == _bits(v_gold[:n]))) if n else 0
        )
        v_total = _bits(v_gold[:n]).size if n else 0
        v_ok, v_pct = _verdict(v_matched, v_total)

        # index:坐标主序两段——行号 [0:N]、列号 [numel:numel+N]
        i_npu, i_gold = (
            np.asarray(npu_index).reshape(-1),
            np.asarray(gold_index).reshape(-1),
        )
        seg = [(0, n), (numel, numel + n)]
        i_matched = sum(
            int(np.count_nonzero(i_npu[a:b] == i_gold[a:b])) for a, b in seg
        )
        i_ok, i_pct = _verdict(i_matched, 2 * n)

        return [
            {
                "pass": v_ok and count_ok,
                "precision": v_pct,
                "error_info": None
                if v_ok
                else f"value valid-prefix[0:{n}] mismatch {v_total - v_matched}",
            },
            {
                "pass": i_ok and count_ok,
                "precision": i_pct,
                "error_info": None
                if i_ok
                else f"index valid-prefix mismatch {2 * n - i_matched}",
            },
            {
                "pass": count_ok,
                "precision": 100.0 if count_ok else 0.0,
                "error_info": None if count_ok else f"count NPU={npu_n} golden={n}",
            },
        ]

    third_party = {"torch": _NzvCompose}
    tolerance = _TOL


def __golden_non_zero_with_value(x, **kwargs):
    # 保留 __golden__ 约定入口;实现在 _golden_impl(类体内引用双下划线名会被 Python 私有改写)。
    return _golden_impl(x, **kwargs)


__spec__ = {"non_zero_with_value": "NonZeroWithValueSpec"}
__golden__ = {"kernel": {"non_zero_with_value": "__golden_non_zero_with_value"}}

# 【不存在】aclnn 通路：canndev/ops-nn 均无 NonZeroWithValue aclnn（普通 NonZero 才有，01 §3.3）。
# 【不存在】e2e 通路：torch_npu 无本算子绑定；aten::nonzero 是不同算子（dispatcher 权威判定，01 §3.3）。
# 【不存在】onnx / caffe 通路：onnx_plugin 仅映射普通 NonZero，caffe 无（01 §3.3）。
#  tf 通路存在（tf_plugin/nonzero_with_value_plugin.cc），其验证走 aclgrphParseTensorFlow
#  预生成 .pb，不在 TTK invoke_path 格式内（01 §3.3）。
