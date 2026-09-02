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

"""TTK kernel 模式自定义 golden：foreach_addcmul_list。

编写依据：docs/aclnnForeachAddcmulList.md「功能说明 / 计算公式」节：
    y_i = x1_i + scalars * x2_i * x3_i   (i = 0 .. n-1)

⚠️ 结合序取 (x2*x3) 先乘、再乘标量。公式里三个因子的乘积在数学上与结合序无关，但在
fp32 中间量下不等价：scalars 取极值(如 3.35e38)时 (x2*scalars) 这一步就会冲破 fp32 上限
变成 inf，而 x2*x3 通常是小量、乘上 scalars 后并不溢出。实测某 bf16 用例 8192 个元素中有
7361 个的 |x2*scalars| 超限，按该结合序算出的是一片 inf，与 fp64 参照给出的有限真值相差
甚远。golden 的职责是给出最接近真值的参照，故取不引入伪溢出的那一种。整型分支同理。

输入顺序（与 CSV input_shapes 一致）：x1, x2, x3, scalars
输出顺序（与 CSV output_dtypes 一致）：y

说明：shape_mapping 将每个张量列表映射为单张量（列表长度 1），scalars 为
shape [1] 的单元素张量，对应 totalTensorCount_ == 1。golden 直接对单张量计算。

计算实现改用竞品 torch._foreach_*（红线 R3：golden 只能是竞品接口实现或竞品算子
拼接实现，禁 numpy 纯公式），numpy 仅保留 I/O 与 dtype 转换；浮点在 fp32 中间量、
整型在 int64 中间量按原结合次序计算，与算子的计算精度一致。
"""

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


def _per_tensor_dtypes(od):
    """把 output_dtypes 拍平成 per-tensor 列表。

    TTK 归一化后，TensorList 输出的 output_dtypes 是"按输出分组"的嵌套形式
    (( 'float32', ... ) ,)；单 tensor 输出则是扁平的。老用例集写成扁平 N 项，
    当前 TTK 会以 CASE_FIELD_AMBIGUOUS 拒收。两种形式都要能收，否则
    output_dtypes[i] 取到的是元组，np.dtype(tuple) 直接抛 GOLDEN_FAILURE。
    """
    if od is None:
        return None
    flat = []
    for e in od:
        if isinstance(e, (tuple, list)):
            flat.extend(e)
        else:
            flat.append(e)
    return flat


def _to_fp32(a):
    """bf16 按 |V2 原始字节传入时先 view 回 bf16(torch 不收 bf16), 再升 fp32 计算。

    用 fp32 而不是 fp64：算子在 fp32 上算(arch35 DAG 的计算类型 U = float)，golden 抬到
    fp64 会得出算子不会产出的中间量。"""
    a = np.asarray(a)
    if a.dtype.kind == "V" and _bf16 is not None:
        a = a.view(_bf16)
    # dtype 已匹配时复用原缓冲: 大张量(单份 GB 级)无谓复制会把进程推向 OOM。
    # 下游 torch 算子均非原地，不会改写输入，故复用安全。
    return a.astype(np.float32, copy=False)


def __golden_foreach_addcmul_list(x1_list, x2_list, x3_list, scalars, **kwargs):
    output_dtypes = _per_tensor_dtypes(kwargs.get("output_dtypes"))
    # 保留 scalars 原始 dtype：整数分支要用精确的整数标量，先过一道 float32 会把
    # 超过 2^24 的 int32 抹掉低位（1564714939 -> 1564714880），使整个整数结果偏掉。
    scalars_raw = np.asarray(scalars).reshape(-1)
    scalars_arr = scalars_raw.astype(np.float32)
    results = []
    for i, (a, b, c) in enumerate(zip(x1_list, x2_list, x3_list)):
        if output_dtypes is not None and i < len(output_dtypes):
            od = output_dtypes[i]
            target = od[0] if isinstance(od, (tuple, list)) else str(od)
        else:
            target = str(np.asarray(a).dtype)
        s = scalars_arr[i] if i < scalars_arr.size else scalars_arr[-1]
        if target != "bfloat16" and np.issubdtype(np.dtype(target), np.integer):
            dt = np.dtype(target)
            narrow = torch.from_numpy(np.empty(0, dtype=dt)).dtype
            a_i = torch.from_numpy(np.asarray(a).astype(dt)).to(torch.int64)
            b_i = torch.from_numpy(np.asarray(b).astype(dt)).to(torch.int64)
            c_i = torch.from_numpy(np.asarray(c).astype(dt)).to(torch.int64)
            s_raw = scalars_raw[i] if i < scalars_raw.size else scalars_raw[-1]
            s_i = int(np.asarray(s_raw).astype(dt))
            prod = torch._foreach_mul(torch._foreach_mul([b_i], [c_i]), s_i)
            y = torch._foreach_add([a_i], prod)[0].to(narrow).numpy()
        else:
            # 浮点路径在 fp32 中间量算,对齐算子的计算精度,再 cast 回目标 dtype。
            ta = torch.from_numpy(_to_fp32(a))
            tb = torch.from_numpy(_to_fp32(b))
            tc = torch.from_numpy(_to_fp32(c))
            ts = torch.from_numpy(np.asarray([s], dtype=np.float32))[0]
            prod = torch._foreach_mul(torch._foreach_mul([tb], [tc]), ts)
            out = torch._foreach_add([ta], prod)[0].numpy()
            if target == "bfloat16":
                y = out.astype(_bf16) if _bf16 is not None else out.astype(np.float32)
            else:
                y = out.astype(target)
        results.append(y)
    return results


__golden__ = {"kernel": {"foreach_addcmul_list": "__golden_foreach_addcmul_list"}}

# ----------------------------------------------------------------------------
# TTK 新版 spec 注册（kernel 通路）: 在保留原 golden 的基础上补三方标杆能力。
# third_party 直接对标 torch 的 _foreach_* 竞品 API，在设备侧跑，供 cross_check 比对。
# ----------------------------------------------------------------------------
_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _tp_list(xs):
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备。"""
    out = []
    for a in xs:
        t = a if isinstance(a, torch.Tensor) else torch.as_tensor(_to_fp32(a))
        out.append(t.to(torch.float32))
    return out


def _tp_scalar(x):
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x, "float32"))
    return float(t.reshape(-1)[0])


_GOLDEN_FN = __golden_foreach_addcmul_list


class _ForeachAddcmulListCompose:
    def __call__(self, x1, x2, x3, scalars, **kwargs):
        st = (
            scalars
            if isinstance(scalars, torch.Tensor)
            else torch.as_tensor(np.asarray(scalars))
        )
        sc = [float(v) for v in st.reshape(-1)]
        return torch._foreach_addcmul(_tp_list(x1), _tp_list(x2), _tp_list(x3), sc)


class ForeachAddcmulListKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _ForeachAddcmulListCompose}
    tolerance = _TOL_KERNEL


__spec__ = {
    "foreach_addcmul_list": "ForeachAddcmulListKernelSpec",
    "aclnnForeachAddcmulList": "ForeachAddcmulListAclnnSpec",
}


def _tp_one(t):
    """aclnn 通路: 框架传入的已是设备侧 torch.Tensor, 不经 numpy。"""
    t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
    return t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t


def _tp_scalars(v):
    if isinstance(v, torch.Tensor):
        return [float(x) for x in v.reshape(-1)]
    return [float(x) for x in v]


def _keep_dtype(res, ref):
    """golden 输出 dtype 必须与算子输出一致: 比对按 dtype 判定, fp16/bf16 提到 fp32
    算完必须还原, 否则 binary_equal 直接判 "dtype 不可比"(实测 GOLD 0%)。
    golden_mode=Promote 时入参本身已是 fp32, 此处是恒等操作。"""
    refs = ref if isinstance(ref, (list, tuple)) else [ref] * len(res)
    return [
        t.to(r.dtype)
        if isinstance(t, torch.Tensor) and isinstance(r, torch.Tensor)
        else t
        for t, r in zip(res, refs)
    ]


class ForeachAddcmulListAclnnSpec:
    """aclnn 通路 spec。golden 收设备侧 torch.Tensor(README: ACLNN 传入已 H2D 的
    torch.Tensor), 由 TTK 按 aclnn 头文件形参**位置**下发(AclnnParamPlan.build_args),
    故签名逐项对齐 aclnnForeachAddcmulListGetWorkspaceSize 的形参;
    third_party 走按名绑定(pool 的 key 取自头文件形参名), 复用 kernel 通路的竞品类
    ——其形参名即 def 注册名, 与头文件一致。"""

    @staticmethod
    def golden(x1, x2, x3, scalars, out=None, **kwargs):
        a = [_tp_one(t) for t in x1]
        b = [_tp_one(t) for t in x2]
        c = [_tp_one(t) for t in x3]
        sc = _tp_scalars(scalars)
        return _keep_dtype(
            torch._foreach_add(a, torch._foreach_mul(torch._foreach_mul(b, c), sc)), x1
        )

    third_party = {"torch": _ForeachAddcmulListCompose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# e2e / TensorFlow / ONNX / 融合 pass: 均未交付——算子目录下无 framework/ 插件、
# 无 graph pass, 也未发现 torch_npu eager/aten 绑定到该 aclnn 接口。
