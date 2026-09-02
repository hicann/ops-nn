#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
TTK custom golden for foreach_addcdiv_list.

Compute formula (docs/aclnnForeachAddcdivList.md:28; regbase.h:13):
    y[t][i] = x1[t][i] + scalars[t] * (x2[t][i] / x3[t][i])   (t = 0..n-1, per list element)

Non-inplace: output y is a separate TensorList of n sub-tensors, same shape/dtype as x1.
Output order == x1 sub-tensor order.

Positional args (TTK passes context.input_arrays unflattened, in CSV input_shapes order):
    x1_list : list of numpy arrays  (DYNAMIC TensorList x1, n sub-tensors)
    x2_list : list of numpy arrays  (DYNAMIC TensorList x2, sync with x1)
    x3_list : list of numpy arrays  (DYNAMIC TensorList x3, sync with x1)
    scalars : numpy array, shape (n,)  (one scalar coefficient per list element, docs:113)

计算实现改用竞品 torch._foreach_* (红线 R3: golden 只能是竞品接口实现或竞品算子拼接实现,
禁 numpy 纯公式), numpy 仅保留 I/O 与 dtype 转换。这里用 _foreach_div + _foreach_mul +
_foreach_add 拼接而不是 _foreach_addcdiv: 后者按 (s*x2)/x3 结合, 与公式的
s*(x2/x3) 结合次序不同(fp32 下实测有 1ULP 级偏差); 拼接式与改造前的 golden 逐位一致。
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
    """bf16 按 |V2 原始字节传入时先 view 回 bf16(torch 不收 bf16), 再升 fp32 计算。"""
    a = np.asarray(a)
    if a.dtype.kind == "V" and _bf16 is not None:
        a = a.view(_bf16)
    # dtype 已匹配时复用原缓冲: 大张量(单份 GB 级)无谓复制会把进程推向 OOM。
    # 下游 torch 算子均非原地，不会改写输入，故复用安全。
    return a.astype(np.float32, copy=False)


_F32_TINY = torch.tensor(float(np.finfo(np.float32).tiny), dtype=torch.float32)


def _ftz(x):
    """把落入 fp32 Subnormal 区间的结果清零，对齐算子两代共同的行为。

    A2(910B) 的 Div 没有 config 参数(asc-devkit Div.md 里带 config 的原型对 Atlas A2 标注
    "不支持")，只有单指令一条路，Subnormal 必然 FTZ；arch35 的 Vec::Div 用默认
    DivConfig{DivAlgo::INTRINSIC}，文档写明该档"Subnormal 均被 FTZ"。CPU 默认保留
    Subnormal，故显式补这一步。"""
    return torch.where((x != 0) & (x.abs() < _F32_TINY), torch.zeros_like(x), x)


def __golden_foreach_addcdiv_list(x1_list, x2_list, x3_list, scalars, **kwargs):
    output_dtypes = _per_tensor_dtypes(kwargs.get("output_dtypes"))

    # scalars is a length-n vector: scalars[t] applies to list element t.
    scalars_arr = np.asarray(scalars).astype(np.float32).reshape(-1)

    results = []
    for i, (a, b, c) in enumerate(zip(x1_list, x2_list, x3_list)):
        ta = torch.from_numpy(_to_fp32(a))
        tb = torch.from_numpy(_to_fp32(b))
        tc = torch.from_numpy(_to_fp32(c))
        s = scalars_arr[i] if i < scalars_arr.size else scalars_arr[-1]
        ts = torch.from_numpy(np.asarray([s], dtype=np.float32))[0]

        # y_i = x1_i + scalars[i] * (x2_i / x3_i)
        quot = [_ftz(q) for q in torch._foreach_div([tb], [tc])]
        out = torch._foreach_add([ta], torch._foreach_mul(quot, ts))[0].numpy()

        if output_dtypes is not None and i < len(output_dtypes):
            od = output_dtypes[i]
            target = od[0] if isinstance(od, (tuple, list)) else str(od)
        else:
            target = str(np.asarray(a).dtype)
        if target == "bfloat16":
            out = out.astype(_bf16) if _bf16 is not None else out
        else:
            out = out.astype(target)
        results.append(out)
    return results


__golden__ = {"kernel": {"foreach_addcdiv_list": "__golden_foreach_addcdiv_list"}}

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


_GOLDEN_FN = __golden_foreach_addcdiv_list


class _ForeachAddcdivListCompose:
    def __call__(self, x1, x2, x3, scalars, **kwargs):
        st = (
            scalars
            if isinstance(scalars, torch.Tensor)
            else torch.as_tensor(np.asarray(scalars))
        )
        sc = [float(v) for v in st.reshape(-1)]
        return torch._foreach_addcdiv(_tp_list(x1), _tp_list(x2), _tp_list(x3), sc)


class ForeachAddcdivListKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _ForeachAddcdivListCompose}
    tolerance = _TOL_KERNEL


__spec__ = {
    "foreach_addcdiv_list": "ForeachAddcdivListKernelSpec",
    "aclnnForeachAddcdivList": "ForeachAddcdivListAclnnSpec",
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


class ForeachAddcdivListAclnnSpec:
    """aclnn 通路 spec。golden 收设备侧 torch.Tensor(README: ACLNN 传入已 H2D 的
    torch.Tensor), 由 TTK 按 aclnn 头文件形参**位置**下发(AclnnParamPlan.build_args),
    故签名逐项对齐 aclnnForeachAddcdivListGetWorkspaceSize 的形参;
    third_party 走按名绑定(pool 的 key 取自头文件形参名), 复用 kernel 通路的竞品类
    ——其形参名即 def 注册名, 与头文件一致。"""

    @staticmethod
    def golden(x1, x2, x3, scalars, out=None, **kwargs):
        a = [_tp_one(t) for t in x1]
        b = [_tp_one(t) for t in x2]
        c = [_tp_one(t) for t in x3]
        sc = _tp_scalars(scalars)
        # 两步拼接: 先 div 再按标量乘, 最后相加, 不用 addcdiv 的融合舍入
        return _keep_dtype(
            torch._foreach_add(a, torch._foreach_mul(torch._foreach_div(b, c), sc)), x1
        )

    third_party = {"torch": _ForeachAddcdivListCompose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# e2e / TensorFlow / ONNX / 融合 pass: 均未交付——算子目录下无 framework/ 插件、
# 无 graph pass, 也未发现 torch_npu eager/aten 绑定到该 aclnn 接口。
