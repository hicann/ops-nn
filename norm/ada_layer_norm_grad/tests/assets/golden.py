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

"""AdaLayerNormGrad 多通路 golden（TestSpec 范式）。

支持通路取自 01_requirement.md §3.3（与 03_spec.yaml pathways 一致）：

| 通路   | 支持 | 注册键                         | 依据 |
|--------|------|--------------------------------|------|
| kernel | ✅   | `ada_layer_norm_grad`（蛇形名） | op_kernel/arch35/ 有 arch35 实现 |
| geir   | ✅   | `ada_layer_norm_grad`（与 kernel 共用同一类） | op_graph/ REG_OP + infershape |
| aclnn  | ✅   | `aclnnAdaLayerNormBackward`    | op_host/op_api/aclnn_ada_layer_norm_backward.{h,cpp} |
| e2e    | ❌   | 无                             | torch_npu 二进制无本算子 aclnn 符号 |

计算核 `_compute` 只写一遍，全程 torch.Tensor（按算法语义拼接，不照抄内核步骤划分）；
kernel/geir 壳收 numpy、aclnn 壳收已 H2D 的 torch.Tensor。

输入顺序（7）：dy / x / rstd / mean / scale / gamma / beta
输出顺序（5）：pd_x / pd_scale / pd_shift / pd_gamma / pd_beta

语义（与 01_requirement.md §2、03_spec.yaml 一致）：
    y     = (x - mean) * rstd
    dy_g  = dy * gamma * (1 + scale)
    temp1 = sum(dy_g,     over H) / N
    temp2 = sum(dy_g * y, over H) / N
    pd_x      = (dy_g - temp1 - y * temp2) * rstd
    pd_scale  = sum(dy * (gamma * y + beta), over S)
    pd_shift  = sum(dy,                      over S)
    pd_gamma  = sum(dy * (1 + scale) * y,    over [B..., S])
    pd_beta   = sum(dy * (1 + scale),        over [B..., S])
"""

__spec__ = {
    # ── kernel + geir：同一个注册键（算子蛇形名），共用一个类 ──
    "ada_layer_norm_grad": "AdaLayerNormGradKernelSpec",
    # ── aclnn：键 = aclnn 接口名 ──
    "aclnnAdaLayerNormBackward": "AdaLayerNormGradAclnnSpec",
}

import numpy as np
import torch

try:
    import ml_dtypes  # 导入后 numpy 才识别 "bfloat16" dtype 名
except ImportError:  # pragma: no cover
    ml_dtypes = None

_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}


def _to_torch(a):
    """numpy -> torch，保留入参 dtype（Promote 语义：fp64 原样保留）。

    TTK 开 Promote（cross_check 判据）时 fp32 入参已被框架抬成 fp64 递进来，
    此处不得再压回 fp32——否则真值塌到与三方腿同精度，|三方-golden| 趋零，
    比值判定的分母被夹穿，出现非确定性假红。bf16 按位 view 成 torch.bfloat16，
    是否升档、升到什么精度统一交给 _compute_dtype 决定。
    """
    if isinstance(a, torch.Tensor):
        return a
    if a.dtype == np.dtype("float16"):
        return torch.from_numpy(a)
    if a.dtype.name == "bfloat16":
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(a)


def _compute_dtype(*tensors):
    """计算精度：任一入参为 fp64 即 Promote 模式（框架把 fp32 升到了 fp64），
    整体用 fp64 算高精度真值；否则整体 fp32（对齐算子内部 fp32 累加）。"""
    for tensor in tensors:
        if tensor.dtype == torch.float64:
            return torch.float64
    return torch.float32


def _to_numpy(tensor, dtype_name):
    """计算结果 -> numpy；dtype_name 为 TTK 下发的输出 dtype（Promote 时已抬高，
    golden 出口不得按算子声明 dtype 砍回，也不得多 cast）。

    bfloat16：class golden 无框架桥接（TTK 只桥接 numpy-API 路径），
    位视图经 uint16 无损转 ml_dtypes.bfloat16 后交付。
    """
    if dtype_name == "bfloat16":
        if ml_dtypes is None:
            raise RuntimeError("bfloat16 output requires the ml_dtypes package")
        bits = tensor.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
        return bits.view(ml_dtypes.bfloat16)
    if dtype_name in _TORCH_DTYPE:
        tensor = tensor.to(_TORCH_DTYPE[dtype_name])
    return tensor.detach().cpu().numpy()


def _compute(dy, x, rstd, mean, scale, gamma, beta):
    """全算子只写这一份计算核：torch.Tensor 进、torch.Tensor 出（精度由调用侧
    的 _compute_dtype 决定：Promote 时 fp64 真值，否则 fp32 对齐算子累加）。"""
    n = x.shape[-1]
    y = (x - mean) * rstd
    # scale 形如 [B..., H] 或 [B..., 1, H]：统一 reshape 成 [B..., 1, H] 与 dy 沿 S 广播
    scale_b = scale.reshape(x.shape[:-2] + (1, x.shape[-1]))
    dy_g = dy * gamma * (1 + scale_b)
    temp1 = dy_g.sum(-1, keepdim=True) / n
    temp2 = (dy_g * y).sum(-1, keepdim=True) / n
    pd_x = (dy_g - temp1 - y * temp2) * rstd

    if x.ndim >= 2:
        pd_scale = (dy * (gamma * y + beta)).sum(-2)
        pd_shift = dy.sum(-2)
        pd_gamma = (dy * (1 + scale_b) * y).sum(tuple(range(x.ndim - 1)))
        pd_beta = (dy * (1 + scale_b)).sum(tuple(range(x.ndim - 1)))
    else:
        pd_scale = dy * (gamma * y + beta)
        pd_shift = dy
        pd_gamma = dy * (1 + scale_b) * y
        pd_beta = dy * (1 + scale_b)

    pd_x = pd_x.reshape(x.shape)
    pd_scale = pd_scale.reshape(scale.shape)
    pd_shift = pd_shift.reshape(scale.shape)
    pd_gamma = pd_gamma.reshape(gamma.shape)
    pd_beta = pd_beta.reshape(beta.shape)
    return [pd_x, pd_scale, pd_shift, pd_gamma, pd_beta]


# 判据：kernel / geir 三方 cross_check L1（NPU 误差 / 竞品误差比值）。
# aclnn/e2e 当前 TTK 不取用三方，tolerance 仅占位、不影响本地 golden 判定。
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


class _Compose:
    """三方竞品：torch 按算法语义拼接。远端 GPU 上执行；属性喂 __init__，张量喂 __call__。

    与 _compute 相互独立地实现同一契约语义（竞品交叉验证），浮点输出 cast 回 NPU 输出 dtype。
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, dy, x, rstd, mean, scale, gamma, beta):
        outs = _compute(dy, x, rstd, mean, scale, gamma, beta)
        # 输出 dtype 对齐契约：pd_x/pd_scale/pd_shift 同 dy，pd_gamma/pd_beta 同 gamma
        o0 = outs[0].to(dy.dtype)
        o1 = outs[1].to(dy.dtype)
        o2 = outs[2].to(dy.dtype)
        o3 = outs[3].to(gamma.dtype)
        o4 = outs[4].to(gamma.dtype)
        return [o0, o1, o2, o3, o4]


class AdaLayerNormGradKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray，返 numpy.ndarray。

    参数名取自 op_host/ada_layer_norm_grad_def.cpp：dy/x/rstd/mean/scale/gamma/beta。
    """

    def golden(*inputs, **kwargs):
        tensors = [_to_torch(a) for a in inputs]
        compute = _compute_dtype(*tensors)
        dy, x, rstd, mean, scale, gamma, beta = [t.to(compute) for t in tensors]
        outs = _compute(dy, x, rstd, mean, scale, gamma, beta)

        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]

        return [
            _to_numpy(o, od[i] if i < len(od) and od[i] else None)
            for i, o in enumerate(outs)
        ]

    third_party = {"torch": _Compose}
    tolerance = _TOL


class AdaLayerNormGradAclnnSpec:
    """aclnn。golden 收【已 H2D 的 torch.Tensor】，直接返 Tensor。

    参数名取自 op_host/op_api/aclnn_ada_layer_norm_backward.h：
        gradOut / input / normalizedShape / rstd / mean / scale / shift /
        weightOptional / biasOptional / gradInputOut / gradScaleOut /
        gradShiftOut / gradWeightOut / gradBiasOut
    对应 kernel 契约的 dy / x / rstd / mean / scale / gamma / beta。
    签名须与 header 位置参数逐字对齐，否则会 TypeError（位置参数个数不符）。
    """

    def golden(
        gradOut,
        input,
        normalizedShape=None,
        rstd=None,
        mean=None,
        scale=None,
        shift=None,
        weightOptional=None,
        biasOptional=None,
        gradInputOut=None,
        gradScaleOut=None,
        gradShiftOut=None,
        gradWeightOut=None,
        gradBiasOut=None,
        **kwargs,
    ):
        # 与 aclnn C++ 一致：weight/bias 缺省时填 1/0（CreateContiguousTensorsAndComputeGrad
        # 的 Fill 语义），dtype 按 infer_dtype 取——优先输出 dtype（pd_gamma/pd_beta，
        # 即 C++ 的 gradWeightOut/gradBiasOut），否则回落入参 dtype。
        # shift 的值不进 kernel（l0op::AdaLayerNormGrad 的 OP_INPUT 仅 7 张量），
        # 仅参与 gradShift 输出的形状/校验，故此处不参与计算。
        # 调用约定守卫：TTK 须按 C header 顺序喂全部位置参数（实测 14 个）。只喂输入侧
        # 会让 normalizedShape 错位收到张量，算错且无日志线索，这里显式拦掉。
        if normalizedShape is not None and not isinstance(
            normalizedShape, (list, tuple)
        ):
            raise TypeError(
                "[aclnnAdaLayerNormBackward] golden expects positional args in C header order "
                "(gradOut, input, normalizedShape, rstd, mean, scale, shift, weight*, bias*, "
                "gradInputOut, gradScaleOut, gradShiftOut, gradWeightOut, gradBiasOut); "
                f"got {type(normalizedShape).__name__} for normalizedShape"
            )
        for _name, _arg in (
            ("rstd", rstd),
            ("mean", mean),
            ("scale", scale),
            ("shift", shift),
        ):
            if _arg is None:
                raise TypeError(
                    f"[aclnnAdaLayerNormBackward] golden requires non-null required input '{_name}'; "
                    "the caller did not pass arguments in C header order"
                )

        # 输出 dtype 兜底：aclnn 通路不传 output_dtypes，只传 tensor_dtypes（长度=全部
        # 张量），输出槽固定为末尾 5 项；序列形态的 dtype 项需先取元素再 str。
        def _dtype_name(d):
            if isinstance(d, (list, tuple)):
                d = d[0] if d else None
            return str(d) if d else None

        od = [_dtype_name(d) for d in (kwargs.get("output_dtypes") or [])]
        if not od:
            td = list(kwargs.get("tensor_dtypes") or [])
            if len(td) < 5:
                # 不足 5 项说明不是全张量 dtype 表，静默取尾会错位，显式报错。
                raise ValueError(
                    "[aclnnAdaLayerNormBackward] golden needs the full tensor dtype list "
                    f"(>=5 entries; outputs are the last 5) but got {len(td)}: {td!r}"
                )
            od = [_dtype_name(d) for d in td[-5:]]

        def _infer_fill_dtype():
            """与 C++ infer_dtype() 同序：weight > bias > gradWeightOut > gradBiasOut > fp32。"""
            for tensor in (weightOptional, biasOptional, gradWeightOut, gradBiasOut):
                if isinstance(tensor, torch.Tensor):
                    return tensor.dtype
            return torch.float32

        fill_dtype = _infer_fill_dtype()
        if weightOptional is None:
            weightOptional = torch.ones(
                input.shape[-1:], dtype=fill_dtype, device=input.device
            )
        if biasOptional is None:
            biasOptional = torch.zeros(
                input.shape[-1:], dtype=fill_dtype, device=input.device
            )
        # 先提升到计算精度再算（Promote 用 fp64，否则 fp32）：直接按 b16 算会在
        # pd_gamma/pd_beta 的抵消处放大舍入误差而必然 FAIL。
        operands = [gradOut, input, rstd, mean, scale, weightOptional, biasOptional]
        compute = _compute_dtype(*operands)
        outs = _compute(*[t.to(compute) for t in operands])
        # 出口 cast 回输出张量自身 dtype（对应 C++ GenOutWithMask 的 Cast）：取
        # mask.dtype 不依赖 tensor_dtypes 的槽位形态，兜底才回落 od。
        masks = (gradInputOut, gradScaleOut, gradShiftOut, gradWeightOut, gradBiasOut)
        cast_outs = []
        for i, (out, mask) in enumerate(zip(outs, masks)):
            if isinstance(mask, torch.Tensor):
                cast_outs.append(out.to(mask.dtype))
            elif i < len(od) and od[i] in _TORCH_DTYPE:
                cast_outs.append(out.to(_TORCH_DTYPE[od[i]]))
            else:
                cast_outs.append(out)
        # 输出张量为 None 时算子不产出，golden 按位返回 None，避免 NO_OUTPUT 误判。
        # 全部输出均为 None 是合法用例状态（如 L2_exc_dtype_007），不得回填或报错。
        return [o if m is not None else None for o, m in zip(cast_outs, masks)]

    # 【预留】TTK aclnn 通路当前不取用 third_party（仅 kernel/geir 取用）；写在此处不生效也无副作用。
    third_party = {"torch": _Compose}
    tolerance = _TOL


# 【不存在】e2e 通路：torch_npu 2.10.0 二进制无本算子 aclnn 符号
#     strings libtorch_npu.so | grep -c aclnnAdaLayerNormBackward = 0；
#     对照同族 aclnnAddLayerNormGrad 有命中，证明非搜漏；torch 绑定属 torch_npu 仓，算子仓只交付 aclnn。
#     （依据 01_requirement.md §3.3 / 03_spec.yaml pathways.torch_single_op=unsupported）
