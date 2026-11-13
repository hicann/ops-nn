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

"""GroupNormGrad golden：kernel/GEIR = group_norm_grad（numpy），aclnn = aclnnGroupNormBackward，
e2e = torch.ops.aten.native_group_norm_backward（torch）。真值为 CPU aten.native_group_norm_backward。

布局：dy/x: (N, C, ...)；mean/rstd: (N, group)；gamma: (C,)。
三输出均为 dy dtype；掩码（kernel 的 *_is_require / aclnn/e2e 的 outputMask）
为 False 的输出返回 None。"""

import numpy as np
import torch

try:
    import ml_dtypes  # noqa: F401
except ImportError:  # pragma: no cover
    ml_dtypes = None

__spec__ = {
    "group_norm_grad": "GroupNormGradSpec",
    "aclnnGroupNormBackward": "AclnnGroupNormBackwardSpec",
    "torch.ops.aten.native_group_norm_backward": "AtenGroupNormBackwardSpec",
}


def _np_to_torch(array):
    """numpy → torch CPU 张量（bf16 经 uint16 位视图无损转换）。"""
    array = np.ascontiguousarray(array)
    if ml_dtypes is not None and array.dtype == np.dtype(ml_dtypes.bfloat16):
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _compute_dtype(*tensors):
    """计算精度：任一入参 fp64 则整体 fp64（Promote），否则 fp32。"""
    for tensor in tensors:
        if tensor.dtype == torch.float64:
            return torch.float64
    return torch.float32


def _torch_to_numpy(tensor, dtype_name):
    """结果 → numpy，cast 到输出 dtype。"""
    array = tensor.detach().cpu().numpy()
    return np.ascontiguousarray(array.astype(dtype_name, copy=False))


def _check_mask(mask):
    """输出掩码归一化为 3 个 bool（dx, dgamma, dbeta）。"""
    if mask is None:
        return [True, True, True]
    mask = [bool(v) for v in mask]
    if len(mask) != 3:
        raise ValueError(f"output mask length must be 3, got {len(mask)}")
    return mask


def _nchw(x_tensor):
    """(N, C, ...) → (N, C, HxW)。0元素张量时-1维存在歧义（torch无法从0推断），
    显式按原始后缀维乘积折叠。"""
    n, c = x_tensor.shape[0], x_tensor.shape[1]
    if x_tensor.numel() != 0:
        return x_tensor.reshape(n, c, -1)
    hw = 1
    for dim in x_tensor.shape[2:]:
        hw *= dim
    return x_tensor.reshape(n, c, hw)


def _unify_for_aten(*tensors):
    """混合精度时统一升到最宽 dtype 并保证连续（aten 要求连续输入），
    输出由调用方 cast 回原 dtype。"""
    dtypes = [t.dtype for t in tensors if t is not None]
    if not dtypes:
        return tensors
    widest = dtypes[0]
    for d in dtypes[1:]:
        widest = torch.promote_types(widest, d)
    return tuple(t.to(widest).contiguous() if t is not None else None for t in tensors)


def _aten_group_norm_backward(g, xv, m, r, w, n, c, hw, group, mask, orig_shape):
    """aten 真值；dx 还原原始形状，mask=False 时对应输出为 None。"""
    dx, dgamma, dbeta = torch.ops.aten.native_group_norm_backward(
        g.contiguous(),
        xv.contiguous(),
        m.contiguous(),
        r.contiguous(),
        w.contiguous(),
        n,
        c,
        hw,
        group,
        mask,
    )
    dx = dx.reshape(orig_shape) if dx is not None else None
    return dx, dgamma, dbeta


class GroupNormGradSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    def golden(
        dy,
        mean,
        rstd,
        x,
        gamma,
        *,
        num_groups,
        data_format="NCHW",
        dx_is_require=True,
        dgamma_is_require=True,
        dbeta_is_require=True,
        **kwargs,
    ):
        """参数跟随 group_norm_grad_def.cpp，num_groups 必选。"""
        mask = _check_mask([dx_is_require, dgamma_is_require, dbeta_is_require])

        dy_raw = _np_to_torch(dy)
        mean_raw = _np_to_torch(mean)
        rstd_raw = _np_to_torch(rstd)
        x_raw = _np_to_torch(x)
        gamma_raw = _np_to_torch(gamma)
        compute = _compute_dtype(dy_raw, mean_raw, rstd_raw, x_raw, gamma_raw)
        g = _nchw(dy_raw.to(compute))
        m = mean_raw.to(compute)
        r = rstd_raw.to(compute)
        xv = _nchw(x_raw.to(compute))
        w = gamma_raw.to(compute)
        n, c = xv.shape[0], xv.shape[1]
        hw = xv.shape[2]

        dx, dgamma, dbeta = _aten_group_norm_backward(
            g, xv, m, r, w, n, c, hw, int(num_groups), mask, tuple(dy.shape)
        )

        out_dtypes = [str(d) for d in (kwargs.get("output_dtypes") or ())]
        outputs = []
        for i, result in enumerate((dx, dgamma, dbeta)):
            if not mask[i]:
                outputs.append(None)
            elif i < len(out_dtypes):
                outputs.append(_torch_to_numpy(result, out_dtypes[i]))
            else:
                outputs.append(_torch_to_numpy(result, "float32"))
        return outputs

    class ThirdPartyImpl:
        """torch 三方标杆：num_groups → group 适配，*_is_require → 掩码。"""

        def __init__(
            self,
            *,
            num_groups=None,
            dx_is_require=True,
            dgamma_is_require=True,
            dbeta_is_require=True,
            **kwargs,
        ):
            self.group = int(num_groups) if num_groups is not None else 1
            self.mask = [
                bool(dx_is_require),
                bool(dgamma_is_require),
                bool(dbeta_is_require),
            ]

        def __call__(self, dy, mean, rstd, x, gamma, **kwargs):
            orig_dtype = dy.dtype
            dy, mean, rstd, x, gamma = _unify_for_aten(dy, mean, rstd, x, gamma)
            # 复用_nchw折叠：N=0时原numel//(n*c)除零崩溃、HW=0时被错误置1，
            # reshape(n,c,-1)对0元素张量存在-1维歧义
            xv = _nchw(x)
            g = _nchw(dy)
            n, c = xv.shape[0], xv.shape[1]
            hw = xv.shape[2]
            dx, dgamma, dbeta = torch.ops.aten.native_group_norm_backward(
                g,
                xv,
                mean,
                rstd,
                gamma,
                n,
                c,
                hw,
                self.group,
                self.mask,
            )
            dx = dx.to(orig_dtype).reshape(x.shape) if dx is not None else None
            dgamma = dgamma.to(orig_dtype) if dgamma is not None else None
            dbeta = dbeta.to(orig_dtype) if dbeta is not None else None
            return [dx, dgamma, dbeta]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }


class AclnnGroupNormBackwardSpec:
    """aclnn 流程：torch 入/出。"""

    def golden(
        gradOut,
        input,
        mean,
        rstd,
        gamma,
        N,
        C,
        HxW,
        group,
        outputMask,
        gradInput=None,
        gradGammaOut=None,
        gradBetaOut=None,
        **kwargs,
    ):
        """参数跟随 aclnnGroupNormBackwardGetWorkspaceSize（不含 workspaceSize 与 executor）。"""
        mask = _check_mask(outputMask)

        compute = _compute_dtype(gradOut, input, mean, rstd, gamma)
        g = _nchw(gradOut.to(compute))
        xv = _nchw(input.to(compute))
        m = mean.to(compute)
        r = rstd.to(compute)
        w = gamma.to(compute)

        dx, dgamma, dbeta = _aten_group_norm_backward(
            g,
            xv,
            m,
            r,
            w,
            int(N),
            int(C),
            int(HxW),
            int(group),
            mask,
            tuple(gradOut.shape),
        )

        outputs = []
        for result, out in zip(
            (dx, dgamma, dbeta), (gradInput, gradGammaOut, gradBetaOut)
        ):
            if result is None:
                outputs.append(None)
            elif out is not None:
                outputs.append(result.to(out.dtype))
            else:
                outputs.append(result)
        return outputs

    class ThirdPartyImpl:
        """torch 三方标杆：N/C/HxW/group/outputMask 直传。"""

        def __init__(self, *, N=1, C=1, HxW=1, group=1, outputMask=None, **kwargs):
            self.n, self.c, self.hw = int(N), int(C), int(HxW)
            self.group = int(group)
            self.mask = _check_mask(outputMask)

        def __call__(self, gradOut, input, mean, rstd, gamma, **kwargs):
            orig_dtype = gradOut.dtype
            gradOut, input, mean, rstd, gamma = _unify_for_aten(
                gradOut, input, mean, rstd, gamma
            )
            dx, dgamma, dbeta = torch.ops.aten.native_group_norm_backward(
                gradOut,
                input,
                mean,
                rstd,
                gamma,
                self.n,
                self.c,
                self.hw,
                self.group,
                self.mask,
            )
            return [
                o.to(orig_dtype) if o is not None else None for o in (dx, dgamma, dbeta)
            ]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }


class AtenGroupNormBackwardSpec:
    """e2e 流程：torch 入/出。"""

    def golden(
        grad_out,
        input,
        mean,
        rstd,
        weight,
        N,
        C,
        HxW,
        group,
        output_mask=None,
        **kwargs,
    ):
        """参数跟随 aten schema；grad_weight/grad_bias 同 weight dtype，grad_input 同 grad_out。"""
        mask = _check_mask(output_mask)

        g_raw = grad_out.detach().cpu()
        x_raw = input.detach().cpu()
        m_raw = mean.detach().cpu()
        r_raw = rstd.detach().cpu()
        w_raw = weight.detach().cpu() if weight is not None else None
        compute = _compute_dtype(
            g_raw, x_raw, m_raw, r_raw, *([w_raw] if w_raw is not None else [])
        )
        g = _nchw(g_raw.to(compute))
        xv = _nchw(x_raw.to(compute))
        m = m_raw.to(compute)
        r = r_raw.to(compute)
        if w_raw is not None:
            w = w_raw.to(compute)
            weight_dtype = weight.dtype
        else:
            w = torch.ones([int(C)], dtype=compute)
            weight_dtype = grad_out.dtype

        dx, dgamma, dbeta = _aten_group_norm_backward(
            g,
            xv,
            m,
            r,
            w,
            int(N),
            int(C),
            int(HxW),
            int(group),
            mask,
            tuple(grad_out.shape),
        )

        target_dtypes = [grad_out.dtype, weight_dtype, grad_out.dtype]
        outputs = []
        for result, dtype in zip((dx, dgamma, dbeta), target_dtypes):
            outputs.append(result.to(dtype) if result is not None else None)
        return outputs

    class ThirdPartyImpl:
        """torch 三方标杆：N/C/HxW/group/output_mask 直传。"""

        def __init__(self, *, N=1, C=1, HxW=1, group=1, output_mask=None, **kwargs):
            self.n, self.c, self.hw = int(N), int(C), int(HxW)
            self.group = int(group)
            self.mask = _check_mask(output_mask)

        def __call__(self, *args, **kwargs):
            # 参数按 schema 名匹配，未命中按位置取
            names = ["grad_out", "input", "mean", "rstd", "weight"]
            vals = [kwargs.get(nm) for nm in names]
            idx = 0
            for i, v in enumerate(vals):
                if v is None and idx < len(args):
                    vals[i] = args[idx]
                    idx += 1
            orig_dtype = vals[0].dtype
            unified = _unify_for_aten(*vals)
            dx, dgamma, dbeta = torch.ops.aten.native_group_norm_backward(
                unified[0],
                unified[1],
                unified[2],
                unified[3],
                unified[4],
                self.n,
                self.c,
                self.hw,
                self.group,
                self.mask,
            )
            return [
                o.to(orig_dtype) if o is not None else None for o in (dx, dgamma, dbeta)
            ]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }
