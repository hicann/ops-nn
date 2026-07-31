# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""swiglu_group_grad TestSpec — kernel (numpy) + aclnn (torch) golden.

Ported from tbetoolkits golden.

Outputs (order matches def.cpp / aclnn*.h):
  grad_x      (always)   — gradient w.r.t. x,       shape (*outer, 2H), dtype = grad_y.dtype
  grad_weight (optional) — gradient w.r.t. weight,  shape (*outer, 1),  dtype = float32
                           returned only when weight input is provided.

Attribute:
  clamp_limit (float, default=0.0) — clamp threshold; 0 means no clamp.

Core formula (ClampedSwiGLU backward):
  x split into g = x[..., :H], u = x[..., H:]
  g_tilde = min(g, c);  u_tilde = clip(u, -c, c)   [when clamp_limit > 0]
  f = silu(g_tilde);  silu' = sigmoid'(g_tilde)
  dg = grad_y * silu' * u_tilde * w * m_g * m_r
  du = grad_y * f * w * m_u * m_r
  grad_x = concat([dg, du], axis=-1)
  grad_weight = sum(grad_y * y_origin, axis=-1, keepdims)  [if weight given]
"""

import numpy
import torch

__spec__ = {
    "swiglu_group_grad": "SwigluGroupGradTestSpec",
    "aclnnSwigluGroupGrad": "AclnnSwigluGroupGradTestSpec",
}


def _stable_sigmoid(x):
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        y = numpy.empty_like(x, dtype=numpy.float32)
        pos = x >= 0
        y[pos] = 1.0 / (1.0 + numpy.exp(-x[pos]))
        exp_x = numpy.exp(x[~pos])
        y[~pos] = exp_x / (1.0 + exp_x)
    return y


def _silu_and_grad(g_tilde):
    s = _stable_sigmoid(g_tilde)
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        f = g_tilde * s
        f = numpy.where(numpy.isneginf(g_tilde), 0.0, f)
        silu_prime = s + f - f * s
        silu_prime = numpy.where(numpy.isposinf(g_tilde), 1.0, silu_prime)
        silu_prime = numpy.where(numpy.isneginf(g_tilde), 0.0, silu_prime)
    return f.astype(numpy.float32, copy=False), silu_prime.astype(
        numpy.float32, copy=False
    )


def _get_row_mask(outer_shape, group_index):
    total_rows = int(numpy.prod(outer_shape))
    if group_index is None:
        return numpy.ones(outer_shape + (1,), dtype=numpy.float32)
    trunc = int(numpy.sum(group_index))
    valid_rows = min(max(trunc, 0), total_rows)
    m_r_flat = numpy.zeros(total_rows, dtype=numpy.float32)
    if valid_rows > 0:
        m_r_flat[:valid_rows] = 1.0
    return m_r_flat.reshape(outer_shape + (1,))


def _compute_clamped_swiglu_grad(
    grad_output, x, weight, y_origin, group_index, clamp_limit
):
    """Core numpy computation. Returns (grad_x, grad_weight or None)."""
    if x.shape[-1] % 2 != 0:
        raise Exception(
            "ClampedSwigluGrad x last dim must be even, but got shape: {}".format(
                x.shape
            )
        )

    H = x.shape[-1] // 2

    if grad_output.shape[-1] != H:
        raise Exception(
            "ClampedSwigluGrad grad_output last dim must be half of x last dim. "
            "grad_output shape={}, x shape={}".format(grad_output.shape, x.shape)
        )

    if grad_output.shape[:-1] != x.shape[:-1]:
        raise Exception(
            "ClampedSwigluGrad grad_output outer shape must be same as x outer shape. "
            "grad_output shape={}, x shape={}".format(grad_output.shape, x.shape)
        )

    x_f = x.astype(numpy.float32)
    dy_f = grad_output.astype(numpy.float32)

    g = x_f[..., :H]
    u = x_f[..., H:]

    c = float(clamp_limit)
    has_clamp = c > 0.0
    if has_clamp:
        g_tilde = numpy.minimum(g, c)
        u_tilde = numpy.clip(u, -c, c)
        m_g = (g < c).astype(numpy.float32)
        m_u = ((u > -c) & (u < c)).astype(numpy.float32)
    else:
        g_tilde = g
        u_tilde = u
        m_g = numpy.ones_like(g, dtype=numpy.float32)
        m_u = numpy.ones_like(u, dtype=numpy.float32)

    outer_shape = g.shape[:-1]
    m_r = _get_row_mask(outer_shape, group_index)

    if weight is not None:
        if weight.shape[:-1] != outer_shape or weight.shape[-1] != 1:
            raise Exception(
                "ClampedSwigluGrad weight shape must be outer_shape + (1,). "
                "weight shape={}, outer_shape={}".format(weight.shape, outer_shape)
            )
        w = weight.astype(numpy.float32)
    else:
        w = numpy.ones(outer_shape + (1,), dtype=numpy.float32)

    f, silu_prime = _silu_and_grad(g_tilde)

    dg = dy_f * silu_prime * u_tilde * w * m_g * m_r
    du = dy_f * f * w * m_u * m_r
    grad_x = numpy.concatenate([dg, du], axis=-1)

    grad_weight = None
    if weight is not None:
        if y_origin is not None:
            if y_origin.shape != grad_output.shape:
                raise Exception(
                    "ClampedSwigluGrad y_origin shape must be same as grad_output shape. "
                    "y_origin shape={}, grad_output shape={}".format(
                        y_origin.shape, grad_output.shape
                    )
                )
            y_origin_f = y_origin.astype(numpy.float32)
        else:
            y_origin_f = f * u_tilde
        grad_weight = numpy.sum(dy_f * y_origin_f, axis=-1, keepdims=True).astype(
            numpy.float32
        )

    return grad_x, grad_weight


class SwigluGroupGradTestSpec:
    """swiglu_group_grad kernel golden (numpy in; params match def.cpp).
    def.cpp inputs : grad_y, x, weight(optional), y_origin(optional), group_index(optional)
    def.cpp attr   : clamp_limit (float, default=0)
    def.cpp outputs: grad_x, grad_weight(optional)
    """

    def golden(
        grad_y,
        x,
        weight=None,
        y_origin=None,
        group_index=None,
        clamp_limit=0.0,
        **kwargs,
    ):
        grad_output = grad_y

        if x.size == 0:
            grad_x = numpy.zeros(x.shape, dtype=grad_output.dtype)
            if weight is not None:
                grad_weight = numpy.zeros(weight.shape, dtype=numpy.float32)
                return [grad_x, grad_weight]
            return [grad_x]

        grad_x, grad_weight = _compute_clamped_swiglu_grad(
            grad_output, x, weight, y_origin, group_index, clamp_limit
        )
        grad_x = grad_x.astype(grad_output.dtype, copy=False)

        if weight is not None:
            # 保留原始 golden 的重复计算逻辑:丢弃 _compute 算出的 grad_weight,重新计算
            if y_origin is not None:
                grad_weight = numpy.sum(
                    grad_output.astype(numpy.float32) * y_origin.astype(numpy.float32),
                    axis=-1,
                    keepdims=True,
                ).astype(numpy.float32)
            else:
                # def.cpp 约束 weight/y_origin 必须同时提供,此分支不可达
                outer_shape = grad_output.shape[:-1]
                grad_weight = numpy.zeros(outer_shape + (1,), dtype=numpy.float32)
            return [grad_x, grad_weight]
        return [grad_x]


class AclnnSwigluGroupGradTestSpec:
    """aclnnSwigluGroupGrad aclnn golden (torch.Tensor in; params match aclnn*.h).
    header tensors : gradY, x, weightOptional, yOriginOptional, groupIndexOptional
    header scalar  : clampLimit (float)
    header outputs : gradXOut, gradWeightOutOptional
    """

    def golden(
        gradY,
        x,
        weightOptional=None,
        yOriginOptional=None,
        groupIndexOptional=None,
        clampLimit=0.0,
        gradXOut=None,
        gradWeightOutOptional=None,
        **kwargs,
    ):
        def _to_numpy(t):
            if t is None:
                return None
            if hasattr(t, "numpy"):
                arr = t.detach().cpu().numpy() if t.requires_grad else t.numpy()
                if t.dtype == torch.bfloat16:
                    return arr.astype(numpy.float32)
                return arr
            return t

        grad_output_np = _to_numpy(gradY)
        x_np = _to_numpy(x)
        weight_np = _to_numpy(weightOptional)
        y_origin_np = _to_numpy(yOriginOptional)
        group_index_np = _to_numpy(groupIndexOptional)

        orig_dtype = gradY.dtype

        if x_np.size == 0:
            grad_x_np = numpy.zeros(x_np.shape, dtype=grad_output_np.dtype)
            results = [grad_x_np]
            if weight_np is not None:
                grad_weight_np = numpy.zeros(weight_np.shape, dtype=numpy.float32)
                results.append(grad_weight_np)
        else:
            grad_x_np, _ = _compute_clamped_swiglu_grad(
                grad_output_np,
                x_np,
                weight_np,
                y_origin_np,
                group_index_np,
                float(clampLimit),
            )
            results = [grad_x_np]
            if weight_np is not None:
                # 保留原始 golden 的重复计算逻辑:丢弃 _compute 算出的 grad_weight,重新计算
                if y_origin_np is not None:
                    grad_weight_np = numpy.sum(
                        grad_output_np.astype(numpy.float32)
                        * y_origin_np.astype(numpy.float32),
                        axis=-1,
                        keepdims=True,
                    ).astype(numpy.float32)
                else:
                    outer_shape = grad_output_np.shape[:-1]
                    grad_weight_np = numpy.zeros(
                        outer_shape + (1,), dtype=numpy.float32
                    )
                results.append(grad_weight_np)

        torch_results = [torch.from_numpy(r) for r in results]
        torch_results[0] = torch_results[0].to(orig_dtype)
        return torch_results
