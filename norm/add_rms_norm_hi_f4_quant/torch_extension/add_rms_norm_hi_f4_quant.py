# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library

HIF4_BLOCK_SIZE = 64
FP4_IN_BYTE = 2
SUPPORTED_SCALE_ALGS = (0, 1)


def _ceil_div(value, factor):
    return (value + factor - 1) // factor


def _check_scale_alg(scale_alg):
    if scale_alg not in SUPPORTED_SCALE_ALGS:
        raise RuntimeError(
            f"scale_alg should be one of {SUPPORTED_SCALE_ALGS} "
            f"(0=mse_first, 1=unbiased), but got {scale_alg}"
        )


def _y_shape(x):
    shape = list(x.shape)
    shape[-1] = _ceil_div(x.size(x.dim() - 1), FP4_IN_BYTE)
    return shape


def _scale_shape(x):
    shape = list(x.shape)
    shape[-1] = _ceil_div(x.size(x.dim() - 1), HIF4_BLOCK_SIZE)
    shape.append(1)
    return shape


def _rstd_shape(x):
    shape = list(x.shape)
    shape[-1] = 1
    return shape


class AddRmsNormHiF4QuantOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("add_rms_norm_hi_f4_quant")

    def sources(self):
        return [self.resolve_source("add_rms_norm_hi_f4_quant.cpp")]

    def schema(self):
        return (
            "add_rms_norm_hi_f4_quant(Tensor x1, Tensor x2, Tensor gamma, *, Tensor? beta_optional=None, "
            "float epsilon=1e-6, bool output_rstd=False, int scale_alg=0) -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def add_rms_norm_hi_f4_quant_meta(
            x1,
            x2,
            gamma,
            *,
            beta_optional=None,
            epsilon=1e-6,
            output_rstd=False,
            scale_alg=0,
        ):
            _check_scale_alg(scale_alg)
            y = x1.new_empty(_y_shape(x1), dtype=torch.uint8)
            x = x1.new_empty(x1.shape, dtype=x1.dtype)
            scale = x1.new_empty(_scale_shape(x1), dtype=torch.float32)
            rstd = x1.new_empty(_rstd_shape(x1), dtype=torch.float32)
            return y, x, scale, rstd


builder = AddRmsNormHiF4QuantOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def add_rms_norm_hi_f4_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    *,
    beta_optional: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    output_rstd: bool = False,
    scale_alg: int = 0,
):
    r"""Add 与 RMSNorm 融合 HiF4（HiFloat4）动态量化。

    对输入 x1 与 x2 逐元素相加得到 s（s = x1 + x2），对 s 在尾轴做 RMSNorm 归一化
    （y = s * rstd * gamma + beta），随后按 64 个元素为一组（Block）做 HiF4 三层分层缩放
    量化（Block=64 + Sub-block=8 + Micro-block=4），输出 S1P2 编码的 4-bit 量化结果、求和结果
    及三层缩放因子。

    Args:
        x1 (torch.Tensor): 输入张量，shape 任意（尾轴为量化轴），dtype float16/bfloat16，
            尾轴维度需为 64 的倍数，NPU 张量。
        x2 (torch.Tensor): 输入张量，shape 与 dtype 均需与 x1 一致，NPU 张量。
        gamma (torch.Tensor): RMSNorm 缩放参数，1-D，shape 与 x1 尾轴一致，dtype 与 x1 一致
            或 float32。
        beta_optional (torch.Tensor | None, optional): RMSNorm 偏置参数，1-D，shape 与
            gamma 一致，dtype 与 x1 一致或 float32。默认 None。
        epsilon (float, optional): 防止除零的小常数。默认 1e-6。
        output_rstd (bool, optional): 是否计算并输出 rstd 的真实值（RMS 倒数）。默认 False；
            False 时 rstdOut 返回全零占位 Tensor。
        scale_alg (int, optional): HiF4 Eb/Ec 门限策略。0=mse_first，1=unbiased。默认 0。

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): 四元组。
        - yOut: HiF4 量化输出，uint8（4-bit 打包，尾轴为 ceil(尾轴/2)），shape 为
          x1.shape[:-1] + [ceil(x1 尾轴 / 2)]。
        - xOut: x1 与 x2 的逐元素和，dtype 与 x1 一致，shape 与 x1 一致。
        - scaleOut: 三层缩放因子打包输出，float32，shape 为
          x1.shape[:-1] + [ceil(x1 尾轴 / 64), 1]。
        - rstdOut: RMS 倒数，float32，shape 为 x1.shape[:-1] + [1]，恒为有效输出（Tensor）；
          当 output_rstd 为 False 时算子不计算 rstd，该输出保持全零。
    """
    _check_scale_alg(scale_alg)
    op_module = builder.load()
    y, x, scale, rstd = op_module.add_rms_norm_hi_f4_quant(
        x1, x2, gamma, beta_optional, epsilon, output_rstd, scale_alg
    )
    return y, x, scale, rstd
