# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math
from typing import Optional, Tuple

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library

# SwiGLU激活gate/up拆分因子：activate_dim维为2H，激活输出该维为H
SPLIT_FACTOR = 2
# HIFLOAT4为4-bit浮点，每字节存储2个元素，y(uint8)最后一维为元素数的一半
HI_F4_PER_BYTE = 2
# scale量化块大小：每64个元素共享一个scale
QUANT_BLOCK_SIZE = 64
# scale输出尾部对齐维长度
SCALE_TAIL_DIM = 1

# 支持的 scale_alg 取值
SUPPORTED_SCALE_ALGS = (0, 1)


def _check_scale_alg(scale_alg):
    if scale_alg not in SUPPORTED_SCALE_ALGS:
        raise RuntimeError(
            f"scale_alg should be one of {SUPPORTED_SCALE_ALGS}, but got {scale_alg}"
        )


def _normalize_dim(dim, rank):
    return dim if dim >= 0 else dim + rank


def _quant_output_shape(x, activate_dim):
    """输出y的shape：activate_dim维减半，最后一维再减半（HIFLOAT4每字节2个元素）"""
    rank = x.dim()
    select_dim = _normalize_dim(activate_dim, rank)
    shape = []
    for i in range(rank):
        if i == select_dim:
            shape.append(x.size(i) // SPLIT_FACTOR)
        else:
            shape.append(x.size(i))
    shape[-1] = shape[-1] // HI_F4_PER_BYTE
    return shape


def _scale_shape(x, group_index, activate_dim, axis):
    """输出scale的shape：activate_dim维减半，量化维替换为scale组数，末尾追加长度1的对齐维"""
    rank = x.dim()
    select_dim = _normalize_dim(activate_dim, rank)
    quant_dim = _normalize_dim(axis, rank)
    shape = []
    for i in range(rank):
        if i == select_dim:
            shape.append(x.size(i) // SPLIT_FACTOR)
        else:
            shape.append(x.size(i))

    # 量化维scale组数：
    # - 未传入group_index，或量化维为最后一维：ceil(quant_dim_size / 64)
    # - 传入group_index且量化维非最后一维：floor(quant_dim_size / 64) + group_index[0]（组边界补偿）
    if group_index is None or quant_dim == rank - 1:
        quant_size = math.ceil(shape[quant_dim] / QUANT_BLOCK_SIZE)
    else:
        quant_size = (
            math.floor(shape[quant_dim] / QUANT_BLOCK_SIZE) + group_index.shape[0]
        )
    shape[quant_dim] = quant_size
    shape.append(SCALE_TAIL_DIM)
    return shape


class SwigluHiF4QuantOpBuilder(OpBuilder):
    """
    SwigluHiF4Quant 算子的构建器

    基于 aclnnSwigluHiF4Quant API 实现，融合 SwiGLU 激活与 HIFLOAT4 分组量化。
    输入为 FP16/BF16，输出为 HIFLOAT4（按 fp4x2 打包为 uint8）+ float32 scale。
    """

    def __init__(self):
        super().__init__("swiglu_hi_f4_quant")

    def sources(self):
        return [self.resolve_source("swiglu_hi_f4_quant.cpp")]

    def schema(self):
        return (
            "swiglu_hi_f4_quant("
            "Tensor x, "
            "*, "
            "Tensor? group_index=None, "
            "int activate_dim=-1, "
            "bool activate_left=False, "
            "int swiglu_mode=0, "
            "float clamp_limit=7.0, "
            "float glu_alpha=1.702, "
            "float glu_bias=1.0, "
            "int axis=-1, "
            "int scale_alg=0"
            ") -> (Tensor y, Tensor scale)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def swiglu_hi_f4_quant_meta(
            x: torch.Tensor,
            *,
            group_index: Optional[torch.Tensor] = None,
            activate_dim: int = -1,
            activate_left: bool = False,
            swiglu_mode: int = 0,
            clamp_limit: float = 7.0,
            glu_alpha: float = 1.702,
            glu_bias: float = 1.0,
            axis: int = -1,
            scale_alg: int = 0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            _check_scale_alg(scale_alg)
            torch._check(
                x.dim() > 1,
                lambda: f"x dim should be greater than 1, but got {x.dim()}",
            )
            select_dim = _normalize_dim(activate_dim, x.dim())
            quant_dim = _normalize_dim(axis, x.dim())
            torch._check(
                0 <= select_dim <= x.dim() - 1,
                lambda: (
                    f"activate_dim should be in range [0, x.dim()-1], but got {activate_dim}"
                ),
            )
            torch._check(
                0 <= quant_dim <= x.dim() - 1,
                lambda: f"axis should be in range [0, x.dim()-1], but got {axis}",
            )

            # y：HIFLOAT4按fp4x2打包为uint8（每字节2个fp4元素）
            y = x.new_empty(_quant_output_shape(x, activate_dim), dtype=torch.uint8)
            scale = x.new_empty(
                _scale_shape(x, group_index, activate_dim, axis), dtype=torch.float32
            )
            return y, scale


builder = SwigluHiF4QuantOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def swiglu_hi_f4_quant(
    x: torch.Tensor,
    *,
    group_index: Optional[torch.Tensor] = None,
    activate_dim: int = -1,
    activate_left: bool = False,
    swiglu_mode: int = 0,
    clamp_limit: float = 7.0,
    glu_alpha: float = 1.702,
    glu_bias: float = 1.0,
    axis: int = -1,
    scale_alg: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NPU 上的 SwigluHiF4Quant —— 融合 SwiGLU 激活与 HIFLOAT4 分组量化

    对输入 x 先做 SwiGLU 激活（gate/up 拆分与门控），再按 64 元素一组做
    HIFLOAT4 量化，返回量化结果 y 与 float32 格式的 scale 张量。

    Args:
        x (torch.Tensor): 必选，输入Tensor。数据类型为 float16/bfloat16，
            shape 为 (..., 2H, ...)，维度大于1，activate_dim 维大小应为偶数（2H）；
            必须是NPU上的Tensor。
        group_index (torch.Tensor, optional): 组索引Tensor，用于非最后一维量化时
            补偿不完整组的scale数量，shape 为 (group_num,)。默认None。
        activate_dim (int, optional): SwiGLU门控拼接所在维，负值按倒数第N维解释，
            取值范围为 [-x.dim(), x.dim()-1]。默认-1（最后一维）。
        activate_left (bool, optional): 为True时gate取x的前半部分、up取后半部分；
            为False时gate取后半部分、up取前半部分。默认False。
        swiglu_mode (int, optional): SwiGLU激活模式，仅支持{0, 1}。默认0。
        clamp_limit (float, optional): 激活输出的钳位上限，必须为正的有限值。默认7.0。
        glu_alpha (float, optional): 激活函数的alpha参数，必须为有限值。默认1.702。
        glu_bias (float, optional): 激活函数的bias参数，必须为有限值。默认1.0。
        axis (int, optional): 量化轴，负值按倒数第N维解释，
            取值范围为 [-x.dim(), x.dim()-1]。默认-1（最后一维）。
        scale_alg (int, optional): scale量化算法选择，0或1代表两种不同的量化算法，
            仅支持{0, 1}。默认0。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - y (torch.Tensor): HIFLOAT4量化结果，按fp4x2打包为uint8（每字节2个
              fp4元素）。shape 为 x 的 shape 中 activate_dim 维减半、最后一维再减半。
            - scale (torch.Tensor): 量化scale，dtype 为 float32，
              shape 为 x 的 shape 中 activate_dim 维减半、axis 维替换为
              ceil(axis_dim_size/64)（或传入group_index时的
              floor(axis_dim_size/64)+group_num），末尾追加长度1的维度。
    """
    _check_scale_alg(scale_alg)
    op_module = builder.load()
    return op_module.swiglu_hi_f4_quant(
        x,
        group_index,
        activate_dim,
        activate_left,
        swiglu_mode,
        clamp_limit,
        glu_alpha,
        glu_bias,
        axis,
        scale_alg,
    )
