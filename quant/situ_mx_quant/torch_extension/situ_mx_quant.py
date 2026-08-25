# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Tuple

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library

# Situ激活gate/up拆分因子：x最后一维为2H，激活输出最后一维为H
SPLIT_FACTOR = 2
# MX量化块大小：每32个元素共享一个scale
MX_BLOCK_SIZE = 32
# y_scale最后一维对齐数：每(align_num * block_size)=64个元素共享一组scale
SCALE_ALIGN_NUM = 2
# dst_type枚举值（与aclDataType保持一致）：36=FLOAT8_E4M3FN，35=FLOAT8_E5M2
DST_TYPE_FLOAT8_E4M3FN = 36
DST_TYPE_FLOAT8_E5M2 = 35
# 支持的量化舍入模式
SUPPORTED_ROUND_MODES = ("rint", "round", "floor")


class SituMxQuantOpBuilder(OpBuilder):
    """
    SituMxQuant 算子的构建器

    基于 aclnnSituMxQuant API 实现，融合 Situ 激活函数和 MX 量化 (Microscaling Quantization)。
    输入为 FP16/BF16，输出为 FP8 (E4M3FN/E5M2) + E8M0 scale。
    """

    def __init__(self):
        super().__init__("situ_mx_quant")

    def sources(self):
        return [self.resolve_source("situ_mx_quant.cpp")]

    def schema(self):
        return (
            "situ_mx_quant("
            "Tensor x, "
            "*, "
            "float beta=1.0, "
            "float linear_beta=0.0, "
            "bool activate_left=False, "
            "int dst_type=36, "
            "str round_mode='rint'"
            ") -> (Tensor y, Tensor y_scale)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def situ_mx_quant_meta(
            x: torch.Tensor,
            *,
            beta: float = 1.0,
            linear_beta: float = 0.0,
            activate_left: bool = False,
            dst_type: int = DST_TYPE_FLOAT8_E4M3FN,
            round_mode: str = "rint",
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            torch._check(
                x.dim() >= 1,
                lambda: f"x must be at least 1-dimensional, but got {x.dim()}-d",
            )
            torch._check(
                x.shape[-1] % SPLIT_FACTOR == 0,
                lambda: f"x last dim must be even, but got {x.shape[-1]}",
            )
            torch._check(
                x.dtype in (torch.float16, torch.bfloat16),
                lambda: f"x dtype must be float16 or bfloat16, but got {x.dtype}",
            )
            torch._check(
                beta > 0.0,
                lambda: f"beta must be greater than 0, but got {beta}",
            )
            torch._check(
                dst_type in (DST_TYPE_FLOAT8_E4M3FN, DST_TYPE_FLOAT8_E5M2),
                lambda: (
                    f"dst_type must be {DST_TYPE_FLOAT8_E4M3FN}(E4M3FN) or "
                    f"{DST_TYPE_FLOAT8_E5M2}(E5M2), but got {dst_type}"
                ),
            )
            torch._check(
                round_mode in SUPPORTED_ROUND_MODES,
                lambda: (
                    f"round_mode must be one of {SUPPORTED_ROUND_MODES}, "
                    f"but got {round_mode}"
                ),
            )
            torch._check(
                round_mode == "rint",
                lambda: f"FP8 output requires round_mode='rint', but got {round_mode}",
            )

            # 输出y：x最后一维减半（Situ激活按gate/up拆分后逐元素相乘）
            y_shape = list(x.shape)
            y_shape[-1] = x.shape[-1] // SPLIT_FACTOR

            # 输出y_scale：每scale_group_size(64)个元素共享一组E8M0 scale
            scale_group_size = SCALE_ALIGN_NUM * MX_BLOCK_SIZE
            scale_num = (y_shape[-1] + scale_group_size - 1) // scale_group_size
            y_scale_shape = y_shape[:-1] + [scale_num, SCALE_ALIGN_NUM]

            if dst_type == DST_TYPE_FLOAT8_E5M2:
                y_dtype = torch.float8_e5m2
            else:
                y_dtype = torch.float8_e4m3fn
            y = torch.empty(y_shape, dtype=y_dtype, device="meta")
            y_scale = torch.empty(
                y_scale_shape, dtype=torch.float8_e8m0fnu, device="meta"
            )
            return y, y_scale


builder = SituMxQuantOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def situ_mx_quant(
    x: torch.Tensor,
    *,
    beta: float = 1.0,
    linear_beta: float = 0.0,
    activate_left: bool = False,
    dst_type: int = DST_TYPE_FLOAT8_E4M3FN,
    round_mode: str = "rint",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NPU 上的 SituMxQuant —— 融合 Situ 激活与 MX（Microscaling）量化

    对输入 x 先做 Situ 激活（gate/up 拆分与 tanh/sigmoid 门控），再按 MX 算法
    做 FP8 动态量化，返回量化结果 y 与 E8M0 格式的 scale 张量 y_scale。

    Args:
        x (torch.Tensor): 必选，输入Tensor。数据类型为 float16/bfloat16，
            shape 为 (..., 2H)，1-7维，最后一维必须为偶数；
            必须是NPU上的Tensor，数据格式ND（空Tensor输出为空）。
        beta (float, optional): Situ激活的beta参数，必须大于0。默认1.0。
        linear_beta (float, optional): up分支的线性beta参数，小于等于0时不启用。默认0.0。
        activate_left (bool, optional): 为True时gate取x的前半部分、up取后半部分；
            为False时gate取后半部分、up取前半部分。默认False。
        dst_type (int, optional): 输出y的数据类型枚举，36=FLOAT8_E4M3FN，
            35=FLOAT8_E5M2，仅支持{35, 36}。默认36。
        round_mode (str, optional): 量化舍入模式，取值为"rint"/"round"/"floor"，
            当前FP8输出仅支持"rint"。默认"rint"。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - y (torch.Tensor): 量化结果，shape 为 x.shape[:-1]+[H]，其中 H=x.shape[-1]//2；
              dtype 为 FLOAT8_E4M3FN 或 FLOAT8_E5M2（由 dst_type 决定）。
            - y_scale (torch.Tensor): MX量化的scale，dtype 为 FLOAT8_E8M0，
              shape 为 x.shape[:-1]+[ceil(H/64), 2]。
    """
    op_module = builder.load()
    return op_module.situ_mx_quant(
        x, beta, linear_beta, activate_left, dst_type, round_mode
    )
