# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional, Tuple

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class DequantSituQuantOpBuilder(OpBuilder):
    """
    DequantSituQuant 算子的构建器

    基于 aclnnDequantSituQuant API 实现，融合反量化 (Dequant)、Situ 激活函数和量化 (Quant) 三个操作。
    支持 INT8 (per-channel dequant + static/dynamic quant)、INT32 (MoE grouped-matmul + dynamic quant)
    和 BF16/FP16 (pre-dequantized + dynamic quant) 三种输入路径。
    """

    def __init__(self):
        super().__init__("dequant_situ_quant")

    def sources(self):
        return [self.resolve_source("dequant_situ_quant.cpp")]

    def schema(self):
        return (
            "dequant_situ_quant("
            "Tensor x, "
            "*, Tensor? weight_scale=None, "
            "Tensor? activation_scale=None, "
            "Tensor? bias=None, "
            "Tensor? quant_scale=None, "
            "Tensor? quant_offset=None, "
            "Tensor? group_index=None, "
            "float beta=4.0, "
            "float linear_beta=25.0, "
            "bool activate_left=True, "
            'str quant_type="dynamic"'
            ") -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def dequant_situ_quant_meta(
            x: torch.Tensor,
            *,
            weight_scale: Optional[torch.Tensor] = None,
            activation_scale: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor] = None,
            quant_scale: Optional[torch.Tensor] = None,
            quant_offset: Optional[torch.Tensor] = None,
            group_index: Optional[torch.Tensor] = None,
            beta: float = 4.0,
            linear_beta: float = 25.0,
            activate_left: bool = True,
            quant_type: str = "dynamic",
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            is_int8 = x.dtype == torch.int8
            is_int32 = x.dtype == torch.int32
            is_bf16 = x.dtype == torch.bfloat16
            is_fp16 = x.dtype == torch.float16
            torch._check(
                is_int8 or is_int32 or is_bf16 or is_fp16,
                lambda: (
                    f"x dtype must be int8, int32, bfloat16, or float16, but got {x.dtype}"
                ),
            )

            if is_int8:
                torch._check(
                    x.dim() >= 2,
                    lambda: (
                        f"x must be at least 2-dimensional for int8, but got {x.dim()}-d"
                    ),
                )
            else:
                torch._check(
                    x.dim() == 2,
                    lambda: (
                        f"x must be 2-dimensional for int32/bfloat16, but got {x.dim()}-d"
                    ),
                )

            torch._check(
                x.shape[-1] % 2 == 0,
                lambda: f"x last dim must be even, but got {x.shape[-1]}",
            )

            y_shape = list(x.shape)
            y_shape[-1] = x.shape[-1] // 2
            if is_int8:
                scale_shape = list(x.shape[:-1])
            else:
                scale_shape = [x.shape[0]]
            y = torch.empty(y_shape, dtype=torch.int8, device="meta")
            scale = torch.empty(scale_shape, dtype=torch.float32, device="meta")
            return y, scale


builder = DequantSituQuantOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def _dequant_situ_quant(
    x: torch.Tensor,
    *,
    weight_scale: Optional[torch.Tensor] = None,
    activation_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    quant_scale: Optional[torch.Tensor] = None,
    quant_offset: Optional[torch.Tensor] = None,
    group_index: Optional[torch.Tensor] = None,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    activate_left: bool = True,
    quant_type: str = "dynamic",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PrivateUse1 分发的 NPU 实现：透传到 JIT 编译产物。"""
    op_module = builder.load()
    return op_module.dequant_situ_quant(
        x,
        weight_scale,
        activation_scale,
        bias,
        quant_scale,
        quant_offset,
        group_index,
        beta,
        linear_beta,
        activate_left,
        quant_type,
    )


def dequant_situ_quant(
    x: torch.Tensor,
    *,
    weight_scale: Optional[torch.Tensor] = None,
    activation_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    quant_scale: Optional[torch.Tensor] = None,
    quant_offset: Optional[torch.Tensor] = None,
    group_index: Optional[torch.Tensor] = None,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    activate_left: bool = True,
    quant_type: str = "dynamic",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    融合反量化 (Dequant)、Situ 激活与量化 (Quant) 算子（底层调用 aclnnDequantSituQuant）。

    按 x 的 dtype 分三条路径：
      - int8：dequantOut = cast(x) * weight_scale + bias，weight_scale 必选，
        activation_scale/group_index 必须为 None；
      - int32：dequantOut = cast(x) * weight_scale * activation_scale + bias，
        weight_scale/activation_scale 必选，quant_scale/quant_offset 必须为 None；
      - bfloat16/float16：预反量化路径，dequantOut = cast(x)，所有可选输入必须为 None。
    随后执行 Situ 激活（activate_left 决定 gate/up 分别取 dequantOut 的前/后半），
    再按 quant_type 执行 static 或 dynamic 量化。

    Args:
        x (torch.Tensor): 必选，NPU 上的连续 Tensor。int8 时维度 >= 2 且最后一维为 2H；
            其余 dtype 时必须为 2 维 [M, 2H]；最后一维必须为偶数。
            dtype 取值：torch.int8 / torch.int32 / torch.bfloat16 / torch.float16。
        weight_scale (Optional[torch.Tensor]): 反量化 weight scale，1 维 float32，
            shape 为 (2H,) 或 (1,)。int8/int32 路径必选；bfloat16/float16 路径必须为 None。
        activation_scale (Optional[torch.Tensor]): 反量化 activation scale，1 维 float32，
            shape 为 (1,)。仅 int32 路径必选；int8/bfloat16/float16 路径必须为 None。
        bias (Optional[torch.Tensor]): 反量化 bias，1 维 float32，shape 为 (2H,) 或 (1,)。
            int8/int32 路径可选；bfloat16/float16 路径必须为 None。
        quant_scale (Optional[torch.Tensor]): 量化 scale，1 维 float32，shape 为 (H,) 或 (1,)。
            quant_type="static" 时必选；"dynamic" 时可选（作为 smoothScale）；
            int32 路径必须为 None。
        quant_offset (Optional[torch.Tensor]): 量化 offset，1 维 float32，shape 为 (H,) 或 (1,)，
            仅 static 模式生效；int32 路径必须为 None。
        group_index (Optional[torch.Tensor]): MoE 分组索引，1 维 int64，shape 为 (K,)。
            仅 int32 路径可选提供；int8/bfloat16/float16 路径必须为 None。
        beta (float): Situ 激活 beta 参数，默认 4.0，不能为 0。
        linear_beta (float): Situ 激活 up 分支线性 beta 参数，默认 25.0，<=0 时不启用。
        activate_left (bool): True 时 gate 取 dequantOut 前半部分、up 取后半部分，
            False 时相反。默认 True。
        quant_type (str): 量化模式，取值 "static" 或 "dynamic"，默认 "dynamic"。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - y：量化输出，int8，shape 为 x.shape[:-1] + [H]（int8 路径）或 [M, H]
              （其余路径），其中 H = x.shape[-1] // 2；
            - y_scale：动态量化 scale，float32，shape 为 x.shape[:-1]（int8 路径）或 [M]
              （其余路径），仅 dynamic 模式有意义。

    Examples:
        >>> import torch, torch_npu, cann_ops_nn
        >>> x = torch.randint(-127, 127, (16, 64), dtype=torch.int8).npu()
        >>> weight_scale = torch.full((64,), 0.1, dtype=torch.float32).npu()
        >>> y, y_scale = cann_ops_nn.dequant_situ_quant(x, weight_scale=weight_scale)
        >>> y.shape, y_scale.shape
        (torch.Size([16, 32]), torch.Size([16]))
    """
    return torch.ops.cann_ops_nn.dequant_situ_quant(
        x,
        weight_scale=weight_scale,
        activation_scale=activation_scale,
        bias=bias,
        quant_scale=quant_scale,
        quant_offset=quant_offset,
        group_index=group_index,
        beta=beta,
        linear_beta=linear_beta,
        activate_left=activate_left,
        quant_type=quant_type,
    )
