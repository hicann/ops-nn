# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional, Tuple, List
import torch
from torch.library import impl
from cann_ops_nn.op_builder import OpBuilder, get_as_library


class QuantMatmulActivationQuantOpBuilder(OpBuilder):
    """
    QuantMatmulActivationQuant 算子的构建器
    基于 aclnnQuantMatmulActivationQuant API 实现，融合量化矩阵乘法、激活函数与输出量化。
    """

    def __init__(self):
        super().__init__("quant_matmul_activation_quant")

    def sources(self) -> list:
        return ["csrc/matmul/quant_matmul_activation_quant.cpp"]

    def schema(self) -> str:
        return (
            "quant_matmul_activation_quant("
            "Tensor x1, Tensor x2, Tensor x2_scale, *, Tensor? x1_scale=None, Tensor? bias=None, "
            "int? output_dtype=None, int? x1_dtype=None, int? x2_dtype=None, int? x1scale_dtype=None, int? x2scale_dtype=None, "
            'int[]? group_sizes=None, str activation_type="gelu_tanh", str quant_mode="mx", '
            'str round_mode="rint", int? scale_alg=0, float dst_type_max=0.0)'
            " -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def quant_matmul_activation_quant_meta(
            x1: torch.Tensor,
            x2: torch.Tensor,
            x2_scale: torch.Tensor,
            *,
            x1_scale: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor] = None,
            output_dtype: Optional[int] = None,
            x1_dtype: Optional[int] = None,
            x2_dtype: Optional[int] = None,
            x1scale_dtype: Optional[int] = None,
            x2scale_dtype: Optional[int] = None,
            group_sizes: Optional[List[int]] = None,
            activation_type: str = "gelu_tanh",
            quant_mode: str = "mx",
            round_mode: str = "rint",
            scale_alg: Optional[int] = 0,
            dst_type_max: float = 0.0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            x1_dim_num = x1.dim()
            x2_dim_num = x2.dim()

            x1_a = x1.size(x1_dim_num - 2)
            x1_b = x1.size(x1_dim_num - 1)
            x2_a = x2.size(x2_dim_num - 2)
            x2_b = x2.size(x2_dim_num - 1)

            if x1_b == x2_a:
                m = x1_a
                n = x2_b
            elif x1_b == x2_b:
                m = x1_a
                n = x2_a
            elif x1_a == x2_a:
                m = x1_b
                n = x2_b
            elif x1_a == x2_b:
                m = x1_b
                n = x2_a
            else:
                torch._check(
                    False,
                    lambda: f"cannot infer m/n: no matching k dimension between "
                    f"x1 last two dims [{x1_a}, {x1_b}] and "
                    f"x2 last two dims [{x2_a}, {x2_b}]",
                )

            # 计算 output_size (广播 batch dims + M + N)
            x1_batch = list(x1.shape[:-2])
            x2_batch = list(x2.shape[:-2])
            batch_size = list(torch.broadcast_shapes(x1_batch, x2_batch))
            output_size = batch_size + [m, n]

            # 3. 计算 scale_size (包含分块与对齐逻辑)
            scale_size = list(output_size)
            axis_change = len(output_size) - 1

            ALIGN_NUM = 2
            BLOCK_SIZE = 32

            # 等价于 C++ 的 op_infer::CeilDiv(a, b) -> (a + b - 1) // b
            dim_size = (scale_size[axis_change] + BLOCK_SIZE - 1) // BLOCK_SIZE
            dim_size = (dim_size + ALIGN_NUM - 1) // ALIGN_NUM

            scale_size[axis_change] = dim_size
            scale_size.append(ALIGN_NUM)

            # 4. 判断特殊输出类型 (FLOAT4)
            output_dtype_val = output_dtype if output_dtype is not None else 0
            DTYPE_FLOAT4_E2M1 = 30
            DTYPE_FLOAT4_E1M2 = 31
            special_output_type = (
                output_dtype_val == DTYPE_FLOAT4_E2M1
                or output_dtype_val == DTYPE_FLOAT4_E1M2
            )

            # 5. 分配 output tensor
            if special_output_type:
                last_dim_val = output_size[-1]
                torch._check(
                    last_dim_val % 2 == 0,
                    lambda: "The last dim output shape must be divisible by 2 if output dtype is FLOAT4_E2M1 or FLOAT4_E1M2",
                )
                output_size[-1] = last_dim_val // 2
                y = torch.empty(output_size, dtype=torch.uint8, device="meta")
            else:
                y = torch.empty(output_size, dtype=x1.dtype, device="meta")

            # 6. 分配 scale tensor
            scale = torch.empty(scale_size, dtype=torch.float8_e8m0fnu, device="meta")

            return y, scale


# 实例化 Builder 并加载 C++ 扩展模块
quant_matmul_activation_quant_builder = QuantMatmulActivationQuantOpBuilder()
quant_matmul_activation_quant_builder._ensure_initialized()


# 注册 NPU (PrivateUse1) 后端的具体实现
@impl(get_as_library(), quant_matmul_activation_quant_builder.name, "PrivateUse1")
def quant_matmul_activation_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x2_scale: torch.Tensor,
    *,
    x1_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[int] = None,
    x1_dtype: Optional[int] = None,
    x2_dtype: Optional[int] = None,
    x1scale_dtype: Optional[int] = None,
    x2scale_dtype: Optional[int] = None,
    group_sizes: Optional[List[int]] = None,
    activation_type: str = "gelu_tanh",
    quant_mode: str = "mx",
    round_mode: str = "rint",
    scale_alg: Optional[int] = 0,
    dst_type_max: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NPU 上的 QuantMatmulActivationQuant — 融合量化矩阵乘法、激活函数与输出量化
    """
    op_module_matmul = quant_matmul_activation_quant_builder.load()
    return op_module_matmul.quant_matmul_activation_quant(
        x1,
        x2,
        x2_scale,
        x1_scale,
        bias,
        output_dtype,
        x1_dtype,
        x2_dtype,
        x1scale_dtype,
        x2scale_dtype,
        group_sizes,
        activation_type,
        quant_mode,
        round_mode,
        scale_alg,
        dst_type_max,
    )
