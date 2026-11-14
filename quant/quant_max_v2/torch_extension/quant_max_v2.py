# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""PyTorch binding for aclnnQuantMaxV2, exposed as quant_max."""

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


_OUTPUT_DTYPES = {
    34: torch.uint8,  # HIFLOAT8 bits; PyTorch has no native HIFLOAT8 dtype.
    35: torch.float8_e5m2,
    36: torch.float8_e4m3fn,
}


class QuantMaxV2OpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("quant_max_v2")

    def sources(self):
        return [self.resolve_source("quant_max_v2.cpp")]

    def schema(self):
        return (
            "quant_max(Tensor x, Tensor? scale=None, "
            "Tensor? group_list=None, *, "
            "str round_mode, int dst_type, "
            "float dst_type_max=15.0) -> (Tensor y, Tensor? y_scale, Tensor amax)"
        )

    def register_meta(self):
        @impl(get_as_library(), "quant_max", "Meta")
        def meta(
            x, scale=None, group_list=None, *, round_mode, dst_type, dst_type_max=15.0
        ):
            torch._check(x.dim() >= 2 and x.dim() <= 8, lambda: "x rank must be 2..8")
            torch._check(dst_type in _OUTPUT_DTYPES, lambda: "unsupported dst_type")
            groups = 1 if group_list is None else group_list.numel()
            # 第二输出与 C++ 实现保持一致：动态为 [groups] fp32，静态为 [0] fp32 占位张量
            # （而非 None），使 eager / Meta / 编译路径的输出结构相同。
            return (
                torch.empty_like(x, dtype=_OUTPUT_DTYPES[dst_type]),
                x.new_empty((groups,) if scale is None else (0,), dtype=torch.float32),
                x.new_empty((groups,)),
            )


_builder = QuantMaxV2OpBuilder()
_builder._ensure_initialized()


@impl(get_as_library(), "quant_max", "PrivateUse1")
def quant_max(
    x, scale=None, group_list=None, *, round_mode, dst_type, dst_type_max=15.0
):
    """Quantize per tensor or per group; None scale selects dynamic mode.

    Args:
        x: 2..8-D NPU tensor, float32/float16/bfloat16.
        scale: per-group float32 multiplier (= inverse of the conventional
            quantization scale). Omit for dynamic mode, where y_scale is
            derived from amax as amax / dst_type_max.
        group_list: int64 [num_groups] of cumulative end rows along dim 0
            (adjacent equal entries mean empty groups; monotonic, last entry
            equals x.shape[0] -- caller-guaranteed, consistent with V1).
            Omit for per-tensor mode (equivalent to G=1).
        round_mode: "round"/"hybrid" for dst_type=34 (HIFLOAT8), "rint" only
            for FP8 targets.
        dst_type: 34=HIFLOAT8, 35=FLOAT8_E5M2, 36=FLOAT8_E4M3FN.
        dst_type_max: full-scale value in [0, 32768]; effective only for
            HIFLOAT8 (FP8 targets override it with the type maximum).

    Returns:
        (y, y_scale, amax): y has x's shape with dst_type's dtype (uint8 bit
        patterns for HIFLOAT8); amax is always [num_groups]; y_scale is
        [num_groups] float32 in dynamic mode and a [0] placeholder otherwise.
    """
    module = _builder.load()
    return module.quant_max(x, scale, group_list, round_mode, dst_type, dst_type_max)


# Entry-point compatibility: setup.py derives the distribution entry point and
# staged module name from the operator directory name.
quant_max_v2 = quant_max
