# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


def _situ_glu_shape(x, dim=-1):
    if x.dim() < 1:
        raise RuntimeError("x rank should be greater than 0")
    actual_dim = dim if dim >= 0 else dim + x.dim()
    if actual_dim < 0 or actual_dim >= x.dim():
        raise RuntimeError(
            f"dim should be in [{-x.dim()}, {x.dim() - 1}], but got {dim}"
        )
    dim_size = x.size(actual_dim)
    if dim_size % 2 != 0:
        raise RuntimeError(
            f"x size at dim {actual_dim} should be even, but got {dim_size}"
        )
    shape = list(x.shape)
    shape[actual_dim] = dim_size // 2
    return shape


class SituGluOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("situ_glu")

    def sources(self):
        return [self.resolve_source("situ_glu.cpp")]

    def schema(self):
        return (
            "situ_glu(Tensor x, *, int dim=-1, float beta=1.0, float linear_beta=0.0, "
            "bool activate_left=True) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def situ_glu_meta(x, *, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True):
            return x.new_empty(_situ_glu_shape(x, dim))


builder = SituGluOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def situ_glu(x, *, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True):
    op_module = builder.load()
    return op_module.situ_glu(x, dim, beta, linear_beta, activate_left)
