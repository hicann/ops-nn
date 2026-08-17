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


class SituGluGradOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("situ_glu_grad")

    def sources(self):
        return [self.resolve_source("situ_glu_grad.cpp")]

    def schema(self):
        return (
            "situ_glu_grad(Tensor grad_y, Tensor x, *, int dim=-1, "
            "float beta=1.0, float linear_beta=0.0, bool activate_left=True) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def situ_glu_grad_meta(
            grad_y, x, *, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True
        ):
            return x.new_empty(x.shape)


builder = SituGluGradOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def situ_glu_grad(grad_y, x, *, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True):
    op_module = builder.load()
    return op_module.situ_glu_grad(grad_y, x, dim, beta, linear_beta, activate_left)
