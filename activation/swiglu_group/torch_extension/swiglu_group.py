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


def _swiglu_shape(x):
    if x.dim() < 1:
        raise RuntimeError("x rank should be greater than 0")
    last_dim = x.size(x.dim() - 1)
    if last_dim % 2 != 0:
        raise RuntimeError("x last dim size should be even")
    shape = list(x.shape)
    shape[-1] = last_dim // 2
    return shape


class SwigluGroupOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("swiglu_group")

    def sources(self):
        return ["csrc/activation/swiglu_group.cpp"]

    def schema(self):
        return (
            "swiglu_group(Tensor x, *, Tensor? weight=None, Tensor? group_index=None, "
            "float clamp_limit=-1.0) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def swiglu_group_meta(x, *, weight=None, group_index=None, clamp_limit=-1.0):
            return x.new_empty(_swiglu_shape(x))


builder = SwigluGroupOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def swiglu_group(x, *, weight=None, group_index=None, clamp_limit=-1.0):
    op_module = builder.load()
    return op_module.swiglu_group(x, weight, group_index, clamp_limit)
