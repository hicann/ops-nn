# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

__all__ = [
    "swiglu_group_grad",
    "swiglu_group_backward",
    "convert_swiglu_group_grad",
    "convert_swiglu_group_backward",
]

from .swiglu_group_grad import swiglu_group_backward
from .graph_convert_swiglu_group_grad import convert_swiglu_group_backward

# The repository operator directory is named swiglu_group_grad, while the
# public Torch dispatcher API keeps the reviewed swiglu_group_backward name.
swiglu_group_grad = swiglu_group_backward
convert_swiglu_group_grad = convert_swiglu_group_backward
