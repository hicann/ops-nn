# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# The backward replay recomputes the forward gates, so op_kernel/arch35 includes
# single_layer_lstm's single_layer_lstm_gate_math.h. Kernel sources are compiled out of
# build/tbe/ascendc/<op>/, where the two operators sit side by side, which is what that
# include is relative to. Naming the dependency is what copies the forward's kernel sources
# there when a build or a single-operator package selects only this operator.
add_all_modules_sources(
    OPTYPE single_layer_lstm_grad
    ACLNNTYPE aclnn_exclude
    DEPENDENCIES single_layer_lstm
)

add_kernel_sources(
    KERNEL_SRC single_layer_lstm_grad.cpp
    COMPUTE_UNITS ascend910b ascend910_93
)

add_kernel_sources(
    KERNEL_SRC arch35/single_layer_lstm_grad.cpp
    COMPUTE_UNITS ascend950
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush"
)
