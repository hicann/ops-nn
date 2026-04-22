# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------


if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    set(fused_bias_leaky_relu_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/fused_bias_leaky_relu_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/fused_bias_leaky_relu_infershape.cpp
        )
    AddOpTestCase(fused_bias_leaky_relu_apt "ascend950" "-DDTYPE_X=float" "${fused_bias_leaky_relu_tiling_files}")
endif()