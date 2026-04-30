# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # SmoothL1LossV2 的 tiling 依赖文件（来自 op_host）
    set(smooth_l1_loss_v2_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/smooth_l1_loss_v2_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/smooth_l1_loss_v2_infershape.cpp
    )

    # AddOpTestCase 参数说明：
    # param1: 算子名称（与 kernel 函数名一致）
    # param2: soc 版本（多个用分号分隔）
    # param3: 编译宏（选择一组典型 dtype 组合即可）
    # param4: 该算子依赖的所有 tiling 源码文件
    AddOpTestCase(smooth_l1_loss_v2 "ascend910b"
        "-DDTYPE_PREDICT=float -DDTYPE_LABEL=float -DDTYPE_LOSS=float"
        "${smooth_l1_loss_v2_tiling_files}")
endif()
