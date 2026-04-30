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
    # 算子自己的tiling文件路径
    set(lp_loss_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/lp_loss_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/lp_loss_infershape.cpp
        )
    # 使用AddOpTestCase
    # param1：算子名称，以kernel方式命名
    # param2：soc版本
    # param3：自定义编译选项
    # param4：该算子依赖的所有tiling源码文件
    AddOpTestCase(lp_loss "ascend910B1" "-DDTYPE_PREDICT=float -DDTYPE_Y=float" "${lp_loss_tiling_files}")
endif()