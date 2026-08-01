# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/
if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(deep_norm "ascend910B1" "" )
    AddOpTestCase(deep_norm "ascend950pr_9599" "-DDEEP_NORM_ARCH35_KERNEL_UT" MULTI_KERNEL_TARGET norm_common)
    if(TARGET ascendc_kernel_ut_src_copy_deep_norm)
        if(NOT TARGET norm_common_ut_src_copy)
            add_custom_target(norm_common_ut_src_copy
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${PROJECT_SOURCE_DIR}/norm/norm_common/op_kernel
                    ${CMAKE_BINARY_DIR}/tbe/ascendc/norm_common
            )
        endif()
        add_dependencies(ascendc_kernel_ut_src_copy_deep_norm norm_common_ut_src_copy)
    endif()
endif()
