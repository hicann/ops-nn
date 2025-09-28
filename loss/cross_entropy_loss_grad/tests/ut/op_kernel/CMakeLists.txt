# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
return()
file(GLOB CURRENT_SOURCE_DIRS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)
message(STATUS "=== Debug: CURRENT_SOURCE_DIRS =${CURRENT_SOURCE_DIRS} ")
foreach(SUB_DIR ${CURRENT_SOURCE_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

set(add_cross_entropy_loss_grad_tiling_files
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/cross_entropy_loss_grad_tiling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/cross_entropy_loss_grad_tiling_arch35.cpp
    )

if (UT_TEST_ALL OR FAST_OP_UT)
    AddOpTestCase(cross_entropy_loss_grad "ascend910B1" "-DDTYPE_INPUT=float -DDTYPE_TARGET=int64_t" "${add_cross_entropy_loss_grad_tiling_files}")
endif()
