# -----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
#
# Please refer to the License for details. You may not use this file except in compliance with the License.
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------

# Keep the directory build-neutral if a future partial checkout contains only
# the public ACLNN scaffold.  A complete checkout has both the renamed lower-op
# definition and the 910B kernel entry and is registered normally below.
set(MAXPOOL2D_MASK_BACKWARD_DEF
    "${CMAKE_CURRENT_SOURCE_DIR}/op_host/max_pool2d_with_mask_backward_def.cpp")
set(MAXPOOL2D_MASK_BACKWARD_KERNEL
    "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/max_pool2d_with_mask_backward.cpp")

if(NOT EXISTS "${MAXPOOL2D_MASK_BACKWARD_DEF}" OR
   NOT EXISTS "${MAXPOOL2D_MASK_BACKWARD_KERNEL}")
    message(STATUS "max_pool2d_with_mask_backward: scaffold only; build registration deferred")
    return()
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST AND NOT BENCHMARK)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
