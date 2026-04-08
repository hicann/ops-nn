# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# NOTE: Portions of this code were AI-generated and have been
# technically reviewed for functional accuracy and security


# 设置算子定义时支持的芯片类型
set(SUPPORT_COMPUTE_UNIT "ascend950")
# 设置每种芯片类型对应的tiling文件目录，即采用op_host目录下哪个文件夹下的tiling文件编译
set(SUPPORT_TILING_DIR "arch35")

message(STATUS "=== Debug: start ops.activation.hard_swish.CMakeLists.txt ")
file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST AND NOT BENCHMARK)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    else()
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/op_host/CMakeLists.txt")
            add_subdirectory(${SUB_DIR}/op_host)
        endif()
    endif()
endforeach()