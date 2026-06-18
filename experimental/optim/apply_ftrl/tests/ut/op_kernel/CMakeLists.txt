# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# apply_ftrl 算子 op_kernel UT（CPU 孪生 tikicpulib，ICPU_RUN_KF 直跑主线 kernel）
# ----------------------------------------------------------------------------------------------------------

set(OP_NAME apply_ftrl)
set(KERNEL_UT_EXE ${OP_NAME}_op_kernel_ut)

# CPU 孪生芯片型号（本机 Ascend910B3 / DAV_2201）。可经 -DKERNEL_SOC 覆盖。
if(NOT DEFINED KERNEL_SOC)
    set(KERNEL_SOC "Ascend910B3")
endif()

# tikicpulib CMake 包（提供 __CCE_KT_TEST__ 编译环境 + CPU 孪生运行时）
list(APPEND CMAKE_PREFIX_PATH $ENV{ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake)
find_package(tikicpulib REQUIRED)

set(OP_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel)

add_executable(${KERNEL_UT_EXE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_${OP_NAME}.cpp
)

# asc/include 必须排在 tikicpulib 的 tikcfw INTERFACE include 之前，使 "kernel_operator.h"
# 解析到 asc 树（提供 GET_TILING_DATA_WITH_STRUCT / REGISTER_TILING_DEFAULT，与主线 op_kernel 一致）。
target_include_directories(${KERNEL_UT_EXE} BEFORE PRIVATE
    ${OP_KERNEL_DIR}
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/asc/include
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/include
    $ENV{ASCEND_HOME_PATH}/include
)

# tikicpulib 目标已带 __CCE_KT_TEST__ / __NPU_ARCH__=2201 / ASCENDC_CPU_DEBUG / OLD-ABI 等定义。
target_link_libraries(${KERNEL_UT_EXE} PRIVATE
    tikicpulib::${KERNEL_SOC}
    gtest
    gtest_main
)

set_target_properties(${KERNEL_UT_EXE} PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ENV{ASCEND_HOME_PATH}/lib64;$ENV{ASCEND_HOME_PATH}/tools/tikicpulib/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

message(STATUS "=== ApplyFtrl op_kernel UT (CPU twin) ===")
message(STATUS "  Kernel UT Executable : ${KERNEL_UT_EXE}")
message(STATUS "  CPU twin SoC         : ${KERNEL_SOC}")
message(STATUS "=========================================")
