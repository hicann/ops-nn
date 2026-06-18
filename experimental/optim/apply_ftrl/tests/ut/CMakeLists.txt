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
# apply_ftrl 算子 UT 工程顶层 CMakeLists.txt（standalone；对齐 apply_rms_prop UT 工程，
# 目标芯片 ascend910b / DAV_2201，架构自适应 aarch64/x86_64）
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(apply_ftrl_ut CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# CANN SDK Paths
# =============================================================================
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()
message(STATUS "ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")

# 架构子目录自适应（aarch64-linux / x86_64-linux）
if(NOT DEFINED ARCH_DIR)
    set(ARCH_DIR "${CMAKE_HOST_SYSTEM_PROCESSOR}-linux")
endif()
message(STATUS "ARCH_DIR: ${ARCH_DIR}")

# apply_ftrl 目标芯片 ascend910b（DAV_2201）。小写以匹配 tiling_case_executor 的 socToUpper 映射。
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "ascend910b")
endif()

# =============================================================================
# Build Google Test with OLD ABI (match CANN libraries)
# =============================================================================
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildGoogleTest.cmake)

# =============================================================================
# Common UT Framework Sources
# =============================================================================
set(UT_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(UT_COMMON_SRCS
    ${UT_COMMON_DIR}/infershape_case_executor.cpp
    ${UT_COMMON_DIR}/infershape_context_faker.cpp
    ${UT_COMMON_DIR}/tiling_case_executor.cpp
    ${UT_COMMON_DIR}/tiling_context_faker.cpp
)

# 仓根（ops-nn worktree 根）：tests/ut 上溯 5 级。third_party/json 为构建期已 fetch 的依赖。
get_filename_component(OPS_NN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../.. ABSOLUTE)
message(STATUS "OPS_NN_ROOT: ${OPS_NN_ROOT}")

# =============================================================================
# Common Include Directories
# =============================================================================
set(UT_COMMON_INCLUDE_DIRS
    ${UT_COMMON_DIR}
    ${OPS_NN_ROOT}/third_party/json/single_include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/include
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/include/aclnn
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/include/base
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/include/base/context_builder
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/asc/include
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/pkg_inc
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/pkg_inc/exe_graph
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/pkg_inc/graph
)

# =============================================================================
# Common Library Directories
# =============================================================================
set(UT_COMMON_LIB_DIRS
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/lib64
)

# =============================================================================
# Compiler Options
# =============================================================================
add_compile_options(
    -Wall
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -Wno-unused-parameter
    -fno-access-control
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "SOC Version: ${SOC_VERSION}")

# =============================================================================
# Add Subdirectories
# =============================================================================
add_subdirectory(op_host)

# op_kernel UT（CPU 孪生 tikicpulib，ICPU_RUN_KF）。默认开启；如本机无 tikicpulib 可 -DENABLE_KERNEL_UT=OFF。
option(ENABLE_KERNEL_UT "Build op_kernel CPU-twin UT" ON)
if(ENABLE_KERNEL_UT)
    add_subdirectory(op_kernel)
endif()
