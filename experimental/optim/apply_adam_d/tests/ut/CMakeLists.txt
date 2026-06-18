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
# apply_adam_d 算子 UT 工程顶层 CMakeLists.txt（独立自包含 op_host UT：Tiling + InferShape）
# 参考 experimental/optim/apply_rms_prop/tests/ut，适配 ascend910b（DAV_2201）+ 多架构(aarch64/x86_64)。
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(apply_adam_d_ut CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# CANN SDK Paths - use official ASCEND_HOME_PATH environment variable
# =============================================================================
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()
message(STATUS "ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")

# Target SoC for this op is ascend910b (DAV_2201)
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend910B")
endif()

# =============================================================================
# Per-arch include/lib directory under CANN (aarch64-linux / x86_64-linux)
# =============================================================================
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ASCEND_ARCH_DIR "aarch64-linux")
else()
    set(ASCEND_ARCH_DIR "x86_64-linux")
endif()
message(STATUS "ASCEND_ARCH_DIR: ${ASCEND_ARCH_DIR}")

# nlohmann/json.hpp (used by tiling_case_executor.cpp for platform-info JSON).
# Resolved by run.sh (repo third_party) and passed via -DNLOHMANN_JSON_INCLUDE.
if(NOT DEFINED NLOHMANN_JSON_INCLUDE)
    set(NLOHMANN_JSON_INCLUDE "")
endif()
message(STATUS "NLOHMANN_JSON_INCLUDE: ${NLOHMANN_JSON_INCLUDE}")

# =============================================================================
# Build Google Test from source with OLD ABI (match CANN libraries)
# =============================================================================
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildGoogleTest.cmake)

# =============================================================================
# Common UT framework sources (op-agnostic fakers / executors)
# =============================================================================
set(UT_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(UT_COMMON_SRCS
    ${UT_COMMON_DIR}/infershape_case_executor.cpp
    ${UT_COMMON_DIR}/infershape_context_faker.cpp
    ${UT_COMMON_DIR}/tiling_case_executor.cpp
    ${UT_COMMON_DIR}/tiling_context_faker.cpp
)

# =============================================================================
# Common include directories
# =============================================================================
set(UT_COMMON_INCLUDE_DIRS
    ${UT_COMMON_DIR}
    ${NLOHMANN_JSON_INCLUDE}
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/include
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/include/aclnn
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/include/base
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/include/base/context_builder
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc/include
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/exe_graph
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/graph
)

# =============================================================================
# Common library directories
# =============================================================================
set(UT_COMMON_LIB_DIRS
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/lib64
)

# =============================================================================
# Compiler options / build type
# =============================================================================
add_compile_options(
    -Wall
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -fno-access-control
)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "SOC Version: ${SOC_VERSION}")

# =============================================================================
# Subdirectories
# =============================================================================
add_subdirectory(op_host)
