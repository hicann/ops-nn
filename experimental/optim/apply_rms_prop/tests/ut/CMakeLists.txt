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
# apply_rms_prop 算子 UT 工程顶层 CMakeLists.txt
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(apply_rms_prop_ut CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# CANN SDK Paths - Use official ASCEND_HOME_PATH environment variable
# =============================================================================
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()

message(STATUS "ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")

# Default SOC version for UT（atvoss 仅支持 ascend950）
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend950")
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

# =============================================================================
# Common Include Directories
# =============================================================================
set(UT_COMMON_INCLUDE_DIRS
    ${UT_COMMON_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/include
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/include/aclnn
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/include/base
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/include/base/context_builder
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/exe_graph
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/graph
)

# =============================================================================
# Common Library Directories
# =============================================================================
set(UT_COMMON_LIB_DIRS
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/lib64
)

# =============================================================================
# Compiler Options
# =============================================================================
add_compile_options(
    -Wall
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -fno-access-control
)

# =============================================================================
# Build Type
# =============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "SOC Version: ${SOC_VERSION}")

# =============================================================================
# Add Subdirectories
# =============================================================================
add_subdirectory(op_host)
# op_api 在迭代一（P1 按需）暂不编译
