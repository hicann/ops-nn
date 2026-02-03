# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(acl_execute_matmul_custom)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../output")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../output")

set(INC_PATH $ENV{DDK_PATH})

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/cann")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/custom_nn/op_api")

set(LIB_PATH $ENV{NPU_HOST_LIB})

# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_NAME_LOWER)
    set(LIB_PATH "/usr/local/Ascend/cann/${CMAKE_SYSTEM_PROCESSOR}-${SYSTEM_NAME_LOWER}/devlib")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -B/usr/lib/gold-ld")

# Header path
include_directories(
    ../inc
    ${INC_PATH}/include
    ${CUST_PKG_PATH}/include
)

# add host lib path
link_directories(
    ${INC_PATH}/lib64
    ${LIB_PATH}
    ${CUST_PKG_PATH}/lib
    ${INC_PATH}/x86_64-linux/lib64
)

add_executable(execute_weight_quant_batch_matmul_experiment_op
    operator_desc.cpp
    op_runner.cpp
    main.cpp
    op_runner.cpp
    common.cpp
)

target_link_libraries(execute_weight_quant_batch_matmul_experiment_op
    nnopbase
    cust_opapi
    ascendcl
    acl_op_compiler
    stdc++
)

install(TARGETS execute_weight_quant_batch_matmul_experiment_op DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
