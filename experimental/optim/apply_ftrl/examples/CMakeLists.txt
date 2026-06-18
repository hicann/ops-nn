# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# CMake 构建 — ApplyFtrl examples
#
# 构建 Kernel 直调 (<<<>>>) 可运行样例 test_kernel_direct_apply_ftrl.asc，
# 复用本算子 op_kernel/apply_ftrl_kernel.h（NsApplyFtrl::ApplyFtrl<T,PAD_TAIL,HAS_L1>）；
# 目标架构 dav-2201 (Ascend910B / Atlas A2/A3)。一键运行见 run.sh。
#
# 注：GE 图模式样例 test_geir_apply_ftrl.cpp 不在此 CMake 构建——它依赖 GE/graph 运行时
# 与算子安装进 OPP，经仓库统一的 `bash build.sh --run_example apply_ftrl graph` 流程构建运行。
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(apply_ftrl_examples LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

# 本算子 op_kernel 目录（apply_ftrl_kernel.h / apply_ftrl_tiling_data.h / apply_ftrl_tiling_key.h）。
get_filename_component(OP_KERNEL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel" ABSOLUTE)

add_executable(test_kernel_direct_apply_ftrl test_kernel_direct_apply_ftrl.asc)

target_link_libraries(test_kernel_direct_apply_ftrl PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(test_kernel_direct_apply_ftrl PRIVATE
    ${OP_KERNEL_DIR}
)

# Ascend910B (Atlas A2/A3) -> DAV_2201
target_compile_options(test_kernel_direct_apply_ftrl PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
