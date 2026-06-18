# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
#
# ApplyAdamD examples 构建脚本。
# 提供三类示例：
#   - test_aclnn_apply_adam_d.cpp               : aclnn 两段式接口调用，g++ 直编（见 run.sh）。
#   - test_geir_apply_adam_d.cpp                : 图模式（GE/IR 构图）标准调用形态，g++ 直编（见 run.sh）。
#   - test_kernel_direct_apply_adam_d.asc       : kernel 直调（<<<>>>），真实 NPU 可跑通（本 CMakeLists 构建）。
#
# 本 CMakeLists 仅构建 kernel 直调示例（依赖 ASC/bisheng，--npu-arch=dav-2201 = ascend910b）。
# aclnn 与图模式示例由 run.sh 用 g++ + CANN 头/库直编。
cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(apply_adam_d_example LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel")

add_executable(test_kernel_direct_apply_adam_d test_kernel_direct_apply_adam_d.asc)

target_include_directories(test_kernel_direct_apply_adam_d PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${KERNEL_DIR}
    ${ACL_INCLUDE_DIR}
)

target_link_libraries(test_kernel_direct_apply_adam_d PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
    ascendcl
)

target_compile_options(test_kernel_direct_apply_adam_d PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
