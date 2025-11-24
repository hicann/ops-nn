# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

message(STATUS "BUILD_TORCH_OPS ON in conv3d_custom")
# CONV3D_CUSTOM operation sources
file(GLOB CONV3D_CUSTOM_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
set(CONV3D_TILING_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../conv/conv3d_v2/op_host/op_tiling")
set(CONV3D_TILING_SOURCES
    "${CONV3D_TILING_DIR}/conv3d_api_tiling_algorithm.cpp"
    "${CONV3D_TILING_DIR}/conv3d_api_tiling_algorithm_hw_mode.cpp"
    "${CONV3D_TILING_DIR}/conv3d_api_tiling_algorithm_pointwise.cpp"
    "${CONV3D_TILING_DIR}/conv3d_api_tiling_base.cpp"
    "${CONV3D_TILING_DIR}/conv3d_api_tiling.cpp"
    "${CONV3D_TILING_DIR}/conv3d_api_tiling_utils.cpp"
    "${CONV3D_TILING_DIR}/conv3d_tiling_engine.cpp"
    "${CONV3D_TILING_DIR}/conv3d_tiling_utils.cpp"
)
set(CONV3D_CUSTOM_SOURCES ${CONV3D_CUSTOM_NPU_SOURCES} ${CONV3D_TILING_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${CONV3D_CUSTOM_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=CubeCore --cce-auto-sync -xcce"
)

# Create object library
add_library(conv3d_custom_objects OBJECT ${CONV3D_CUSTOM_SOURCES})

target_compile_options(conv3d_custom_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(conv3d_custom_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()
