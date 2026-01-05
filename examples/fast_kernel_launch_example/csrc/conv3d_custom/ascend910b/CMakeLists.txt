# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

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

add_sources("--npu-arch=dav-2201" "${CONV3D_TILING_SOURCES}")