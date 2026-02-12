# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(CONV3D_TILING_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../conv/conv3d_v2/op_host/op_tiling/arch35")
set(COMMOM_TILING_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../conv/common/op_host/op_tiling/arch35")

set(CONV3D_TILING_SOURCES 
    "${CONV3D_TILING_DIR}/conv3d_v2_api_tiling.cpp"
    "${CONV3D_TILING_DIR}/conv3d_v2_base_tiling_template_tilingkey.cpp"
    "${COMMOM_TILING_DIR}/conv_api_tiling_algorithm_base.cpp"
    "${COMMOM_TILING_DIR}/conv_api_tiling_algorithm_BBmode.cpp"
    "${COMMOM_TILING_DIR}/conv_api_tiling_algorithm_HWmode.cpp"
    "${COMMOM_TILING_DIR}/conv_api_tiling_algorithm_Mmode.cpp"
    "${COMMOM_TILING_DIR}/conv_api_tiling_base.cpp"
    "${COMMOM_TILING_DIR}/conv_api_tiling_util.cpp"
    "${COMMOM_TILING_DIR}/conv_base_numblocks_decision.cpp"
)
add_sources("--npu-arch=dav-3101" "${CONV3D_TILING_SOURCES}")