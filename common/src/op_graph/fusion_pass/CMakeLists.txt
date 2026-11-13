# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_cube_utils_plugin_modules()
target_include_directories(${CUBE_UTILS_PLUGIN_NAME}_obj PRIVATE
    ${ASCEND_DIR}/include/external)
target_sources(${CUBE_UTILS_PLUGIN_NAME}_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bias_optimize_quant_rollback_base.cpp)
