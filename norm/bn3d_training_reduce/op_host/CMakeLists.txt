# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 本算子不对外提供 aclnn 接口（图模式专用），故 ACLNNTYPE 取 aclnn_exclude。
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE bn3d_training_reduce
        ACLNNTYPE aclnn_exclude DEPENDENCIES norm_common)

# CheckSupport 需要随 tiling so 注册；独立编译避免 OP_ADD 覆盖算子已有的 TilingParse 注册。
target_sources(${OPHOST_NAME}_tiling_obj PRIVATE bn3d_training_reduce_check_support.cpp)
