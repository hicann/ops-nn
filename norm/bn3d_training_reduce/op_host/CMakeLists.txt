# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# BN3DTrainingReduce 在 canndev 中已有 910/910B/910_93 的 TBE 实现。本仓只为
# Ascend 950 提供 arch35 Tiling；必须在构建期隔离，不能让 950 的
# IMPL_OP_OPTILING/CheckSupport 注册进入旧产品的 tiling so 并覆盖 built-in 路径。
set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")

# 本算子不对外提供 aclnn 接口（图模式专用），故 ACLNNTYPE 取 aclnn_exclude。
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE bn3d_training_reduce
        ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR}
        DISABLE_IN_OPP TRUE DEPENDENCIES norm_common)
