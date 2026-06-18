# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# NOTE: Portions of this code were AI-generated and have been
# technically reviewed for functional accuracy and security

# 设置算子定义时支持的芯片类型（本任务：ascend910b 端侧原生 AscendC 路径）
set(SUPPORT_COMPUTE_UNIT "ascend910b")
# tiling 文件目录：flat op_host（对标相邻 apply_adamax/apply_adadelta）
set(SUPPORT_TILING_DIR "${CMAKE_CURRENT_SOURCE_DIR}/op_host")
# 手写 aclnn L2/L0 接口：ACLNNTYPE aclnn_exclude 表示公共头由 op_api/aclnn_*.h 提供；
# DISABLE_IN_OPP 不并入已部署 OPP。
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE apply_adam_d ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE)
