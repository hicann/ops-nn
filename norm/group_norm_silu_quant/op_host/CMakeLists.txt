#
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/
message(STATUS "=== Debug: start ops.norm.group_norm_silu_quant.op_host.CMakeLists.txt ")

# A2(arch22) tiling 与 arch35(dav_3510) regbase tiling 按 SoC 路由分开编译, 避免双 IMPL_OP_OPTILING 冲突。
set(SUPPORT_COMPUTE_UNIT "ascend910b" "ascend910_93" "dav_3510")
set(SUPPORT_TILING_DIR "arch22" "arch22" "arch35")
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE group_norm_silu_quant ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE)
