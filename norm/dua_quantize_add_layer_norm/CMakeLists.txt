# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").

# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
message(STATUS "=== Debug: start ops.norm.dua_quantize_add_layer_norm.CMakeLists.txt ")

# 设置算子定义时支持的芯片类型
set(SUPPORT_COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950")
# 设置每种芯片类型对应的 tiling 文件目录，与 SUPPORT_COMPUTE_UNIT 一一对应
# （arch22 = 910 原有实现；arch35 = Ascend950 实现）
set(SUPPORT_TILING_DIR "arch22" "arch22" "arch35")
# DEPENDENCIES 必留：910 分支的 op_kernel/dua_quantize_add_layer_norm_single_row_kernel.h
# 复用兄弟算子 op_kernel 下的 quantize_add_layer_norm_base.h（相对路径 ../quantize_add_layer_norm/）。
# 单算子编译场景下只有声明了依赖，构建目录才会一并 stage 该算子的 kernel 源，否则报
# "../quantize_add_layer_norm/quantize_add_layer_norm_base.h file not found"。
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE dua_quantize_add_layer_norm ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE DEPENDENCIES quantize_add_layer_norm)
