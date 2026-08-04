# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# ⚠️ 本算子【不提供 aclnn 单算子接口】，只走 GE 图模式下发。
#    依据：CANN 9.1.0 未定义 aclnnSgd（include/aclnnop/ 下只有 aclnn_fused_sgd.h，
#    libopapi.so 无 aclnnSgd* 符号），canndev 全仓亦无 aclnn_sgd —— SGD 在上游本就是
#    纯图模式算子。故本仓不再引入 op_api/ 目录。
#
#    add_modules_sources 仍放在【算子根 CMakeLists】而非 op_host/CMakeLists.txt：
#    func.cmake:391 仅在 <op>/op_host/CMakeLists.txt 不存在时 add_subdirectory(<op>)，
#    二者【只能存在其一】，否则 add_modules_sources 被调用两次。
#    DIR 传算子根路径是安全的：func.cmake:513 会在 SOURCE_DIR 不以 /op_host 结尾时自动补上，
#    OPDEF_SRCS 仍正确 glob 到 <op>/op_host/sgd_def*.cpp。

# 设置算子定义时支持的芯片类型
set(SUPPORT_COMPUTE_UNIT "ascend950")
# 设置每种芯片类型对应的 tiling 文件目录，即采用 op_host 目录下哪个文件夹下的 tiling 文件编译
set(SUPPORT_TILING_DIR "arch35")

# ACLNNTYPE aclnn_exclude：把 sgd_def.cpp 归入 aclnnExc 桶 —— 该桶【不触发 aclnn 接口自动生成】
# （opbuild.cmake:57 分支只收集手写的 op_api/aclnn_*.h，本算子已无该目录，故不产出任何 aclnn 头）。
# 算子定义本身照常注册，GE 图通路不受影响。
# ⛔ 不得改成 aclnn / aclnn_inner —— 那会让构建系统重新生成 aclnnSgd 接口。
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE sgd ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE)
