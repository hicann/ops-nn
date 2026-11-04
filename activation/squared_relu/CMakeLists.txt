# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/

# aclnn_exclude：使用 op_api/ 下手写的 aclnn 两段式接口（支持非连续 out，经 ViewCopy 回写），
# opbuild 不再为本算子生成 wrapper；op_api 位于算子根目录，DIR 传算子根路径。
# 子目录（op_graph/tests）由 add_category_subdirectory 的 add_op_subdirectory 宏统一收编，此处不再迭代。
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE squared_relu ACLNNTYPE aclnn_exclude DISABLE_IN_OPP TRUE)
