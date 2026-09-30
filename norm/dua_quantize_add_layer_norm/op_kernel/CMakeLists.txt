# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").

# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# 910 分支：kernel 入口位于 op_kernel/ 顶层（KERNEL_SRC 缺省取算子名 .cpp，不用 arch 子目录）
add_kernel_sources(
    COMPUTE_UNITS ascend910b ascend910_93
    AUTO_SYNC false
)

# Ascend950（arch35）分支：入口位于 arch35/ 子目录，与 op_host/CMakeLists 的 TILING_DIR 一一对应
add_kernel_sources(
    KERNEL_SRC arch35/dua_quantize_add_layer_norm.cpp
    COMPUTE_UNITS ascend950
    AUTO_SYNC false
)
