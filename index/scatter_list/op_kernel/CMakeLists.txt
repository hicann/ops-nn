# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 仅为 A5 注册 arch35 的 SIMT 实现。非 A5 不声明, 沿用构建默认(KERNEL_SRC 缺省为共享入口、
# auto_sync 默认开), 与本改动前一致; 显式补一块会把非 A5 的 auto_sync 改成 false。
add_kernel_sources(
    KERNEL_SRC arch35/scatter_list.cpp
    COMPUTE_UNITS ascend950
    AUTO_SYNC false
)
