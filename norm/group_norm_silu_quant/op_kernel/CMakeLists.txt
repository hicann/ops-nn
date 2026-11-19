# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# A2 SoC -> 顶层 group_norm_silu_quant.cpp; dav_3510 -> arch35/group_norm_silu_quant.cpp
# A2 内核为 TQue 型、在 auto_sync=true(框架默认)下开发并验证;此处不能关自动同步,
# 否则缺编译器自动插入的同步 -> 数据竞争 -> A2 精度回退。故 A2 段不设 AUTO_SYNC(取默认 true)。
add_kernel_sources(
    COMPUTE_UNITS ascend910b ascend910_93
)
add_kernel_sources(
    KERNEL_SRC arch35/group_norm_silu_quant.cpp
    COMPUTE_UNITS dav_3510
    AUTO_SYNC false
)
