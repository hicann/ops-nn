# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# arch22 SoC -> 顶层 mse_loss_grad_v2.cpp; ascend950 -> arch35/mse_loss_grad_v2.cpp
# arch22 内核为 TQue 型、在 auto_sync=true(框架默认)下开发并验证;此处不能关自动同步,
# 否则缺编译器自动插入的同步 -> 数据竞争 -> arch22 精度回退。故 arch22 段不设 AUTO_SYNC(取默认 true)。
add_kernel_sources(
    COMPUTE_UNITS ascend910b ascend910_93 ascend310p
)
add_kernel_sources(
    KERNEL_SRC arch35/mse_loss_grad_v2.cpp
    COMPUTE_UNITS ascend950
    AUTO_SYNC false
)
