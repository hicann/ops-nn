# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# A5(arch35) 使用 arch35 目录下的 scatter_nd_apt.cpp 作为 kernel 入口
add_kernel_sources(
    KERNEL_SRC arch35/scatter_nd_apt.cpp
    COMPUTE_UNITS ascend950 ascend350 mc62
)
# A6(arch92) 使用 arch92 目录下的入口，额外引入 arch92 的 SIMD DMA 模板
add_kernel_sources(
    KERNEL_SRC arch92/scatter_nd_apt.cpp
    COMPUTE_UNITS ascend960dt
)
