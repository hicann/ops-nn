# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# A5(arch35) 使用 arch35 目录下的 scatter_nd_add.cpp 作为 kernel 入口
# GM 写均走原子路径（SetAtomicAdd/asc_atomic_add），自管同步与 cache 一致性，
# 故在下方 add_kernel_sources 中通过 AUTO_SYNC false 与 OPTIONS 统一关闭
# 编译器自动插入的同步与 kernel 末尾全量 dcache flush
add_kernel_sources(
    KERNEL_SRC arch35/scatter_nd_add.cpp
    COMPUTE_UNITS ascend950 mc62 ascend350
    IMPL_MODE high_precision,support_out_of_bound_index,optional
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush"
)
# A6(arch92) 使用 arch92 目录下的入口（复用 arch35 原子头 + arch92 simd_dma 模板）
add_kernel_sources(
    KERNEL_SRC arch92/scatter_nd_add.cpp
    COMPUTE_UNITS ascend960dt
    IMPL_MODE high_precision,support_out_of_bound_index,optional
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush"
)
