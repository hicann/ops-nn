# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# A5(arch35) 使用 arch35 目录下的 scatter_update.cpp 作为 kernel 入口
# 显式声明 impl_mode，与 scripts/kernel/binary_config/all_ops_impl_mode.ini 中 ScatterUpdate 的配置保持一致；
# 否则 add_kernel_sources 默认值是 high_performance,optional，会覆盖兜底，导致 high_precision 与
# support_out_of_bound_index 两套二进制在 ascend950/ascend350 上丢失（功能回退）。
add_kernel_sources(
    KERNEL_SRC arch35/scatter_update.cpp
    COMPUTE_UNITS ascend950 ascend350
    IMPL_MODE high_performance,high_precision,support_out_of_bound_index
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush"
)
# A6(arch92) 使用 arch92 目录下的入口，额外引入 arch92 的 SIMD_SCATTER_DMA 模板
add_kernel_sources(
    KERNEL_SRC arch92/scatter_update.cpp
    COMPUTE_UNITS ascend960dt
    IMPL_MODE high_performance,high_precision,support_out_of_bound_index
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush"
)
