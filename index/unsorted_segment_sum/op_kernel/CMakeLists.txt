# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# ascend950/ascend350 使用 op_kernel 根目录的 unsorted_segment_sum_apt.cpp（缺省由 opFile.value 定位）；
# ascend960dt/ascend960pr 使用 arch92 目录下的 kernel 入口（tiling 与 950 共用 op_host/arch35），
# tiling data 结构共享 op_kernel/arch35/unsorted_segment_sum_tiling_data.h，模板参数声明见
# arch92/unsorted_segment_sum_tiling_key.h（与 arch35 编码一致，不含 small-innerdim 模板）。
add_kernel_sources(
    KERNEL_SRC arch92/unsorted_segment_sum_apt
    COMPUTE_UNITS ascend960dt ascend960pr
    AUTO_SYNC false
)
