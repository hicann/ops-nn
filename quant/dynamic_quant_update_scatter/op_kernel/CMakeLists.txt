# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# A2 / kirin 根 kernel(保持原路径不变,自动收集本目录下的 A2 kernel 源)
add_kernel_sources(
    COMPUTE_UNITS ascend910b ascend910_93 kirinx90 kirin9030
)
# arch35 (Ascend950) native RegBase kernel.
add_kernel_sources(
    KERNEL_SRC arch35/dynamic_quant_update_scatter.cpp
    COMPUTE_UNITS ascend950
    AUTO_SYNC false
)
