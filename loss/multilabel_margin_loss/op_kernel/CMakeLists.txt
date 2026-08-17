# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 非 regbase (ascend910b/ascend910_93) -> 顶层 multilabel_margin_loss.cpp: A2 原有内核, 保持不变(红线: 不动 A2)。
# regbase (ascend950)                  -> arch35/multilabel_margin_loss.cpp: 新写 arch35 regbase 内核。

add_kernel_sources(
    KERNEL_SRC arch35/multilabel_margin_loss.cpp
    COMPUTE_UNITS ascend950
    AUTO_SYNC false
)
