# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_kernel_sources(
    COMPUTE_UNITS ascend310p ascend910b ascend910_93 kirinx90
    SIMPLIFIED_KEY None
    AUTO_SYNC false
)

add_kernel_sources(
    COMPUTE_UNITS ascend950
    SIMPLIFIED_KEY None
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush"
)

add_kernel_sources(
    COMPUTE_UNITS kirin9030
    AUTO_SYNC false
)
