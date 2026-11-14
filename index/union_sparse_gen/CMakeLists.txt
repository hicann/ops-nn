# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(SUPPORT_COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950")
set(SUPPORT_TILING_DIR "arch22" "arch22" "arch35")
add_all_modules_sources(OPTYPE union_sparse_gen ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR})

add_kernel_sources(
    KERNEL_SRC union_sparse_gen.cpp
    COMPUTE_UNITS ${SUPPORT_COMPUTE_UNIT}
    AUTO_SYNC false)
