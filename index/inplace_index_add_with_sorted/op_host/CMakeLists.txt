# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

set(SUPPORT_COMPUTE_UNIT "ascend910b" "ascend910_93" "kirinx90" "kirin9030" "ascend950")
set(SUPPORT_TILING_DIR "arch32" "arch32" "arch32" "arch32" "arch35")
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE inplace_index_add_with_sorted ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR}
    DISABLE_IN_OPP TRUE)
