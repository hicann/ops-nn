# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(SUPPORT_COMPUTE_UNIT "ascend950" "ascend350")
set(SUPPORT_TILING_DIR "arch35" "arch35")
add_all_modules_sources(OPTYPE unique_with_counts_and_sorting ACLNNTYPE aclnn_exclude
    DEPENDENCIES kth_value unique_consecutive
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR}
    DISABLE_IN_OPP TRUE)
if(TARGET ${OPHOST_NAME}_tiling_obj)
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE op_host/unique_with_counts_and_sorting_def.cpp)
endif()
