# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")
add_all_modules_sources(OPTYPE median ACLNNTYPE aclnn_exclude
    DEPENDENCIES kth_value gather_v2 median_dim gather_elements
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR}
    DISABLE_IN_OPP TRUE)

# Keep graph_base loaded for the Host UT shared library under --as-needed linkers.
if(OP_HOST_UT AND TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
    target_link_libraries(${OP_TILING_MODULE_NAME}_cases_obj INTERFACE
        "-Wl,--no-as-needed,$<TARGET_FILE:graph_base>,--as-needed")
endif()
