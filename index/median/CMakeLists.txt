# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(SUPPORT_COMPUTE_UNIT "ascend950" "ascend350")
set(SUPPORT_TILING_DIR "arch35" "arch35")
add_all_modules_sources(OPTYPE median ACLNNTYPE aclnn_exclude
    DEPENDENCIES kth_value gather_v2 median_dim gather_elements
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR}
    DISABLE_IN_OPP TRUE)

# Keep aclnn_nan_median.h available through the compatibility include in aclnn_median.h
# without introducing a reverse operator dependency on nan_median.
get_filename_component(NAN_MEDIAN_OP_API_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../nan_median/op_api REALPATH)
target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE ${NAN_MEDIAN_OP_API_DIR})

# A full build registers this header from nan_median itself. A median-only build
# needs to export it explicitly because aclnn_median.h keeps the compatibility include.
list(FIND ASCEND_COMPILE_OPS nan_median NAN_MEDIAN_OP_INDEX)
if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND NAN_MEDIAN_OP_INDEX EQUAL -1)
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE
        ${NAN_MEDIAN_OP_API_DIR}/aclnn_nan_median.h)
endif()

if((UT_TEST_ALL OR OP_API_UT) AND TARGET ${OP_API_MODULE_NAME}_cases_obj)
    target_include_directories(${OP_API_MODULE_NAME}_cases_obj PRIVATE ${NAN_MEDIAN_OP_API_DIR})
endif()
unset(NAN_MEDIAN_OP_INDEX)
unset(NAN_MEDIAN_OP_API_DIR)

# Keep graph_base loaded for the Host UT shared library under --as-needed linkers.
if(OP_HOST_UT AND TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
    target_link_libraries(${OP_TILING_MODULE_NAME}_cases_obj INTERFACE
        "-Wl,--no-as-needed,$<TARGET_FILE:graph_base>,--as-needed")
endif()
