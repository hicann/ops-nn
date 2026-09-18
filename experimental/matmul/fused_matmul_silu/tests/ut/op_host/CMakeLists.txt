# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(HOSTNAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    if(TARGET ${OP_TILING_MODULE_NAME}_cases_obj AND NOT TARGET ${OP_TILING_MODULE_NAME}_common_obj)
        target_sources(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
            ${PROJECT_SOURCE_DIR}/tests/ut/common/tiling_context_faker.cpp
            ${PROJECT_SOURCE_DIR}/tests/ut/common/tiling_parse_context_faker.cpp
            ${PROJECT_SOURCE_DIR}/tests/ut/common/test_cube_util.cpp
            ${PROJECT_SOURCE_DIR}/tests/ut/common/legacy_common_manager_stub.cpp
        )
    endif()
    set(TILING_CASE_EXECUTOR_IN_COMMON FALSE)
    if(TARGET ${OP_TILING_MODULE_NAME}_common_obj)
        get_target_property(TILING_COMMON_SOURCES ${OP_TILING_MODULE_NAME}_common_obj SOURCES)
        list(FIND TILING_COMMON_SOURCES
            "${PROJECT_SOURCE_DIR}/tests/ut/common/tiling_case_executor.cpp"
            TILING_CASE_EXECUTOR_INDEX)
        if(NOT TILING_CASE_EXECUTOR_INDEX EQUAL -1)
            set(TILING_CASE_EXECUTOR_IN_COMMON TRUE)
        endif()
    endif()
    if(TARGET ${OP_TILING_MODULE_NAME}_cases_obj AND NOT TILING_CASE_EXECUTOR_IN_COMMON)
        target_sources(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
            ${PROJECT_SOURCE_DIR}/tests/ut/common/tiling_case_executor.cpp
        )
        target_compile_definitions(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
            BUILD_SOC_VERSION=${ASCEND_COMPUTE_UNIT}
        )
    endif()
    add_modules_ut_sources(HOSTNAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
