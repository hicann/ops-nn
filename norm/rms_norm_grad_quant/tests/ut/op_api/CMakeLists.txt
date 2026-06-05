# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(UT_TEST_ALL OR OP_API_UT)
    add_modules_ut_sources(HOSTNAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})

    # aclnn_exclude算子需手动将op_api源码加入opapi库
    # 从 tests/ut/op_api 向上定位算子根目录
    get_filename_component(OP_UT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(OP_TESTS_DIR ${OP_UT_DIR} DIRECTORY)
    get_filename_component(OP_ROOT_DIR ${OP_TESTS_DIR} DIRECTORY)
    set(OP_API_SRC_DIR ${OP_ROOT_DIR}/op_api)
    file(GLOB OP_API_SRCS ${OP_API_SRC_DIR}/*.cpp)
    if(OP_API_SRCS AND TARGET ${OPHOST_NAME}_opapi_obj)
        target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OP_API_SRCS})
        target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE ${OP_API_SRC_DIR})
    endif()
endif()
