# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

cmake_minimum_required(VERSION 3.16)

if(UT_TEST_ALL OR OP_HOST_UT)
    # add op tiling ut common obj
    add_library(${OP_TILING_MODULE_NAME}_common_obj OBJECT)
    add_dependencies(${OP_TILING_MODULE_NAME}_common_obj json)

    # add op tiling ut common obj
    add_library(${OP_INFERSHAPE_MODULE_NAME}_common_obj OBJECT)

    file(GLOB OP_TILING_UT_COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/tiling_context_faker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tiling_parse_context_faker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_cube_util.cpp
    )
    target_sources(${OP_TILING_MODULE_NAME}_common_obj PRIVATE
        ${OP_TILING_UT_COMMON_SRC}
    )
    target_include_directories(${OP_TILING_MODULE_NAME}_common_obj PRIVATE
        ${JSON_INCLUDE}
        ${ASCEND_DIR}/pkg_inc
        ${ASCEND_DIR}/include/external
        ${ASCEND_DIR}/include/exe_graph
        ${ASCEND_DIR}/include/base/context_builder
        )
    target_link_libraries(${OP_TILING_MODULE_NAME}_common_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        c_sec
    )

    target_compile_options(${OP_TILING_MODULE_NAME}_common_obj PRIVATE
        -fno-access-control
    )

    target_compile_definitions(${OP_TILING_MODULE_NAME}_common_obj PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
        LOG_CPP
    )

    file(GLOB OP_INFERSHAPE_UT_COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/infershape_test_util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ut_op_util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ut_op_common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/infer_shape_context_faker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/infer_shaperange_context_faker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/infer_datatype_context_faker.cpp
    )
    target_sources(${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE
        ${OP_INFERSHAPE_UT_COMMON_SRC}
    )
    target_include_directories(${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE
        ${ASCEND_DIR}/pkg_inc
        ${ASCEND_DIR}/include/graph
        ${ASCEND_DIR}/include/external
        ${ASCEND_DIR}/include/exe_graph
        ${ASCEND_DIR}/include/base/context_builder
        ${CMAKE_CURRENT_SOURCE_DIR}/
    )

    target_compile_options(${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE
        -fno-access-control
    )

    target_compile_definitions(${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
        LOG_CPP
    )

    target_link_libraries(${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        c_sec
    )
endif()
