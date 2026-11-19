# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
if(UT_TEST_ALL OR OP_GRAPH_UT)
    add_modules_ut_sources(HOSTNAME ${OP_GRAPH_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})

    add_library(embedding_dense_grad_graph_registry_ut SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_graph/embedding_dense_grad_graph_infer.cpp)
    target_link_libraries(embedding_dense_grad_graph_registry_ut PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        $<BUILD_INTERFACE:dlog_headers>
        register graph exe_graph)
    target_include_directories(embedding_dense_grad_graph_registry_ut PRIVATE
        ${OPBASE_INC_DIRS}
        ${PROJECT_SOURCE_DIR}/common/inc
        ${ASCEND_DIR}/include
        ${ASCEND_DIR}/include/external
        ${ASCEND_DIR}/include/exe_graph)
    target_compile_definitions(embedding_dense_grad_graph_registry_ut PRIVATE
        OPS_UTILS_LOG_SUB_MOD_NAME="GRAPH_PLUGIN" LOG_CPP)
    target_compile_definitions(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
        EMBEDDING_DENSE_GRAD_GRAPH_REGISTRY_SO="$<TARGET_FILE:embedding_dense_grad_graph_registry_ut>")
    add_dependencies(${OP_GRAPH_MODULE_NAME}_cases_obj embedding_dense_grad_graph_registry_ut)
endif()
