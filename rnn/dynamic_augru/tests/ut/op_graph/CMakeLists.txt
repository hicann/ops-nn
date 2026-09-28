# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# The generic collector only matches test_*_pass.cpp; register inference tests explicitly.
if(NOT TARGET ${OP_GRAPH_MODULE_NAME}_cases_obj)
    add_op_graph_ut_modules(${OP_GRAPH_MODULE_NAME})
endif()
target_sources(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dynamic_augru_graph_infer.cpp)

# The common UT registry loads only op_host. Load the graph registration from
# its own DSO, matching the production registry's library loading contract.
add_library(dynamic_augru_graph_registry_ut SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_graph/dynamic_augru_graph_infer.cpp)
target_link_libraries(dynamic_augru_graph_registry_ut PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> register graph exe_graph)
target_include_directories(dynamic_augru_graph_registry_ut PRIVATE
    ${ASCEND_DIR}/include ${ASCEND_DIR}/include/external)
target_compile_definitions(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
    DYNAMIC_AUGRU_GRAPH_REGISTRY_SO="$<TARGET_FILE:dynamic_augru_graph_registry_ut>")
add_dependencies(${OP_GRAPH_MODULE_NAME}_cases_obj dynamic_augru_graph_registry_ut)
