# -----------------------------------------------------------------------------
 # Copyright (c) 2026 Huawei Technologies Co., Ltd.
 # This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 # CANN Open Software License Agreement Version 2.0 (the "License").
 # Please refer to the License for details. You may not use this file except in compliance with the License.
 # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 # See LICENSE in the root of the software repository for the full text of the License.
 # -----------------------------------------------------------------------------

message(STATUS "=== Debug: start ops.matmul.quant_batch_matmul_v4.tests.ut.op_graph.CMakeLists.txt ")
file(GLOB CURRENT_DIR RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(UT_TEST_ALL OR OP_GRAPH_UT)
    add_modules_ut_sources(HOSTNAME ${OP_GRAPH_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_CSV
        ${CMAKE_CURRENT_SOURCE_DIR}/test_quant_batch_matmul_v4_to_v3_fusion_pass.csv)
    configure_file(
        ${QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_CSV}
        ${CMAKE_BINARY_DIR}/tests/ut/op_graph/test_quant_batch_matmul_v4_to_v3_fusion_pass.csv
        COPYONLY)
    if(TARGET ${OP_GRAPH_MODULE_NAME}_cases_obj)
        target_compile_definitions(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
            QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_CSV_PATH="${QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_CSV}")
    endif()
endif()
