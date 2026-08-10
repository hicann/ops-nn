# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_all_modules_sources(
    ACLNNTYPE aclnn_exclude
    DEPENDENCIES quant_batch_matmul_v3 weight_quant_batch_matmul_v2
)

add_kernel_sources(
    AUTO_SYNC false
)

add_kernel_sources(
    KERNEL_SRC arch35/quant_batch_matmul_v4.cpp
    COMPUTE_UNITS ascend950 ascend350
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush"
)

add_kernel_sources(
    KERNEL_SRC arch35/quant_batch_matmul_v4.cpp
    COMPUTE_UNITS mc62
)

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(
        "ascend910B1"
        "-DKFC_L1_RESERVER_SIZE=0 -DDTYPE_X1=int8_t -DDTYPE_X2=int8_t -DDTYPE_BIAS=float32 -DDTYPE_Y=bfloat16_t -DFORMAT_X1=2 -DFORMAT_X2=2"
        quant_batch_matmul_v3
    )
endif()

if ((UT_TEST_ALL OR OP_GRAPH_UT) AND NOT UT_DONE)
    set(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_CSV
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/ut/op_graph/test_quant_batch_matmul_v4_to_v3_fusion_pass.csv)
    configure_file(
        ${QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_CSV}
        ${CMAKE_BINARY_DIR}/tests/ut/op_graph/test_quant_batch_matmul_v4_to_v3_fusion_pass.csv
        COPYONLY)
    if(TARGET ${OP_GRAPH_MODULE_NAME}_cases_obj)
        target_compile_definitions(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
            QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_CSV_PATH="${QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_CSV}")
    endif()
endif()
