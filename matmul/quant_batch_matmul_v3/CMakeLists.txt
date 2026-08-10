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
    DEPENDENCIES trans_quant_param_v2 convert_weight_to_int4_pack quant_batch_matmul_v4 transpose_batch_mat_mul
)

add_kernel_sources(
    COMPUTE_UNITS ascend910b ascend910_93 kirinx90 kirin9030
    AUTO_SYNC false
)

add_kernel_sources(
    KERNEL_SRC arch35/quant_batch_matmul_v3.cpp
    COMPUTE_UNITS ascend950 ascend350
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush" "-DENABLE_CV_COMM_VIA_SSBUF=true"
)

add_kernel_sources(
    KERNEL_SRC arch35/quant_batch_matmul_v3.cpp
    COMPUTE_UNITS mc62
    OPTIONS "--cce-no-dcache-flush"
)

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(
        "ascend910B1"
        "-DDTYPE_X1=int8_t -DDTYPE_X2=int8_t -DORIG_DTYPE_X1=DT_INT8 -DORIG_DTYPE_X2=DT_INT8 -DFORMAT_X1=2 -DFORMAT_X2=2 -DDTYPE_Y=bfloat16_t -DORIG_DTYPE_Y=DT_BF16 -DDTYPE_SCALE=bfloat16_t -DORIG_DTYPE_SCALE=DT_BF16 -DDTYPE_BIAS=bfloat16_t -DORIG_DTYPE_BIAS=DT_BF16 -DDTYPE_PERTOKEN_SCALE=float -DORIG_DTYPE_PERTOKEN_SCALE=DT_FLOAT -D__CCE_AICORE__=220"
    )
    AddOpTestCase(
        "ascend950pr_9599"
        ""
        MULTI_KERNEL_TARGET
    )
endif()

if ((UT_TEST_ALL OR OP_GRAPH_UT) AND NOT UT_DONE)
    set(QUANT_BATCH_MATMUL_V3_TRANSPOSE_FUSION_PASS_CSV
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/ut/op_graph/test_quant_batch_matmul_v3_transpose_fusion_pass.csv)
    configure_file(
        ${QUANT_BATCH_MATMUL_V3_TRANSPOSE_FUSION_PASS_CSV}
        ${CMAKE_BINARY_DIR}/tests/ut/op_graph/test_quant_batch_matmul_v3_transpose_fusion_pass.csv
        COPYONLY)
    set(ZZ_MATMUL_T_QBMMV3_FUSION_PASS_CSV
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/ut/op_graph/test_z_z_matmul_t_qbmmv3_fusion_pass.csv)
    configure_file(
        ${ZZ_MATMUL_T_QBMMV3_FUSION_PASS_CSV}
        ${CMAKE_BINARY_DIR}/tests/ut/op_graph/test_z_z_matmul_t_qbmmv3_fusion_pass.csv
        COPYONLY)
    if(TARGET ${OP_GRAPH_MODULE_NAME}_cases_obj)
        target_compile_definitions(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
            QUANT_BATCH_MATMUL_V3_TRANSPOSE_FUSION_PASS_CSV_PATH="${QUANT_BATCH_MATMUL_V3_TRANSPOSE_FUSION_PASS_CSV}"
            ZZ_MATMUL_T_QBMMV3_FUSION_PASS_CSV_PATH="${ZZ_MATMUL_T_QBMMV3_FUSION_PASS_CSV}")
    endif()
endif()
