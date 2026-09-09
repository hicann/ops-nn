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
    DEPENDENCIES batch_mat_mul_v3 mat_mul_v3 quant_batch_matmul_v3
)

add_kernel_sources(
    AUTO_SYNC false
)

add_kernel_sources(
    KERNEL_SRC arch35/transpose_quant_batch_mat_mul.cpp
    COMPUTE_UNITS ascend950 ascend350
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush" "-DENABLE_CV_COMM_VIA_SSBUF=true"
)

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    # One target per test file under tests/ut/op_kernel/arch35; each file self-declares its DTYPE macros
    # (FP8 mix path, HIFP8 fixpipe-quant path and MXFP4 MX path need separate compilation units).
    AddOpTestCase(
        "ascend950pr_9599"
        "-DKFC_L1_RESERVER_SIZE=0 -D__fp8e5m2=fp8_e5m2_t -D__fp8e4m3=fp8_e4m3fn_t -D__fp4e2m1x2=int8_t -D__fp8e8m0=fp8_e8m0_t"
        mat_mul_v3 batch_mat_mul_v3 quant_batch_matmul_v3
        MULTI_KERNEL_TARGET
    )
    set(STUB_DIR ${CMAKE_BINARY_DIR}/tbe/ascendc/transpose_quant_batch_mat_mul/arch35)
    add_custom_target(tqbmm_stub_blaze_h
        COMMAND bash -c [=[printf '%s\n' '// stub for UT host compile' 'struct BatchMatMulV3TilingData;' 'template <class A_TYPE, class B_TYPE, class SCALE_TYPE, class C_TYPE, class BIAS_TYPE, class aLayout, class bLayout, class cLayout, uint64_t FULL_LOAD_MODE = 0, uint64_t PERM_X1 = 0, uint64_t NON_CONTIGUOUS_TYPE = 0>' '__aicore__ inline void TqbmmMxTensorApiKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR bias, GM_ADDR perTokenScale, GM_ADDR cGM, const BatchMatMulV3TilingData& tilingData) {}' > "$1"]=] stub_gen ${STUB_DIR}/transpose_quant_batch_mat_mul_mx.h
        VERBATIM
        DEPENDS ascendc_kernel_ut_src_copy_transpose_quant_batch_mat_mul
        COMMENT "Generating blaze stub headers for UT"
    )
    foreach(_tqbmm_ut_target
        opkernel_test_transpose_quant_batch_mat_mul
        opkernel_test_transpose_quant_batch_mat_mul_hifp8
        opkernel_test_transpose_quant_batch_mat_mul_mxfp4)
        if (TARGET ${_tqbmm_ut_target})
            add_dependencies(${_tqbmm_ut_target} tqbmm_stub_blaze_h)
        endif()
    endforeach()
endif()
