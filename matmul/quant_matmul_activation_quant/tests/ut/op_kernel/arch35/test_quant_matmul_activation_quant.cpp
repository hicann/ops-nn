/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Compile this source for each CSV dtype/layout group. The production entry uses DTYPE
// macros, so each specialization needs its own translation unit and entry name.
#if defined(QMMAQ_UT_E4M3_WEIGHT_NZ)
#define DTYPE_X1 fp8_e4m3fn_t
#define DTYPE_X2 fp8_e4m3fn_t
#define DTYPE_Y fp8_e4m3fn_t
#define ORIG_DTYPE_X1 DT_FLOAT8_E4M3FN
#define ORIG_DTYPE_X2 DT_FLOAT8_E4M3FN
#define ORIG_DTYPE_Y DT_FLOAT8_E4M3FN
#define quant_matmul_activation_quant quant_matmul_activation_quant_e4m3_weight_nz_ut
#define QMMAQ_KERNEL_SUITE QuantMatmulActivationQuantE4M3WeightNzKernel
#define QMMAQ_CSV_SUITE QuantMatmulActivationQuantE4M3WeightNzKernelCsv
#define QMMAQ_CSV_TARGET "QMMAQ_E4M3_WEIGHT_NZ"
#elif defined(QMMAQ_UT_E5M2_GELU)
#define DTYPE_X1 fp8_e5m2_t
#define DTYPE_X2 fp8_e5m2_t
#define DTYPE_Y fp8_e5m2_t
#define ORIG_DTYPE_X1 DT_FLOAT8_E5M2
#define ORIG_DTYPE_X2 DT_FLOAT8_E5M2
#define ORIG_DTYPE_Y DT_FLOAT8_E5M2
#define quant_matmul_activation_quant quant_matmul_activation_quant_e5m2_gelu_ut
#define QMMAQ_KERNEL_SUITE QuantMatmulActivationQuantE5M2GeluKernel
#define QMMAQ_CSV_SUITE QuantMatmulActivationQuantE5M2GeluKernelCsv
#define QMMAQ_CSV_TARGET "QMMAQ_E5M2_GELU"
#elif defined(QMMAQ_UT_E5M2)
#define DTYPE_X1 fp8_e5m2_t
#define DTYPE_X2 fp8_e4m3fn_t
#define DTYPE_Y fp8_e5m2_t
#define ORIG_DTYPE_X1 DT_FLOAT8_E5M2
#define ORIG_DTYPE_X2 DT_FLOAT8_E4M3FN
#define ORIG_DTYPE_Y DT_FLOAT8_E5M2
#define quant_matmul_activation_quant quant_matmul_activation_quant_e5m2_ut
#define QMMAQ_KERNEL_SUITE QuantMatmulActivationQuantE5M2Kernel
#define QMMAQ_CSV_SUITE QuantMatmulActivationQuantE5M2KernelCsv
#define QMMAQ_CSV_TARGET "QMMAQ_E5M2"
#else
#define DTYPE_X1 fp8_e4m3fn_t
#define DTYPE_X2 fp8_e5m2_t
#define DTYPE_Y fp8_e4m3fn_t
#define ORIG_DTYPE_X1 DT_FLOAT8_E4M3FN
#define ORIG_DTYPE_X2 DT_FLOAT8_E5M2
#define ORIG_DTYPE_Y DT_FLOAT8_E4M3FN
#define quant_matmul_activation_quant quant_matmul_activation_quant_e4m3_ut
#define QMMAQ_KERNEL_SUITE QuantMatmulActivationQuantE4M3Kernel
#define QMMAQ_CSV_SUITE QuantMatmulActivationQuantE4M3KernelCsv
#define QMMAQ_CSV_TARGET "QMMAQ_E4M3"
#endif

#ifdef __CCE_KT_TEST__
#ifndef __CCE_AICORE__
#define __CCE_AICORE__ 310
#endif
#include "quant_matmul_activation_quant_cpu_debug_stub.h"
#if defined(QMMAQ_UT_E4M3_WEIGHT_NZ)
#if !defined(FORMAT_FRACTAL_NZ)
#define FORMAT_FRACTAL_NZ 0x7FFFFFFF
#define QMMAQ_UT_LOCAL_FORMAT_FRACTAL_NZ
#endif
#if !defined(FORMAT_ND)
#define FORMAT_ND 0
#define QMMAQ_UT_LOCAL_FORMAT_ND
#endif
#define FORMAT_X2 FORMAT_FRACTAL_NZ
#else
#define FORMAT_X2 FORMAT_ND
#endif
#include "arch35/quant_matmul_activation_quant.cpp"
#undef FORMAT_X2
#if defined(QMMAQ_UT_LOCAL_FORMAT_FRACTAL_NZ)
#undef FORMAT_FRACTAL_NZ
#undef QMMAQ_UT_LOCAL_FORMAT_FRACTAL_NZ
#endif
#if defined(QMMAQ_UT_LOCAL_FORMAT_ND)
#undef FORMAT_ND
#undef QMMAQ_UT_LOCAL_FORMAT_ND
#endif
#undef make_mem_ptr
#undef asc_get_phy_buf_addr
#endif
#include "gtest/gtest.h"
#include "../test_quant_matmul_activation_quant_utils.h"

// Expand the dtype-specific suite names before GoogleTest stringifies them.
#define QMMAQ_TEST(suite, name) TEST(suite, name)
#define QMMAQ_TEST_P(suite, name) TEST_P(suite, name)
#define QMMAQ_INSTANTIATE_TEST_SUITE_P(...) INSTANTIATE_TEST_SUITE_P(__VA_ARGS__)

namespace {
const QuantMatmulActivationQuantKernelCsvLoadResult& GetKernelCases()
{
    static const auto result = QuantMatmulActivationQuantKernelTestUtils::GetParams("Ascend950", QMMAQ_CSV_TARGET);
    return result;
}

class QMMAQ_KERNEL_SUITE : public testing::TestWithParam<QuantMatmulActivationQuantKernelTestParam> {
protected:
    static void SetUpTestSuite()
    {
#ifdef __CCE_KT_TEST__
        AscendC::SetKernelMode(KernelMode::MIX_MODE);
#endif
    }
};

QMMAQ_TEST(QMMAQ_CSV_SUITE, LoadsAscend950Cases)
{
    const auto& result = GetKernelCases();
    for (const auto& error : result.errors) {
        ADD_FAILURE() << error;
    }
    EXPECT_FALSE(result.params.empty());
}

QMMAQ_TEST_P(QMMAQ_KERNEL_SUITE, ComputesValuesScalesAndPreservesGuards)
{
#ifdef __CCE_KT_TEST__
    QuantMatmulActivationQuantKernelTestUtils::TestOneParamCase950(GetParam());
#else
    GTEST_SKIP() << "Kernel CPU simulator is unavailable";
#endif
}

QMMAQ_INSTANTIATE_TEST_SUITE_P(Ascend950, QMMAQ_KERNEL_SUITE, testing::ValuesIn(GetKernelCases().params),
                               [](const testing::TestParamInfo<QuantMatmulActivationQuantKernelTestParam>& info) {
                                   return info.param.caseName;
                               });
} // namespace

#undef QMMAQ_TEST
#undef QMMAQ_TEST_P
#undef QMMAQ_INSTANTIATE_TEST_SUITE_P
#undef QMMAQ_KERNEL_SUITE
#undef QMMAQ_CSV_SUITE
#undef QMMAQ_CSV_TARGET
#undef quant_matmul_activation_quant
#undef DTYPE_X1
#undef DTYPE_X2
#undef DTYPE_Y
#undef ORIG_DTYPE_X1
#undef ORIG_DTYPE_X2
#undef ORIG_DTYPE_Y
