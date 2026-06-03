/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define DTYPE_X1 fp8_e4m3fn_t
#define DTYPE_X2 fp8_e5m2_t
#define DTYPE_Y float
#define ORIG_DTYPE_X1 DT_FLOAT8_E4M3FN
#define ORIG_DTYPE_X2 DT_FLOAT8_E5M2
#define ORIG_DTYPE_Y DT_FLOAT
#define __CCE_AICORE__ 310
#include "../test_quant_batch_matmul_inplace_add_utils.h"
#include "arch35/quant_batch_matmul_inplace_add_tiling_data.h"

using namespace std;

class QBMIIA_950_mx : public testing::TestWithParam<QuantBatchMatmulInplaceAddTestParam> {
protected:
    static void SetUpTestCase()
    {
        cout << "qbmmia_mx_test SetUp\n" << endl;
        AscendC::SetKernelMode(KernelMode::MIX_MODE);
    }

    static void TearDownTestCase()
    {
        cout << "qbmmia_mx_test TearDown\n" << endl;
        AscendC::SetKernelMode(KernelMode::MIX_MODE);
    }
};

TEST_P(QBMIIA_950_mx, mxTest)
{
    QuantBatchMatmulInplaceAddTestParam param = GetParam();
    QuantBatchMatmulInplaceAddTestUtils::TestOneParamCase950(param, s_funcMapApt);
}

INSTANTIATE_TEST_CASE_P(QBMIIA950, QBMIIA_950_mx,
                        testing::ValuesIn(QuantBatchMatmulInplaceAddTestUtils::GetParams("Ascend950", "MX_FLOW")));
