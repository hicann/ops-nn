/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define DTYPE_X1 int8_t
#define DTYPE_X2 int8_t
#define ORIG_DTYPE_X1 DT_INT8
#define ORIG_DTYPE_X2 DT_INT8
#define DTYPE_Y half
#define ORIG_DTYPE_Y DT_FLOAT16
#define DTYPE_SCALE uint64_t
#define ORIG_DTYPE_SCALE DT_BF16
#define __CCE_AICORE__ 310
#include "../test_quant_batch_matmul_v3_utils.h"
#include "arch35/quant_batch_matmul_v3_tiling_data.h"


using namespace std;
class QBMMV3_910_95_general : public testing::TestWithParam<QuantBatchMatmulV3TestParam>
{
protected:
    static void SetUpTestCase()
    {
        cout << "qbmm_test SetUp\n" << endl;
        AscendC::SetKernelMode(KernelMode::MIX_MODE);
    }

    static void TearDownTestCase()
    {
        cout << "qbmm_test TearDown\n" << endl;
        AscendC::SetKernelMode(KernelMode::MIX_MODE);
    }
};

TEST_P(QBMMV3_910_95_general, generalTest)
{
    QuantBatchMatmulV3TestParam param = GetParam();
    QuantBatchMatmulV3TestUtils::TestOneParamCase910_95(param, s_funcMapApt);
}

INSTANTIATE_TEST_CASE_P(QBMM910_95, QBMMV3_910_95_general,
                        testing::ValuesIn(QuantBatchMatmulV3TestUtils::GetParams("Ascend910_95", "BF16_FLOW")));