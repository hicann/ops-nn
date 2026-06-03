/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define DTYPE_X1 hifloat8_t
#define DTYPE_X2 hifloat8_t
#define DTYPE_Y float
#define ORIG_DTYPE_X1 DT_HIFLOAT8
#define ORIG_DTYPE_X2 DT_HIFLOAT8
#define ORIG_DTYPE_Y DT_FLOAT
#define __CCE_AICORE__ 310
#include "../test_quant_batch_matmul_inplace_add_utils.h"
#include "arch35/quant_batch_matmul_inplace_add_tiling_data.h"

using namespace std;

class QBMIIA_950_hif8 : public testing::TestWithParam<QuantBatchMatmulInplaceAddTestParam> {
protected:
    static void SetUpTestCase()
    {
        cout << "qbmmia_hif8_test SetUp\n" << endl;
        AscendC::SetKernelMode(KernelMode::MIX_MODE);
    }

    static void TearDownTestCase()
    {
        cout << "qbmmia_hif8_test TearDown\n" << endl;
        AscendC::SetKernelMode(KernelMode::MIX_MODE);
    }
};

TEST_P(QBMIIA_950_hif8, generalTest)
{
    QuantBatchMatmulInplaceAddTestParam param = GetParam();
    auto it = s_funcMapApt.find(param.tilingKey);
    ASSERT_NE(it, s_funcMapApt.end());
    ASSERT_NE(it->second, nullptr);
    ASSERT_GT(sizeof(QMMIA::QuantBatchMatmulInplaceAddTilingData), 0UL);
}

INSTANTIATE_TEST_CASE_P(QBMIIA950, QBMIIA_950_hif8,
                        testing::ValuesIn(QuantBatchMatmulInplaceAddTestUtils::GetParams("Ascend950", "HIF8_FLOW")));
