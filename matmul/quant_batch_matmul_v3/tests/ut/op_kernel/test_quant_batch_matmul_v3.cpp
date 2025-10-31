/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "./test_quant_batch_matmul_v3_utils.h"

class QMMV3_SUITE_NAME_test : public testing::TestWithParam<QuantBatchMatmulV3TestParam> {};

TEST_P(QMMV3_SUITE_NAME_test, generalTest)
{
    QuantBatchMatmulV3TestParam param = GetParam();
    QuantBatchMatmulV3TestUtils::TestOneParamCase(param);
}

INSTANTIATE_TEST_CASE_P(QUANTMM910B, QMMV3_SUITE_NAME_test,
                        testing::ValuesIn(QuantBatchMatmulV3TestUtils::GetParams("Ascend910B2", "quant_batch_matmul_v3_bf16")));