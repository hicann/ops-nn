/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "./test_sparse4to2quant_matmul_utils.h"
class SPARSE_SUITE_NAME_test : public testing::TestWithParam<Sparse4to2QuantMatmulTestParam> {};

TEST_P(SPARSE_SUITE_NAME_test, generalTest)
{
    Sparse4to2QuantMatmulTestParam param = GetParam();
    Sparse4to2QuantMatmulTestUtils::TestOneParamCase(param);
}

INSTANTIATE_TEST_CASE_P(SPARSECASE, SPARSE_SUITE_NAME_test,
                        testing::ValuesIn(Sparse4to2QuantMatmulTestUtils::GetParams("Ascend910B2", "sparse_bf16")));