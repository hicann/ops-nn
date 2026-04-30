/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 
#include <iostream>
#include <gtest/gtest.h>
#include "smooth_l1_loss_grad_v2_tiling.h"
#include "../../../op_kernel/smooth_l1_loss_grad_v2_tiling_data.h"
#include "../../../op_kernel/smooth_l1_loss_grad_v2_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class SmoothL1LossGradV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "SmoothL1LossGradV2Tiling SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "SmoothL1LossGradV2Tiling TearDown " << endl;
    }
};

TEST_F(SmoothL1LossGradV2Tiling, ascend9101_test_tiling_fp16_001)
{
    optiling::SmoothL1LossGradV2CompileInfo compileInfo = {64, 262144, true, 1.0f, 2};
    gert::TilingContextPara tilingContextPara(
        "SmoothL1LossGradV2",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // predict
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // label
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dout (none 模式)
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // gradient
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SmoothL1LossGradV2Tiling, ascend9101_test_tiling_bf16_002)
{
    optiling::SmoothL1LossGradV2CompileInfo compileInfo = {64, 262144, true, 1.0f, 2};
    gert::TilingContextPara tilingContextPara(
        "SmoothL1LossGradV2",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SmoothL1LossGradV2Tiling, ascend9101_test_tiling_fp32_003)
{
    optiling::SmoothL1LossGradV2CompileInfo compileInfo = {64, 262144, true, 1.0f, 2};
    gert::TilingContextPara tilingContextPara(
        "SmoothL1LossGradV2",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}