/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <iostream>
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class BatchNormalizationGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BatchNormalizationGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BatchNormalizationGradTiling TearDown" << std::endl;
    }
};

TEST_F(BatchNormalizationGradTiling, batch_normalization_grad_float32_success)
{
    struct BatchNormalizationGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BatchNormalizationGrad",
        {
            {{{16, 16, 64, 2}, {16, 16, 64, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},    // grad_output
            {{{16, 16, 64, 2}, {16, 16, 64, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},    // input
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // weight
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // bias
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // save_mean
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // save_invstd
        },
        {
            {{{16, 16, 64, 2}, {16, 16, 64, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},    // grad_input
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // grad_weight
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // grad_bias
        },
        {
            /* attrs: */
        },
        &compileInfo); 
    uint64_t expectTilingKey = 0;
    string expectTilingData = "68719476752 8796093022336 69877104640 1 549755813888 549755813889 68719476737 68719476737 3974362538702798976 17179869216 8 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(BatchNormalizationGradTiling, batch_normalization_grad_float16_success)
{
    struct BatchNormalizationGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BatchNormalizationGrad",
        {
            {{{16, 16, 64, 2}, {16, 16, 64, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 16, 64, 2}, {16, 16, 64, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 16, 64, 2}, {16, 16, 64, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            /* attrs */
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "68719476752 8796093022336 69877104640 1 549755813888 549755813889 68719476737 68719476737 3974362538702798976 8589934624 16 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(BatchNormalizationGradTiling, batch_normalization_grad_bfloat16_success)
{
    struct BatchNormalizationGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BatchNormalizationGrad",
        {
            {{{16, 16, 64, 2}, {16, 16, 64, 2}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16, 16, 64, 2}, {16, 16, 64, 2}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{16, 16, 64, 2}, {16, 16, 64, 2}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            /* attrs */
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "68719476752 8796093022336 69877104640 1 549755813888 549755813889 68719476737 68719476737 3974362538702798976 8589934624 16 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
