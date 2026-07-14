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

class HuberLossGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HuberLossGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "HuberLossGradTiling TearDown" << std::endl; }
};

TEST_F(HuberLossGradTiling, huber_loss_grad_float32_success)
{
    struct HuberLossGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("HuberLossGrad",
                                              {
                                                  {{{256, 32}, {256, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}, // predictions
                                                  {{{256, 32}, {256, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}, // targets
                                              },
                                              {
                                                  {{{256, 32}, {256, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}, // grad_output
                                              },
                                              {
                                                  /* attrs */
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "35218731835392 8589934594 14018773259072 3272 35184372088832 4575657221408483072 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(HuberLossGradTiling, huber_loss_grad_float16_success)
{
    struct HuberLossGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "HuberLossGrad",
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // predictions
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // targets
        },
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // grad_output
        },
        {
            /* attrs */
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "35253091573760 4294967297 35184372098528 8208 35184372088833 4575657221408482112 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(HuberLossGradTiling, huber_loss_grad_bfloat16_success)
{
    struct HuberLossGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("HuberLossGrad",
                                              {
                                                  {{{256, 32}, {256, 32}}, ge::DT_BF16, ge::FORMAT_ND}, // predictions
                                                  {{{256, 32}, {256, 32}}, ge::DT_BF16, ge::FORMAT_ND}, // targets
                                              },
                                              {
                                                  {{{256, 32}, {256, 32}}, ge::DT_BF16, ge::FORMAT_ND}, // grad_output
                                              },
                                              {
                                                  /* attrs */
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "35253091573760 8589934594 14018773259072 3280 35184372088859 4575657221408483072 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
