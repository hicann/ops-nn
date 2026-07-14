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

class LayerNormalizationGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LayerNormalizationGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "LayerNormalizationGradTiling TearDown" << std::endl; }
};

TEST_F(LayerNormalizationGradTiling, layer_normalization_grad_float32_success)
{
    struct LayerNormalizationGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("LayerNormalizationGrad",
                                              {
                                                  {{{256, 128}, {256, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy
                                                  {{{256, 128}, {256, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x
                                                  {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},           // gamma
                                                  {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},           // mean
                                                  {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},           // rstd
                                              },
                                              {
                                                  {{{256, 128}, {256, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dx
                                                  {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},           // dgamma
                                                  {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},           // dbeta
                                              },
                                              {
                                                  /* attrs: 无 */
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "256 128 128 4 48 52 2 0 128 128 1 128 5 ";
    std::vector<size_t> expectWorkspaces = {16830464};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LayerNormalizationGradTiling, layer_normalization_grad_float16_success)
{
    struct LayerNormalizationGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("LayerNormalizationGrad",
                                              {
                                                  {{{256, 128}, {256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{256, 128}, {256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{256, 128}, {256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  /* attrs */
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "256 128 128 4 48 52 4294967298 0 128 128 1 128 5 ";
    std::vector<size_t> expectWorkspaces = {16830464};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LayerNormalizationGradTiling, layer_normalization_grad_bfloat16_success)
{
    struct LayerNormalizationGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("LayerNormalizationGrad",
                                              {
                                                  {{{256, 128}, {256, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 128}, {256, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{256, 128}, {256, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  /* attrs */
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "256 128 128 4 48 52 4294967298 0 128 128 1 128 5 ";
    std::vector<size_t> expectWorkspaces = {16830464};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
