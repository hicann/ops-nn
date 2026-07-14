/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
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

class GroupNormalizationGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupNormalizationGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupNormalizationGradTiling TearDown" << std::endl; }
};

TEST_F(GroupNormalizationGradTiling, group_normalization_grad_float32_success)
{
    struct GroupNormalizationGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GroupNormalizationGrad",
                                              {
                                                  {{{1, 2, 64}, {1, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x
                                                  {{{1, 2, 64}, {1, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy
                                                  {{{1, 2, 64}, {1, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gamma
                                                  {{{1, 2}, {1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},         // mean
                                                  {{{1, 2}, {1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},         // rstd
                                              },
                                              {
                                                  {{{1, 2, 64}, {1, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dx
                                              },
                                              {
                                                  /* attrs */
                                              },
                                              &compileInfo,
                                              64,     // number of cores
                                              262144, // UB size
                                              4096);  // max tiling data size
    uint64_t expectTilingKey = 0;
    string expectTilingData = "64 2 1 2 1 64 64 64 0 4359484440410324992 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GroupNormalizationGradTiling, group_normalization_grad_float16_success)
{
    struct GroupNormalizationGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GroupNormalizationGrad",
                                              {
                                                  {{{1, 2, 64}, {1, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                  {{{1, 2, 64}, {1, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dy
                                                  {{{1, 2, 64}, {1, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // gamma
                                                  {{{1, 2}, {1, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // mean
                                                  {{{1, 2}, {1, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // rstd
                                              },
                                              {
                                                  {{{1, 2, 64}, {1, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dx
                                              },
                                              {
                                                  /* attrs */
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "64 2 1 2 1 64 64 64 0 4359484440410324992 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GroupNormalizationGradTiling, group_normalization_grad_bfloat16_success)
{
    struct GroupNormalizationGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GroupNormalizationGrad",
                                              {
                                                  {{{1, 2, 64}, {1, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND}, // x
                                                  {{{1, 2, 64}, {1, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND}, // dy
                                                  {{{1, 2, 64}, {1, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND}, // gamma
                                                  {{{1, 2}, {1, 2}}, ge::DT_BF16, ge::FORMAT_ND},         // mean
                                                  {{{1, 2}, {1, 2}}, ge::DT_BF16, ge::FORMAT_ND},         // rstd
                                              },
                                              {
                                                  {{{1, 2, 64}, {1, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND}, // dx
                                              },
                                              {
                                                  /* attrs */
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "64 2 1 2 1 64 64 64 0 4359484440410324992 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
