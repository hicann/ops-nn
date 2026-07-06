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

class SoftmaxGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SoftmaxGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SoftmaxGradTiling TearDown" << std::endl;
    }
};

TEST_F(SoftmaxGradTiling, softmax_grad_float32_success)
{
    struct SoftmaxGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SoftmaxGrad",
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}, // y
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy
        },
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dx
        },
        {
            /* attrs */
        },
        &compileInfo,
        64,
        262144,
        4096);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "256 32 32 4 0 64 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(SoftmaxGradTiling, softmax_grad_float16_success)
{
    struct SoftmaxGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SoftmaxGrad",
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dy
        },
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dx
        },
        {
            /* attrs */
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "256 32 32 4 0 64 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(SoftmaxGradTiling, softmax_grad_bfloat16_success)
{
    struct SoftmaxGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SoftmaxGrad",
        {
            {{{256, 32}, {256, 32}}, ge::DT_BF16, ge::FORMAT_ND}, // y
            {{{256, 32}, {256, 32}}, ge::DT_BF16, ge::FORMAT_ND}, // dy
        },
        {
            {{{256, 32}, {256, 32}}, ge::DT_BF16, ge::FORMAT_ND}, // dx
        },
        {
            /* attrs */
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "256 32 32 4 0 64 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
