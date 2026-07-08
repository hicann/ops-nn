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

class CrossEntropyGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CrossEntropyGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CrossEntropyGradTiling TearDown" << std::endl;
    }
};

TEST_F(CrossEntropyGradTiling, cross_entropy_grad_float32_int32_success)
{
    struct CrossEntropyGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "CrossEntropyGrad",
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},   // logits
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},           // target
        },
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},   // grad_input
        },
        {
            /* attrs */
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1099511627808 21474836484 1099511627776 4294967297 21474836484 137438953472 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CrossEntropyGradTiling, cross_entropy_grad_float16_int32_success)
{
    struct CrossEntropyGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "CrossEntropyGrad",
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // logits
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},             // target
        },
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_input
        },
        {
            /* attrs */
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1099511627808 21474836484 1099511627776 4294967297 21474836484 137438953472 4294967297 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CrossEntropyGradTiling, cross_entropy_grad_bf16_int32_success)
{
    struct CrossEntropyGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "CrossEntropyGrad",
        {
            {{{256, 32}, {256, 32}}, ge::DT_BF16, ge::FORMAT_ND},     // logits
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},             // target
        },
        {
            {{{256, 32}, {256, 32}}, ge::DT_BF16, ge::FORMAT_ND},     // grad_input
        },
        {
            /* attrs */
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1099511627808 21474836484 1099511627776 4294967297 21474836484 137438953472 4294967297 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 列分割模式测试：rowLen 过大导致 UB 无法容纳完整一行时，自动启用 colSplit 模式
TEST_F(CrossEntropyGradTiling, cross_entropy_grad_float32_colsplit_success)
{
    struct CrossEntropyGradCompileInfo {
    } compileInfo;
    // rowLen=65536 远超 UB 容量，强制进入 colSplit 模式
    gert::TilingContextPara tilingContextPara(
        "CrossEntropyGrad",
        {
            {{{1, 65536}, {1, 65536}}, ge::DT_FLOAT, ge::FORMAT_ND},   // logits
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                  // target
        },
        {
            {{{1, 65536}, {1, 65536}}, ge::DT_FLOAT, ge::FORMAT_ND},   // grad_input
        },
        {
            /* attrs */
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    // colSplitMode=1
    string expectTilingData = "4295032832 8589934593 4294967296 8589934593 0 55490977464321 6 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
