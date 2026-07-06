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

static string TilingDataToString(const TilingInfo& tilingInfo)
{
    string result;
    const int64_t* data = reinterpret_cast<const int64_t*>(tilingInfo.tilingData.get());
    size_t len = tilingInfo.tilingDataSize / sizeof(int64_t);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

static void ExecuteHingeLossGradTilingCase(const gert::TilingContextPara& tilingContextPara,
                                           const string& expectTilingData,
                                           const std::vector<int64_t>& expectWorkspaces)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(TilingDataToString(tilingInfo), expectTilingData);
}

class HingeLossGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "HingeLossGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "HingeLossGradTiling TearDown" << std::endl;
    }
};

TEST_F(HingeLossGradTiling, hinge_loss_grad_float32_success)
{
    struct HingeLossGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "HingeLossGrad",
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},  // predict
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},  // target
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},  // grad_output
        },
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},  // grad_input
        },
        {},
        &compileInfo,
        64,  
        262144, 
        4096); 
    string expectTilingData = "128 0 0 1 4360 128 0 0 ";
    std::vector<int64_t> expectWorkspaces = {0};
    ExecuteHingeLossGradTilingCase(tilingContextPara, expectTilingData, expectWorkspaces);
}

TEST_F(HingeLossGradTiling, hinge_loss_grad_float16_success)
{
    struct HingeLossGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "HingeLossGrad",
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // predict
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // target
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_output
        },
        {
            {{{256, 32}, {256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_input
        },
        {},
        &compileInfo);
    string expectTilingData = "128 0 0 1 5232 128 0 0 ";
    std::vector<int64_t> expectWorkspaces = {0};
    ExecuteHingeLossGradTilingCase(tilingContextPara, expectTilingData, expectWorkspaces);
}

TEST_F(HingeLossGradTiling, hinge_loss_grad_bfloat16_success)
{
    struct HingeLossGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "HingeLossGrad",
        {
            {{{256, 32}, {256, 32}}, ge::DT_BF16, ge::FORMAT_ND},  // predict
            {{{256, 32}, {256, 32}}, ge::DT_BF16, ge::FORMAT_ND},  // target
            {{{256, 32}, {256, 32}}, ge::DT_BF16, ge::FORMAT_ND},  // grad_output
        },
        {
            {{{256, 32}, {256, 32}}, ge::DT_BF16, ge::FORMAT_ND},  // grad_input
        },
        {},
        &compileInfo);
    string expectTilingData = "128 0 0 1 5232 128 0 0 ";
    std::vector<int64_t> expectWorkspaces = {0};
    ExecuteHingeLossGradTilingCase(tilingContextPara, expectTilingData, expectWorkspaces);
}
