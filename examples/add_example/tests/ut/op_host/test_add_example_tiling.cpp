/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

class AddExampleTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AddExampleTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AddExampleTiling TearDown" << std::endl; }
};

TEST_F(AddExampleTiling, add_example_0)
{
    struct AddExampleCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AddExample",
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input tensor1
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input tensor2
        },
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // output tensor
        },
        {
            /* attrs */
        },
        &compileInfo,
        64,     // number of cores obtained in the tiling phase
        262144, // the ubsize obtained in the tiling phase, but the actual obtained value is 256 bytes less than the
                // specified value
        4096);  // specifies the maximum size of the tiling data in the tiling phase
    uint64_t expectTilingKey = 0;
    string expectTilingData = "2048 32 10912 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AddExampleTiling, add_example_1)
{
    struct AddExampleCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AddExample",
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND}, // input tensor1
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND}, // input tensor2
        },
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND}, // output tensor
        },
        {
            /* attrs */
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "2048 32 10912 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}