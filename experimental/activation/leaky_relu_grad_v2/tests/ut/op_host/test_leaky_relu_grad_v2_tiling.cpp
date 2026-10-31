/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>

#include "tiling_case_executor.h"

class Leakyrelugradv2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Leakyrelugradv2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Leakyrelugradv2Tiling TearDown" << std::endl; }
};

TEST_F(Leakyrelugradv2Tiling, leakyrelugradv2_tiling_fp32)
{
    struct Leakyrelugradv2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LeakyReluGradV2",
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input dy
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input x
        },
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // output dx
        },
        {
            /* attrs */
        },
        &compileInfo,
        64,     // number of cores obtained in the tiling phase
        262144, // the ubsize obtained in the tiling phase
        4096);  // specifies the maximum size of the tiling data in the tiling phase
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
