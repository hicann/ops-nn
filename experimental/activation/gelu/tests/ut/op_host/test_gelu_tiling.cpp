/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <iostream>
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class GeluTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GeluTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GeluTiling TearDown" << std::endl; }
};

TEST_F(GeluTiling, gelu_tiling_fp32)
{
    struct GeluCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Gelu",
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input x
        },
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // output y
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
    string expectTilingData = "0 2048 384 1 6 2048 2048 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GeluTiling, gelu_tiling_fp16)
{
    struct GeluCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Gelu",
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // input x
        },
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // output y
        },
        {
            /* attrs */
        },
        &compileInfo,
        64,     // number of cores obtained in the tiling phase
        262144, // the ubsize obtained in the tiling phase, but the actual obtained value is 256 bytes less than the
                // specified value
        4096);  // specifies the maximum size of the tiling data in the tiling phase
    uint64_t expectTilingKey = 1;
    string expectTilingData = "0 4096 768 1 6 4096 4096 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GeluTiling, gelu_tiling_bf16)
{
    struct GeluCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Gelu",
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_BF16, ge::FORMAT_ND}, // input x
        },
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_BF16, ge::FORMAT_ND}, // output y
        },
        {
            /* attrs */
        },
        &compileInfo,
        64,     // number of cores obtained in the tiling phase
        262144, // the ubsize obtained in the tiling phase, but the actual obtained value is 256 bytes less than the
                // specified value
        4096);  // specifies the maximum size of the tiling data in the tiling phase
    uint64_t expectTilingKey = 1;
    string expectTilingData = "0 4096 768 1 6 4096 4096 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
