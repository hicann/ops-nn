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

#include <iostream>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;
using namespace ge;

class MedianTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MedianTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MedianTiling TearDown" << std::endl; }
};

// Median: 1 输入(input) / 2 输出(values, indices) + 属性 dim/keepdim。
// 仅校验 tilingKey 与 workspace（MedianTilingData 为普通结构体，tilingData 字符串序列化依赖字段元信息，故不做强校验）。
// float [32,4,4,4]：numel=2048, redLen=4, batch=512, mid=1, dtype=1, nSeg=0；
// workspace = 32MB + numel*32 = 33554432 + 65536 = 33619968。
TEST_F(MedianTiling, median_0)
{
    struct MedianCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Median",
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input
        },
        {
            {{{32, 4, 4}, {32, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // values
            {{{32, 4, 4}, {32, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND}, // indices
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo,
        64,     // number of cores obtained in the tiling phase
        262144, // the ubsize obtained in the tiling phase
        4096);  // specifies the maximum size of the tiling data in the tiling phase
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {33619968};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// int32 [32,4,4,4]：同上，dtype=2。
TEST_F(MedianTiling, median_1)
{
    struct MedianCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Median",
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND}, // input
        },
        {
            {{{32, 4, 4}, {32, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND}, // values
            {{{32, 4, 4}, {32, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND}, // indices
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo, 64, 262144, 4096);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {33619968};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
