/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_deformable_offsets_tiling.cpp
 * \brief
 */

#include "../../../op_host/arch35/deformable_offsets_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class DeformableOffsetsTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DeformableOffsetsTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DeformableOffsetsTiling TearDown" << std::endl;
    }
};

TEST_F(DeformableOffsetsTiling, deformable_offsets_test_0)
{
    optiling::TilingPrepareForDeformableOffsetsCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "DeformableOffsets",
        {
            {{{1, 7, 11, 256}, {1, 7, 11, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{1, 7, 11, 27}, {1, 7, 11, 27}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{1, 21, 33, 256}, {1, 21, 33, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr(
                "strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})),
            gert::TilingContextPara::OpAttr("pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})),
            gert::TilingContextPara::OpAttr("ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({3, 3})),
            gert::TilingContextPara::OpAttr(
                "dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})),
            gert::TilingContextPara::OpAttr("data_format", Ops::NN::AnyValue::CreateFrom<string>("NHWC")),
            gert::TilingContextPara::OpAttr("deformable_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)),
            gert::TilingContextPara::OpAttr("modulated", Ops::NN::AnyValue::CreateFrom<bool>(true)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 1000;
    string expectTilingData =
        "4294967335 4294967297 4294967297 12884901889 1099511627779 30064771083 30064773888 38654705675 1275605286939 "
        "4294967299 36283883717376 84662395364096 8929237028096 4295144704 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}