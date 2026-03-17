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

#include "../../../op_host/arch35/deformable_offsets_grad_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class DeformableOffsetsGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DeformableOffsetsGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DeformableOffsetsGradTiling TearDown" << std::endl;
    }
};

TEST_F(DeformableOffsetsGradTiling, deformable_offsets_grad_test_0)
{
    optiling::TilingPrepareForDeformableOffsetsGradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "DeformableOffsetsGrad",
        {
            {{{72, 2, 45, 64}, {72, 2, 45, 64}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{72, 1, 17, 64}, {72, 1, 17, 64}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{72, 1, 9, 240}, {72, 1, 9, 240}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{72, 1, 17, 64}, {72, 1, 17, 64}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{72, 1, 9, 240}, {72, 1, 9, 240}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr(
                "strides", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 1})),
            gert::TilingContextPara::OpAttr("pads", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2})),
            gert::TilingContextPara::OpAttr("ksize", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({2, 5})),
            gert::TilingContextPara::OpAttr(
                "dilations", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 1, 1})),
            gert::TilingContextPara::OpAttr("data_format", Ops::Cv::AnyValue::CreateFrom<string>("NHWC")),
            gert::TilingContextPara::OpAttr("deformable_groups", Ops::Cv::AnyValue::CreateFrom<int64_t>(8)),
            gert::TilingContextPara::OpAttr("modulated", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 1001;
    string expectTilingData =
        "274877907945 274877907008 8589934593 4294967298 8589934593 21474836482 274877907016 4294967313 38654705665 "
        "3478923509768 5257039970304 10436770529280 336450558099456 155520 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}