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
#include "infershape_test_util.h"
#include "../../../op_graph/bn_inference_proto.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

namespace {
void AddRequiredInputs(ge::op::BNInference& testOp, const std::vector<int64_t>& xShape, ge::DataType xDtype,
                       ge::Format xFormat, int64_t channel)
{
    std::vector<std::pair<int64_t, int64_t>> xRange(xShape.size(), {-1, -1});
    const std::vector<int64_t> parameterShape = {channel};
    const std::vector<std::pair<int64_t, int64_t>> parameterRange = {{-1, -1}};
    const std::vector<int64_t> momentumShape;
    const std::vector<std::pair<int64_t, int64_t>> momentumRange;

    TENSOR_INPUT_WITH_SHAPE(testOp, x, xShape, xDtype, xFormat, xRange);
    TENSOR_INPUT_WITH_SHAPE(testOp, mean, parameterShape, ge::DT_FLOAT, ge::FORMAT_ND, parameterRange);
    TENSOR_INPUT_WITH_SHAPE(testOp, variance, parameterShape, ge::DT_FLOAT, ge::FORMAT_ND, parameterRange);
    TENSOR_INPUT_WITH_SHAPE(testOp, momentum, momentumShape, ge::DT_FLOAT, ge::FORMAT_ND, momentumRange);
}

void ExpectShapeCopied(const std::vector<int64_t>& xShape, ge::DataType dtype, ge::Format format, int64_t channel)
{
    auto testOp = ge::op::BNInference("BNInference");
    AddRequiredInputs(testOp, xShape, dtype, format, channel);
    ASSERT_EQ(InferShapeTest(testOp), ge::GRAPH_SUCCESS);
    EXPECT_EQ(testOp.GetOutputDesc(0).GetShape().GetDims(), xShape);
}
} // namespace

TEST(BNInferenceInferShapeTest, CopiesNchwShape) { ExpectShapeCopied({2, 3, 4, 5}, ge::DT_FLOAT, ge::FORMAT_NCHW, 3); }

TEST(BNInferenceInferShapeTest, CopiesNhwcShape)
{
    ExpectShapeCopied({2, 4, 5, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, 3);
}

TEST(BNInferenceInferShapeTest, CopiesNdRank4Shape) { ExpectShapeCopied({2, 3, 4, 5}, ge::DT_BF16, ge::FORMAT_ND, 3); }

TEST(BNInferenceInferShapeTest, CopiesNdRank5Shape)
{
    ExpectShapeCopied({2, 3, 4, 5, 6}, ge::DT_FLOAT, ge::FORMAT_ND, 3);
}

TEST(BNInferenceInferShapeTest, CopiesEmptyChannelShape)
{
    ExpectShapeCopied({2, 4, 5, 0}, ge::DT_FLOAT, ge::FORMAT_NHWC, 0);
}

TEST(BNInferenceInferShapeTest, CopiesDynamicDimensionShape)
{
    ExpectShapeCopied({-1, 3, 4, 5}, ge::DT_FLOAT, ge::FORMAT_NCHW, 3);
}

TEST(BNInferenceInferShapeTest, CopiesUnknownRankShape) { ExpectShapeCopied({-2}, ge::DT_FLOAT, ge::FORMAT_NCHW, 3); }
