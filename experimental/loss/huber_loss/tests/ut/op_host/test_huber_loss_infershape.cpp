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
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace {
ge::graphStatus RunInferShape(const std::initializer_list<int64_t>& predictionDims,
                              const std::initializer_list<int64_t>& targetDims, gert::Shape& outputShape)
{
    const auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("HuberLoss");
    if (opImpl == nullptr || opImpl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape predictionShape(predictionDims);
    gert::Shape targetShape(targetDims);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("HuberLoss")
                      .NodeIoNum(2, 1)
                      .InputShapes({&predictionShape, &targetShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::graphStatus status = opImpl->infer_shape(context);
    if (status == ge::GRAPH_SUCCESS) {
        const gert::Shape* inferredShape = context->GetOutputShape(0);
        if (inferredShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        outputShape = *inferredShape;
    }
    return status;
}

TEST(HuberLossInferShapeTest, CopiesInputShape)
{
    gert::Shape outputShape;
    ASSERT_EQ(RunInferShape({2, 3, 4}, {2, 3, 4}, outputShape), ge::GRAPH_SUCCESS);
    ASSERT_EQ(outputShape.GetDimNum(), 3);
    EXPECT_EQ(outputShape.GetDim(0), 2);
    EXPECT_EQ(outputShape.GetDim(1), 3);
    EXPECT_EQ(outputShape.GetDim(2), 4);
}

TEST(HuberLossInferShapeTest, SupportsScalarAndEmptyShapes)
{
    gert::Shape scalarOutput;
    EXPECT_EQ(RunInferShape({}, {}, scalarOutput), ge::GRAPH_SUCCESS);
    EXPECT_EQ(scalarOutput.GetDimNum(), 0);

    gert::Shape emptyOutput;
    ASSERT_EQ(RunInferShape({0, 3}, {0, 3}, emptyOutput), ge::GRAPH_SUCCESS);
    ASSERT_EQ(emptyOutput.GetDimNum(), 2);
    EXPECT_EQ(emptyOutput.GetDim(0), 0);
    EXPECT_EQ(emptyOutput.GetDim(1), 3);
}

TEST(HuberLossInferShapeTest, CopiesMatchingDynamicShape)
{
    gert::Shape outputShape;
    ASSERT_EQ(RunInferShape({-1, 3}, {-1, 3}, outputShape), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputShape.GetDim(0), -1);
    EXPECT_EQ(outputShape.GetDim(1), 3);
}

TEST(HuberLossInferShapeTest, RejectsMismatchedInputShapes)
{
    gert::Shape outputShape;
    EXPECT_EQ(RunInferShape({2, 3}, {2, 4}, outputShape), ge::GRAPH_FAILED);
}
} // namespace
