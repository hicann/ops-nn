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
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"

namespace {
ge::graphStatus RunInferShape(const std::initializer_list<int64_t>& inputDims,
                              const std::initializer_list<int64_t>& targetDims, const std::string& reduction,
                              gert::Shape& outputShape)
{
    const auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("HingeEmbeddingLoss");
    if (opImpl == nullptr || opImpl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape inputShape(inputDims);
    gert::Shape targetShape(targetDims);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("HingeEmbeddingLoss")
                      .NodeIoNum(2, 1)
                      .InputShapes({&inputShape, &targetShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs({{"margin", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::graphStatus status = opImpl->infer_shape(context);
    if (status == ge::GRAPH_SUCCESS) {
        outputShape = *context->GetOutputShape(0);
    }
    return status;
}

TEST(HingeEmbeddingLossInferShape, NoneCopiesInputShape)
{
    gert::Shape output;
    ASSERT_EQ(RunInferShape({2, 3, 4}, {2, 3, 4}, "none", output), ge::GRAPH_SUCCESS);
    ASSERT_EQ(output.GetDimNum(), 3);
    EXPECT_EQ(output.GetDim(0), 2);
    EXPECT_EQ(output.GetDim(1), 3);
    EXPECT_EQ(output.GetDim(2), 4);
}

TEST(HingeEmbeddingLossInferShape, SumAndMeanProduceSingleElement)
{
    for (const std::string reduction : {"sum", "mean"}) {
        gert::Shape output;
        ASSERT_EQ(RunInferShape({7, 5}, {7, 5}, reduction, output), ge::GRAPH_SUCCESS);
        ASSERT_EQ(output.GetDimNum(), 1);
        EXPECT_EQ(output.GetDim(0), 1);
    }
}

TEST(HingeEmbeddingLossInferShape, SupportsMatchingDynamicShape)
{
    gert::Shape output;
    ASSERT_EQ(RunInferShape({-1, 3}, {-1, 3}, "none", output), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output.GetDim(0), -1);
    EXPECT_EQ(output.GetDim(1), 3);
}

TEST(HingeEmbeddingLossInferShape, RejectsInvalidContract)
{
    gert::Shape output;
    EXPECT_EQ(RunInferShape({2, 3}, {2, 4}, "none", output), ge::GRAPH_FAILED);
    EXPECT_EQ(RunInferShape({2, 3}, {2, 3}, "invalid", output), ge::GRAPH_FAILED);
}
} // namespace
