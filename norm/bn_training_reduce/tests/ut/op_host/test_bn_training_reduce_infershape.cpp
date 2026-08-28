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

namespace ops {
namespace {

ge::graphStatus RunInferShape(gert::Shape& inputShape, ge::Format format, gert::Shape& sumShape,
                              gert::Shape& squareSumShape)
{
    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("BNTrainingReduce");
    if (opImpl == nullptr || opImpl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1}, {1, 1})
                      .InputShapes({&inputShape})
                      .OutputShapes({&sumShape, &squareSumShape})
                      .NodeInputTd(0, ge::DT_FLOAT, format, format)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    const ge::graphStatus status = opImpl->infer_shape(context);
    if (status == ge::GRAPH_SUCCESS) {
        sumShape = *context->GetOutputShape(0);
        squareSumShape = *context->GetOutputShape(1);
    }
    return status;
}

TEST(BNTrainingReduceInferShapeTest, SupportsNchw)
{
    gert::Shape inputShape = {2, 3, 4, 5};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    ASSERT_EQ(RunInferShape(inputShape, ge::FORMAT_NCHW, sumShape, squareSumShape), ge::GRAPH_SUCCESS);
    ASSERT_EQ(sumShape.GetDimNum(), 1U);
    EXPECT_EQ(sumShape.GetDim(0), 3);
    EXPECT_EQ(squareSumShape.GetDim(0), 3);
}

TEST(BNTrainingReduceInferShapeTest, SupportsNchwRank2)
{
    gert::Shape inputShape = {4, 3};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    ASSERT_EQ(RunInferShape(inputShape, ge::FORMAT_NCHW, sumShape, squareSumShape), ge::GRAPH_SUCCESS);
    ASSERT_EQ(sumShape.GetDimNum(), 1U);
    EXPECT_EQ(sumShape.GetDim(0), 3);
    EXPECT_EQ(squareSumShape.GetDim(0), 3);
}

TEST(BNTrainingReduceInferShapeTest, SupportsNhwc)
{
    gert::Shape inputShape = {2, 4, 5, 3};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    ASSERT_EQ(RunInferShape(inputShape, ge::FORMAT_NHWC, sumShape, squareSumShape), ge::GRAPH_SUCCESS);
    ASSERT_EQ(sumShape.GetDimNum(), 1U);
    EXPECT_EQ(sumShape.GetDim(0), 3);
    EXPECT_EQ(squareSumShape.GetDim(0), 3);
}

TEST(BNTrainingReduceInferShapeTest, SupportsNcdhw)
{
    gert::Shape inputShape = {2, 3, 4, 5, 6};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    ASSERT_EQ(RunInferShape(inputShape, ge::FORMAT_NCDHW, sumShape, squareSumShape), ge::GRAPH_SUCCESS);
    ASSERT_EQ(sumShape.GetDimNum(), 1U);
    EXPECT_EQ(sumShape.GetDim(0), 3);
    EXPECT_EQ(squareSumShape.GetDim(0), 3);
}

TEST(BNTrainingReduceInferShapeTest, RejectsNd)
{
    gert::Shape inputShape = {2, 3, 4, 5};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    EXPECT_EQ(RunInferShape(inputShape, ge::FORMAT_ND, sumShape, squareSumShape), ge::GRAPH_FAILED);
}

TEST(BNTrainingReduceInferShapeTest, RejectsNdc1hwc0OnAscend950)
{
    gert::Shape inputShape = {2, 1, 3, 4, 5, 16};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    EXPECT_EQ(RunInferShape(inputShape, ge::FORMAT_NDC1HWC0, sumShape, squareSumShape), ge::GRAPH_FAILED);
}

TEST(BNTrainingReduceInferShapeTest, RejectsWrongRankForNhwc)
{
    gert::Shape inputShape = {2, 4, 3};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    EXPECT_EQ(RunInferShape(inputShape, ge::FORMAT_NHWC, sumShape, squareSumShape), ge::GRAPH_FAILED);
}

TEST(BNTrainingReduceInferShapeTest, RejectsWrongRankForNcdhw)
{
    gert::Shape inputShape = {2, 3, 4, 5};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    EXPECT_EQ(RunInferShape(inputShape, ge::FORMAT_NCDHW, sumShape, squareSumShape), ge::GRAPH_FAILED);
}

} // namespace
} // namespace ops
