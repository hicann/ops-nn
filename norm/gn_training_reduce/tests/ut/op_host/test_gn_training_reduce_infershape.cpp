/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_GNTrainingReduce_infershape.cpp
 * \brief GNTrainingReduce InferShape unit tests.
 */

#include <gtest/gtest.h>

#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {

ge::graphStatus RunInferShape(gert::Shape& inputShape, ge::Format format, int64_t numGroups, gert::Shape& sumShape,
                              gert::Shape& squareSumShape)
{
    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("GNTrainingReduce");
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
                      .Attr("num_groups", static_cast<int64_t>(numGroups))
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    const ge::graphStatus status = opImpl->infer_shape(context);
    if (status == ge::GRAPH_SUCCESS) {
        sumShape = *context->GetOutputShape(0);
        squareSumShape = *context->GetOutputShape(1);
    }
    return status;
}

} // namespace

TEST(GNTrainingReduceInferShapeTest, SupportsNchw)
{
    gert::Shape inputShape = {2, 4, 8, 8};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    ASSERT_EQ(RunInferShape(inputShape, ge::FORMAT_NCHW, 2, sumShape, squareSumShape), ge::GRAPH_SUCCESS);
    ASSERT_EQ(sumShape.GetDimNum(), 5U);
    EXPECT_EQ(sumShape.GetDim(0), 2);
    EXPECT_EQ(sumShape.GetDim(1), 2);
    EXPECT_EQ(sumShape.GetDim(2), 1);
    EXPECT_EQ(sumShape.GetDim(3), 1);
    EXPECT_EQ(sumShape.GetDim(4), 1);
    EXPECT_EQ(squareSumShape.GetDimNum(), 5U);
    EXPECT_EQ(squareSumShape.GetDim(0), 2);
    EXPECT_EQ(squareSumShape.GetDim(1), 2);
}

TEST(GNTrainingReduceInferShapeTest, SupportsNchwNumGroupsEqualsC)
{
    gert::Shape inputShape = {2, 4, 8, 8};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    ASSERT_EQ(RunInferShape(inputShape, ge::FORMAT_NCHW, 4, sumShape, squareSumShape), ge::GRAPH_SUCCESS);
    ASSERT_EQ(sumShape.GetDimNum(), 5U);
    EXPECT_EQ(sumShape.GetDim(1), 4);
    EXPECT_EQ(squareSumShape.GetDim(1), 4);
}

TEST(GNTrainingReduceInferShapeTest, SupportsNhwc)
{
    gert::Shape inputShape = {2, 8, 8, 4};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    ASSERT_EQ(RunInferShape(inputShape, ge::FORMAT_NHWC, 2, sumShape, squareSumShape), ge::GRAPH_SUCCESS);
    ASSERT_EQ(sumShape.GetDimNum(), 5U);
    EXPECT_EQ(sumShape.GetDim(0), 2);
    EXPECT_EQ(sumShape.GetDim(1), 1);
    EXPECT_EQ(sumShape.GetDim(2), 1);
    EXPECT_EQ(sumShape.GetDim(3), 2);
    EXPECT_EQ(sumShape.GetDim(4), 1);
    EXPECT_EQ(squareSumShape.GetDimNum(), 5U);
    EXPECT_EQ(squareSumShape.GetDim(3), 2);
}

TEST(GNTrainingReduceInferShapeTest, RejectsWrongRank)
{
    gert::Shape inputShape = {2, 4, 8};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    EXPECT_EQ(RunInferShape(inputShape, ge::FORMAT_NCHW, 2, sumShape, squareSumShape), ge::GRAPH_FAILED);
}

TEST(GNTrainingReduceInferShapeTest, RejectsNdFormat)
{
    gert::Shape inputShape = {2, 4, 8, 8};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    EXPECT_EQ(RunInferShape(inputShape, ge::FORMAT_ND, 2, sumShape, squareSumShape), ge::GRAPH_FAILED);
}

TEST(GNTrainingReduceInferShapeTest, RejectsNumGroupsNotDivideC)
{
    gert::Shape inputShape = {2, 4, 8, 8};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    EXPECT_EQ(RunInferShape(inputShape, ge::FORMAT_NCHW, 3, sumShape, squareSumShape), ge::GRAPH_FAILED);
}

TEST(GNTrainingReduceInferShapeTest, RejectsNumGroupsZero)
{
    gert::Shape inputShape = {2, 4, 8, 8};
    gert::Shape sumShape;
    gert::Shape squareSumShape;

    EXPECT_EQ(RunInferShape(inputShape, ge::FORMAT_NCHW, 0, sumShape, squareSumShape), ge::GRAPH_FAILED);
}

} // namespace ops
