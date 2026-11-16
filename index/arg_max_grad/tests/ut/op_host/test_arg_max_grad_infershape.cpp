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
#include "infershape_test_util.h"
#include "ut_op_common.h"

namespace {
class ArgMaxGradInferShapeTest : public testing::Test {};

void CheckInferShape(gert::Shape varShape, ge::graphStatus expectedStatus, ge::DataType dtype = ge::DT_FLOAT)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("ArgMaxGrad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    gert::Shape indexShape = {1};
    gert::Shape outputShape = {99};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&varShape, &indexShape, &indexShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(opImpl->infer_shape(context), expectedStatus);
    if (expectedStatus != ge::GRAPH_SUCCESS) {
        return;
    }
    const auto actualShape = context->GetOutputShape(0);
    ASSERT_NE(actualShape, nullptr);
    ASSERT_EQ(actualShape->GetDimNum(), varShape.GetDimNum());
    for (size_t i = 0; i < varShape.GetDimNum(); ++i) {
        EXPECT_EQ(actualShape->GetDim(i), varShape.GetDim(i));
    }
}

TEST_F(ArgMaxGradInferShapeTest, rejectScalarVar)
{
    for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8}) {
        CheckInferShape({}, ge::GRAPH_FAILED, dtype);
    }
}

TEST_F(ArgMaxGradInferShapeTest, acceptRankOneVar)
{
    for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8}) {
        CheckInferShape({1}, ge::GRAPH_SUCCESS, dtype);
    }
}

TEST_F(ArgMaxGradInferShapeTest, preserveStaticShape) { CheckInferShape({2, 3, 4}, ge::GRAPH_SUCCESS); }

TEST_F(ArgMaxGradInferShapeTest, preserveDynamicDimensions) { CheckInferShape({2, -1, 4}, ge::GRAPH_SUCCESS); }

TEST_F(ArgMaxGradInferShapeTest, preserveUnknownRank) { CheckInferShape({-2}, ge::GRAPH_SUCCESS); }

TEST_F(ArgMaxGradInferShapeTest, preserveEmptyTensor) { CheckInferShape({0}, ge::GRAPH_SUCCESS); }
} // namespace
