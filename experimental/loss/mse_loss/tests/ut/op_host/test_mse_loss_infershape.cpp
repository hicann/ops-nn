/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"

namespace {

ge::graphStatus RunInferShape(const std::initializer_list<int64_t>& predictDims,
                              const std::initializer_list<int64_t>& labelDims, const std::string& reduction,
                              gert::Shape& outputShape)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLoss");
    if (opImpl == nullptr || opImpl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape predictShape(predictDims);
    gert::Shape labelShape(labelDims);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLoss")
                      .NodeIoNum(2, 1)
                      .InputShapes({&predictShape, &labelShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus status = opImpl->infer_shape(context);
    if (status == ge::GRAPH_SUCCESS) {
        const gert::Shape* inferredShape = context->GetOutputShape(0);
        if (inferredShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        outputShape = *inferredShape;
    }
    return status;
}

TEST(MseLossInferShapeTest, NoneCopiesInputShape)
{
    gert::Shape outputShape;
    ASSERT_EQ(RunInferShape({2, 3, 4}, {2, 3, 4}, "none", outputShape), ge::GRAPH_SUCCESS);
    ASSERT_EQ(outputShape.GetDimNum(), 3);
    EXPECT_EQ(outputShape.GetDim(0), 2);
    EXPECT_EQ(outputShape.GetDim(1), 3);
    EXPECT_EQ(outputShape.GetDim(2), 4);
}

TEST(MseLossInferShapeTest, ReductionProducesScalarTensor)
{
    for (const std::string reduction : {"sum", "mean"}) {
        gert::Shape outputShape;
        ASSERT_EQ(RunInferShape({7, 5}, {7, 5}, reduction, outputShape), ge::GRAPH_SUCCESS);
        ASSERT_EQ(outputShape.GetDimNum(), 1);
        EXPECT_EQ(outputShape.GetDim(0), 1);
    }
}

TEST(MseLossInferShapeTest, RejectsMismatchedInputShapes)
{
    gert::Shape outputShape;
    EXPECT_EQ(RunInferShape({2, 3}, {2, 4}, "none", outputShape), ge::GRAPH_FAILED);
}

TEST(MseLossInferShapeTest, RejectsInvalidReduction)
{
    gert::Shape outputShape;
    EXPECT_EQ(RunInferShape({2, 3}, {2, 3}, "invalid", outputShape), ge::GRAPH_FAILED);
}

} // namespace
