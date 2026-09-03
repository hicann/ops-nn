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
 * \file test_huber_loss_infershape.cpp
 * \brief InferShape and InferDataType unit tests.
 */
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "../../../op_kernel/huber_loss_tiling_data.h"

namespace {

ge::graphStatus RunInferShape(const std::initializer_list<int64_t>& inputDims,
                              const std::initializer_list<int64_t>& targetDims, int64_t reduction,
                              gert::Shape& outputShape)
{
    const auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("HuberLoss");
    if (opImpl == nullptr || opImpl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape inputShape(inputDims);
    gert::Shape targetShape(targetDims);
    // Both attributes are supplied even though infershape reads only
    // reduction: the attribute list is addressed positionally, so omitting
    // delta would leave reduction correct by accident rather than by contract.
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("HuberLoss")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&inputShape, &targetShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<int64_t>(reduction)},
                                  {"delta", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::graphStatus status = opImpl->infer_shape(context);
    if (status == ge::GRAPH_SUCCESS) {
        const gert::Shape* inferred = context->GetOutputShape(0);
        if (inferred == nullptr) {
            return ge::GRAPH_FAILED;
        }
        outputShape = *inferred;
    }
    return status;
}

ge::graphStatus RunInferDataType(ge::DataType inputDtype, ge::DataType targetDtype, ge::DataType& outputDtype)
{
    const auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("HuberLoss");
    if (opImpl == nullptr || opImpl->infer_datatype == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto holder = gert::InferDataTypeContextFaker()
                      .SetOpType("HuberLoss")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, targetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<int64_t>(HUBER_LOSS_REDUCE_MEAN)},
                                  {"delta", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::graphStatus status = opImpl->infer_datatype(context);
    if (status == ge::GRAPH_SUCCESS) {
        outputDtype = context->GetOutputDataType(0);
    }
    return status;
}

// --- reduction=none: the output takes the input shape -----------------------

TEST(HuberLossInferShapeTest, NoneCopiesInputShape)
{
    gert::Shape out;
    ASSERT_EQ(RunInferShape({2, 3, 4}, {2, 3, 4}, HUBER_LOSS_REDUCE_NONE, out), ge::GRAPH_SUCCESS);
    ASSERT_EQ(out.GetDimNum(), 3U);
    EXPECT_EQ(out.GetDim(0), 2);
    EXPECT_EQ(out.GetDim(1), 3);
    EXPECT_EQ(out.GetDim(2), 4);
}

TEST(HuberLossInferShapeTest, NoneKeepsRankZero)
{
    gert::Shape out;
    ASSERT_EQ(RunInferShape({}, {}, HUBER_LOSS_REDUCE_NONE, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out.GetDimNum(), 0U);
}

TEST(HuberLossInferShapeTest, NoneKeepsEmptyTensorShape)
{
    gert::Shape out;
    ASSERT_EQ(RunInferShape({0, 3}, {0, 3}, HUBER_LOSS_REDUCE_NONE, out), ge::GRAPH_SUCCESS);
    ASSERT_EQ(out.GetDimNum(), 2U);
    EXPECT_EQ(out.GetDim(0), 0);
    EXPECT_EQ(out.GetDim(1), 3);
}

// --- reduction=mean/sum: the output is a rank-0 scalar ----------------------
//
// Rank 0, not shape {1}. A rank-1 single-element tensor breaks the scalar
// contract and is what the aclnn output view rank check compares against.

TEST(HuberLossInferShapeTest, MeanProducesRankZeroScalar)
{
    gert::Shape out;
    ASSERT_EQ(RunInferShape({2, 3, 4}, {2, 3, 4}, HUBER_LOSS_REDUCE_MEAN, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out.GetDimNum(), 0U);
}

TEST(HuberLossInferShapeTest, SumProducesRankZeroScalar)
{
    gert::Shape out;
    ASSERT_EQ(RunInferShape({1024}, {1024}, HUBER_LOSS_REDUCE_SUM, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out.GetDimNum(), 0U);
}

TEST(HuberLossInferShapeTest, MeanOnEmptyTensorStillProducesScalar)
{
    gert::Shape out;
    ASSERT_EQ(RunInferShape({0}, {0}, HUBER_LOSS_REDUCE_MEAN, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out.GetDimNum(), 0U);
}

// --- dynamic shape ----------------------------------------------------------
//
// An unknown dimension carries no information yet, so infershape must let it
// through rather than reject a legal dynamic-shape graph. Tiling re-checks the
// concrete shapes, where they are known.

TEST(HuberLossInferShapeTest, AcceptsUnknownDimAgainstStaticDim)
{
    gert::Shape out;
    ASSERT_EQ(RunInferShape({-1, 3}, {8, 3}, HUBER_LOSS_REDUCE_NONE, out), ge::GRAPH_SUCCESS);
    ASSERT_EQ(out.GetDimNum(), 2U);
    EXPECT_EQ(out.GetDim(0), -1);
    EXPECT_EQ(out.GetDim(1), 3);
}

TEST(HuberLossInferShapeTest, AcceptsUnknownDimOnBothSides)
{
    gert::Shape out;
    ASSERT_EQ(RunInferShape({-1, -1}, {-1, -1}, HUBER_LOSS_REDUCE_NONE, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out.GetDim(0), -1);
    EXPECT_EQ(out.GetDim(1), -1);
}

// --- rejections -------------------------------------------------------------

TEST(HuberLossInferShapeTest, RejectsMismatchedDim)
{
    gert::Shape out;
    EXPECT_EQ(RunInferShape({2, 3}, {2, 4}, HUBER_LOSS_REDUCE_NONE, out), ge::GRAPH_FAILED);
}

TEST(HuberLossInferShapeTest, RejectsMismatchedRank)
{
    gert::Shape out;
    EXPECT_EQ(RunInferShape({2, 3}, {2, 3, 1}, HUBER_LOSS_REDUCE_NONE, out), ge::GRAPH_FAILED);
}

TEST(HuberLossInferShapeTest, RejectsOutOfRangeReduction)
{
    gert::Shape out;
    EXPECT_EQ(RunInferShape({8}, {8}, 3, out), ge::GRAPH_FAILED);
    EXPECT_EQ(RunInferShape({8}, {8}, -1, out), ge::GRAPH_FAILED);
}

// --- InferDataType ----------------------------------------------------------

TEST(HuberLossInferDataTypeTest, OutputFollowsInputForEveryDtype)
{
    for (const ge::DataType dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        ge::DataType out = ge::DT_UNDEFINED;
        ASSERT_EQ(RunInferDataType(dtype, dtype, out), ge::GRAPH_SUCCESS) << "dtype=" << static_cast<int>(dtype);
        // The float32 accumulation is internal; it narrows once before the
        // store and must not leak into the output type.
        EXPECT_EQ(out, dtype);
    }
}

TEST(HuberLossInferDataTypeTest, RejectsMismatchedInputDtypes)
{
    ge::DataType out = ge::DT_UNDEFINED;
    EXPECT_EQ(RunInferDataType(ge::DT_FLOAT, ge::DT_FLOAT16, out), ge::GRAPH_FAILED);
}

} // namespace
