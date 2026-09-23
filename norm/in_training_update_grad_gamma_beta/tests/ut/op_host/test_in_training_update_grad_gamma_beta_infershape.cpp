/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_in_training_update_grad_gamma_beta_infershape.cpp
 * \brief InferShape UT for INTrainingUpdateGradGammaBeta。
 *        两个输出的 shape 取自 res_gamma（dim0 置 1）。
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "base/context_builder/op_infer_shape_context_builder.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "../../../op_graph/in_training_update_grad_gamma_beta_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

class INTrainingUpdateGradGammaBetaTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "INTrainingUpdateGradGammaBeta Proto Test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "INTrainingUpdateGradGammaBeta Proto Test TearDown" << std::endl; }
};

namespace {
gert::Shape MakeInferShape(const std::vector<int64_t>& dims)
{
    gert::Shape shape;
    for (const auto dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

void CheckDeclaredOutputs(const std::vector<int64_t>& inputDims, const std::vector<int64_t>& gammaOutputDims,
                          const std::vector<int64_t>& betaOutputDims, ge::graphStatus expectedStatus)
{
    gert::Tensor gamma;
    gert::Tensor beta;
    for (auto* tensor : {&gamma, &beta}) {
        tensor->MutableOriginShape() = tensor->MutableStorageShape() = MakeInferShape(inputDims);
        tensor->SetDataType(ge::DT_FLOAT);
        tensor->SetOriginFormat(ge::FORMAT_NCHW);
        tensor->SetStorageFormat(ge::FORMAT_NCHW);
    }
    gert::OpInferShapeContextBuilder builder;
    auto holder = builder.OpType("INTrainingUpdateGradGammaBeta")
                      .OpName("declared_outputs")
                      .IONum(2, 2)
                      .InputTensors({&gamma, &beta})
                      .OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .OutputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetOutputShape(0), nullptr);
    ASSERT_NE(context->GetOutputShape(1), nullptr);
    // Set the real context buffers: the compatibility faker ignores OutputShapes().
    *context->GetOutputShape(0) = MakeInferShape(gammaOutputDims);
    *context->GetOutputShape(1) = MakeInferShape(betaOutputDims);
    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("INTrainingUpdateGradGammaBeta");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    ASSERT_EQ(impl->infer_shape(context), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        auto expected = MakeInferShape(inputDims);
        if (inputDims != std::vector<int64_t>{ge::UNKNOWN_DIM_NUM}) {
            expected.SetDim(0, 1);
        }
        EXPECT_EQ(*context->GetOutputShape(0), expected);
        EXPECT_EQ(*context->GetOutputShape(1), expected);
    }
}
} // namespace

TEST_F(INTrainingUpdateGradGammaBetaTest, accept_concrete_outputs_with_unknown_input_dimension)
{
    CheckDeclaredOutputs({4, -1, 3, 5}, {1, 64, 3, 5}, {1, 64, 3, 5}, ge::GRAPH_SUCCESS);
}

TEST_F(INTrainingUpdateGradGammaBetaTest, accept_concrete_outputs_with_unknown_input_rank)
{
    CheckDeclaredOutputs({-2}, {1, 64, 3, 5}, {1, 64, 3, 5}, ge::GRAPH_SUCCESS);
}

TEST_F(INTrainingUpdateGradGammaBetaTest, accept_matching_declared_outputs)
{
    CheckDeclaredOutputs({4, 64, 3, 5}, {1, 64, 3, 5}, {1, 64, 3, 5}, ge::GRAPH_SUCCESS);
}

TEST_F(INTrainingUpdateGradGammaBetaTest, reject_declared_gamma_output_mismatch)
{
    CheckDeclaredOutputs({4, 64, 3, 5}, {1, 32, 3, 5}, {1, 64, 3, 5}, ge::GRAPH_FAILED);
}

TEST_F(INTrainingUpdateGradGammaBetaTest, reject_declared_beta_output_mismatch)
{
    CheckDeclaredOutputs({4, 64, 3, 5}, {1, 64, 3, 5}, {2, 64, 3, 5}, ge::GRAPH_FAILED);
}

TEST_F(INTrainingUpdateGradGammaBetaTest, reject_known_output_dimension_mismatch_with_dynamic_input)
{
    CheckDeclaredOutputs({4, -1, 3, 5}, {1, 64, 4, 5}, {1, 64, 3, 5}, ge::GRAPH_FAILED);
}

// 输出 shape 跟随 res_gamma（dim0 置 1，其余维原样）。
TEST_F(INTrainingUpdateGradGammaBetaTest, in_training_update_grad_gamma_beta_infer_shape_4d)
{
    ge::op::INTrainingUpdateGradGammaBeta op;

    ge::Format format = ge::FORMAT_NCHW;
    auto in_tensor = create_desc_with_ori({4, 64, 3, 5}, ge::DT_FLOAT, format, {4, 64, 3, 5}, format);
    op.UpdateInputDesc("res_gamma", in_tensor);
    op.UpdateInputDesc("res_beta", in_tensor);

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto pd_gamma_desc = op.GetOutputDescByName("pd_gamma");
    auto pd_beta_desc = op.GetOutputDescByName("pd_beta");
    EXPECT_EQ(pd_gamma_desc.GetShape().GetDimNum(), 4);
    EXPECT_EQ(pd_gamma_desc.GetShape().GetDim(0), 1);
    EXPECT_EQ(pd_gamma_desc.GetShape().GetDim(1), 64);
    EXPECT_EQ(pd_gamma_desc.GetShape().GetDim(2), 3);
    EXPECT_EQ(pd_gamma_desc.GetShape().GetDim(3), 5);
    EXPECT_EQ(pd_beta_desc.GetShape().GetDims(), pd_gamma_desc.GetShape().GetDims());
}

// 5 维 NCDHW 输入：dim0 置 1，其余维原样拷贝。
TEST_F(INTrainingUpdateGradGammaBetaTest, in_training_update_grad_gamma_beta_infer_shape_5d)
{
    ge::op::INTrainingUpdateGradGammaBeta op;

    ge::Format format = ge::FORMAT_NCDHW;
    auto in_tensor = create_desc_with_ori({2, 3, 4, 5, 6}, ge::DT_FLOAT, format, {2, 3, 4, 5, 6}, format);
    op.UpdateInputDesc("res_gamma", in_tensor);
    op.UpdateInputDesc("res_beta", in_tensor);

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto pd_gamma_desc = op.GetOutputDescByName("pd_gamma");
    EXPECT_EQ(pd_gamma_desc.GetShape().GetDimNum(), 5);
    EXPECT_EQ(pd_gamma_desc.GetShape().GetDim(0), 1);
    EXPECT_EQ(pd_gamma_desc.GetShape().GetDim(1), 3);
    EXPECT_EQ(pd_gamma_desc.GetShape().GetDim(4), 6);
}

// -2 UNKNOWN_RANK：输入 unknown-rank 时输出原样透传，不得推成非法形状。
TEST_F(INTrainingUpdateGradGammaBetaTest, in_training_update_grad_gamma_beta_infer_shape_unknown_rank)
{
    ge::op::INTrainingUpdateGradGammaBeta op;
    op.UpdateInputDesc("res_gamma", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("res_beta", create_desc({-2}, ge::DT_FLOAT));
    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    std::vector<int64_t> expected{-2};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected);
}

// InferShape 反向：origin format 不在公开契约域（FRACTAL_NZ）必须被拒绝。
TEST_F(INTrainingUpdateGradGammaBetaTest, in_training_update_grad_gamma_beta_reject_fractal_nz_format)
{
    ge::op::INTrainingUpdateGradGammaBeta op;

    ge::Format format = ge::FORMAT_FRACTAL_NZ;
    auto in_tensor = create_desc_with_ori({4, 64, 3, 5}, ge::DT_FLOAT, format, {4, 64, 3, 5}, format);
    op.UpdateInputDesc("res_gamma", in_tensor);
    op.UpdateInputDesc("res_beta", in_tensor);

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

// InferShape 反向：origin format 与 shape rank 不匹配（NCHW 声明 5 维 shape）必须被拒绝。
TEST_F(INTrainingUpdateGradGammaBetaTest, in_training_update_grad_gamma_beta_reject_format_rank_mismatch)
{
    ge::op::INTrainingUpdateGradGammaBeta op;

    ge::Format format = ge::FORMAT_NCHW;
    auto in_tensor = create_desc_with_ori({2, 3, 4, 5, 6}, ge::DT_FLOAT, format, {2, 3, 4, 5, 6}, format);
    op.UpdateInputDesc("res_gamma", in_tensor);
    op.UpdateInputDesc("res_beta", in_tensor);

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

// InferShape 反向：两输入 origin format 不一致必须被拒绝。
TEST_F(INTrainingUpdateGradGammaBetaTest, in_training_update_grad_gamma_beta_reject_mixed_input_formats)
{
    ge::op::INTrainingUpdateGradGammaBeta op;

    auto gamma_tensor = create_desc_with_ori({4, 64, 3, 5}, ge::DT_FLOAT, ge::FORMAT_NCHW, {4, 64, 3, 5},
                                             ge::FORMAT_NCHW);
    auto beta_tensor = create_desc_with_ori({4, 64, 3, 5}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 64, 3, 5},
                                            ge::FORMAT_NHWC);
    op.UpdateInputDesc("res_gamma", gamma_tensor);
    op.UpdateInputDesc("res_beta", beta_tensor);

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(INTrainingUpdateGradGammaBetaTest, in_training_update_grad_gamma_beta_reject_null_infer_shape_context)
{
    auto* op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl("INTrainingUpdateGradGammaBeta");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_shape, nullptr);
    EXPECT_EQ(op_impl->infer_shape(nullptr), ge::GRAPH_FAILED);
}
