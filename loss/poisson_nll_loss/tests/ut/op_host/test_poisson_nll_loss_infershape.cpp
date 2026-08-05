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
 * \file test_poisson_nll_loss_infershape.cpp
 * \brief PoissonNllLoss infershape UT: reduction=none keeps input shape; sum/mean -> scalar.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "log/log.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "platform/platform_info.h"

#include "../../../op_graph/poisson_nll_loss_proto.h"

class PoissonNllLossInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "poisson_nll_loss Infershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "poisson_nll_loss Infershape TearDown" << std::endl; }
};

// Helper: build op with input_x/target of given shape+dtype, set attrs, run infershape,
// and assert the resulting loss shape equals expectedOutShape.
static void DoInferCase(const std::vector<int64_t>& inShapeDims, ge::DataType dtype, const std::string& reduction,
                        const std::vector<int64_t>& expectedOutShape)
{
    ge::op::PoissonNllLoss op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape(inShapeDims);
    tensorDesc.SetDataType(dtype);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    op.UpdateInputDesc("input_x", tensorDesc);
    op.UpdateInputDesc("target", tensorDesc);
    op.SetAttr("log_input", true);
    op.SetAttr("full", false);
    op.SetAttr("eps", static_cast<float>(1e-8));
    op.SetAttr("reduction", reduction);

    Runtime2TestParam param{{"log_input", "full", "eps", "reduction"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outDesc = op.GetOutputDesc(0);
    EXPECT_EQ(outDesc.GetShape().GetDims(), expectedOutShape);
}

// reduction=none: output keeps input shape.
TEST_F(PoissonNllLossInfershapeTest, none_fp32_keeps_input_shape)
{
    DoInferCase({32, 64}, ge::DT_FLOAT, "none", {32, 64});
}

TEST_F(PoissonNllLossInfershapeTest, none_fp16_keeps_input_shape_1d)
{
    DoInferCase({1024}, ge::DT_FLOAT16, "none", {1024});
}

// reduction=sum: output is scalar (rank 0).
TEST_F(PoissonNllLossInfershapeTest, sum_fp32_scalar) { DoInferCase({128, 128}, ge::DT_FLOAT, "sum", {}); }

// reduction=mean: output is scalar (rank 0).
TEST_F(PoissonNllLossInfershapeTest, mean_fp16_scalar) { DoInferCase({8, 16, 32}, ge::DT_FLOAT16, "mean", {}); }

TEST_F(PoissonNllLossInfershapeTest, mean_fp32_scalar_3d) { DoInferCase({4, 4, 4}, ge::DT_FLOAT, "mean", {}); }

// input_x and target with mismatched shapes must be rejected (aligns with ascend910b entry gate
// `operator.eq(shape_input, shape_target)`; non-broadcast operator).
TEST_F(PoissonNllLossInfershapeTest, mismatched_shape_rejected)
{
    ge::op::PoissonNllLoss op;
    ge::TensorDesc xDesc;
    xDesc.SetDataType(ge::DT_FLOAT);
    xDesc.SetShape(ge::Shape({32, 64}));
    xDesc.SetOriginShape(ge::Shape({32, 64}));
    ge::TensorDesc tDesc;
    tDesc.SetDataType(ge::DT_FLOAT);
    tDesc.SetShape(ge::Shape({32, 128})); // different from input_x
    tDesc.SetOriginShape(ge::Shape({32, 128}));

    op.UpdateInputDesc("input_x", xDesc);
    op.UpdateInputDesc("target", tDesc);
    op.SetAttr("log_input", true);
    op.SetAttr("full", false);
    op.SetAttr("eps", static_cast<float>(1e-8));
    op.SetAttr("reduction", "none");

    Runtime2TestParam param{{"log_input", "full", "eps", "reduction"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

// different rank must also be rejected.
TEST_F(PoissonNllLossInfershapeTest, mismatched_rank_rejected)
{
    ge::op::PoissonNllLoss op;
    ge::TensorDesc xDesc;
    xDesc.SetDataType(ge::DT_FLOAT16);
    xDesc.SetShape(ge::Shape({64, 64}));
    xDesc.SetOriginShape(ge::Shape({64, 64}));
    ge::TensorDesc tDesc;
    tDesc.SetDataType(ge::DT_FLOAT16);
    tDesc.SetShape(ge::Shape({64, 64, 2})); // rank 3 vs rank 2
    tDesc.SetOriginShape(ge::Shape({64, 64, 2}));

    op.UpdateInputDesc("input_x", xDesc);
    op.UpdateInputDesc("target", tDesc);
    op.SetAttr("log_input", false);
    op.SetAttr("full", false);
    op.SetAttr("eps", static_cast<float>(1e-8));
    op.SetAttr("reduction", "sum");

    Runtime2TestParam param{{"log_input", "full", "eps", "reduction"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

// -2 UNKNOWN_RANK：红线 R4 点名的必验项。infershape 用 IsUnknownRank 跳过 shape 相等校验，
// reduction=none 时透传 {-2}；sum/mean 恒为标量。
TEST_F(PoissonNllLossInfershapeTest, unknown_rank_none_passthrough)
{
    ge::op::PoissonNllLoss op;
    op.UpdateInputDesc("input_x", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("target", create_desc({-2}, ge::DT_FLOAT));
    op.SetAttr("log_input", true);
    op.SetAttr("full", false);
    op.SetAttr("eps", static_cast<float>(1e-8));
    op.SetAttr("reduction", "none");
    Runtime2TestParam param{{"log_input", "full", "eps", "reduction"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    std::vector<int64_t> expected{-2};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected);
}

TEST_F(PoissonNllLossInfershapeTest, unknown_rank_sum_is_scalar)
{
    ge::op::PoissonNllLoss op;
    op.UpdateInputDesc("input_x", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("target", create_desc({-2}, ge::DT_FLOAT));
    op.SetAttr("log_input", true);
    op.SetAttr("full", false);
    op.SetAttr("eps", static_cast<float>(1e-8));
    op.SetAttr("reduction", "sum");
    Runtime2TestParam param{{"log_input", "full", "eps", "reduction"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDimNum(), 0);
}
