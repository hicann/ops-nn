/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/situ_mx_quant_proto.h"

namespace {
class SituMxQuantTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SituMxQuantTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SituMxQuantTest TearDown" << std::endl; }
};

TEST_F(SituMxQuantTest, SituMxQuant_infershape_case_0_bf16_e4m3)
{
    ge::op::SituMxQuant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({8, 128, 8192});
    xDesc.SetDataType(ge::DT_BF16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{{"beta", "linear_beta", "activate_left", "axis", "dst_type", "round_mode"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc("y");
    auto outputScale = op.GetOutputDesc("y_scale");
    std::vector<int64_t> expectedYShape = {8, 128, 4096};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

TEST_F(SituMxQuantTest, SituMxQuant_infershape_case_1_bf16_e5m2)
{
    ge::op::SituMxQuant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({4, 64, 2048});
    xDesc.SetDataType(ge::DT_BF16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{{"beta", "linear_beta", "activate_left", "axis", "dst_type", "round_mode"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc("y");
    std::vector<int64_t> expectedYShape = {4, 64, 1024};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

TEST_F(SituMxQuantTest, SituMxQuant_infershape_case_fp16_e4m3)
{
    ge::op::SituMxQuant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({8, 128, 8192});
    xDesc.SetDataType(ge::DT_FLOAT16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{{"beta", "linear_beta", "activate_left", "axis", "dst_type", "round_mode"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc("y");
    auto outputScale = op.GetOutputDesc("y_scale");
    std::vector<int64_t> expectedYShape = {8, 128, 4096};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

TEST_F(SituMxQuantTest, SituMxQuant_infershape_case_bf16_fp4_e2m1)
{
    ge::op::SituMxQuant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({8, 128, 8192});
    xDesc.SetDataType(ge::DT_BF16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{{"beta", "linear_beta", "activate_left", "axis", "dst_type", "round_mode"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc("y");
    auto outputScale = op.GetOutputDesc("y_scale");
    std::vector<int64_t> expectedYShape = {8, 128, 4096};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

TEST_F(SituMxQuantTest, SituMxQuant_infershape_case_2d)
{
    ge::op::SituMxQuant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({4096, 4096});
    xDesc.SetDataType(ge::DT_BF16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{{"beta", "linear_beta", "activate_left", "axis", "dst_type", "round_mode"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc("y");
    std::vector<int64_t> expectedYShape = {4096, 2048};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

TEST_F(SituMxQuantTest, SituMxQuant_infershape_dynamic_shape)
{
    ge::op::SituMxQuant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({-2});
    xDesc.SetDataType(ge::DT_BF16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{{"beta", "linear_beta", "activate_left", "axis", "dst_type", "round_mode"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc("y");
    auto outputScale = op.GetOutputDesc("y_scale");
    std::vector<int64_t> expectedYShape = {-2};
    std::vector<int64_t> expectedScaleShape = {-2};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(SituMxQuantTest, SituMxQuant_infershape_error_invalid_dim)
{
    ge::op::SituMxQuant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({4, 64, 1023});
    xDesc.SetDataType(ge::DT_BF16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{{"beta", "linear_beta", "activate_left", "axis", "dst_type", "round_mode"}};
    EXPECT_NE(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
}
} // namespace
