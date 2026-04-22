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
#include <iostream>
#include <vector>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/fused_bias_leaky_relu_proto.h"
#include "platform/platform_info.h"

class FusedBiasLeakyReluInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedBiasLeakyRelu InferShape test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedBiasLeakyRelu InferShape test TearDown" << std::endl;
    }
};

TEST_F(FusedBiasLeakyReluInferShapeTest, infershape_test_case_1d)
{
    ge::op::FusedBiasLeakyRelu op;
    ge::TensorDesc tensor_x;
    ge::Shape shape({1024});
    tensor_x.SetDataType(ge::DT_FLOAT);
    tensor_x.SetShape(shape);
    tensor_x.SetOriginShape(shape);

    ge::TensorDesc tensor_bias;
    tensor_bias.SetDataType(ge::DT_FLOAT);
    tensor_bias.SetShape(shape);
    tensor_bias.SetOriginShape(shape);

    op.UpdateInputDesc("x", tensor_x);
    op.UpdateInputDesc("bias", tensor_bias);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    std::vector<int64_t> expected_y_shape = {1024};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(FusedBiasLeakyReluInferShapeTest, infershape_test_case_2d)
{
    ge::op::FusedBiasLeakyRelu op;
    ge::TensorDesc tensor_x;
    ge::Shape shape({2, 1024});
    tensor_x.SetDataType(ge::DT_FLOAT);
    tensor_x.SetShape(shape);
    tensor_x.SetOriginShape(shape);

    ge::TensorDesc tensor_bias;
    tensor_bias.SetDataType(ge::DT_FLOAT);
    tensor_bias.SetShape(shape);
    tensor_bias.SetOriginShape(shape);

    op.UpdateInputDesc("x", tensor_x);
    op.UpdateInputDesc("bias", tensor_bias);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    std::vector<int64_t> expected_y_shape = {2, 1024};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(FusedBiasLeakyReluInferShapeTest, infershape_test_case_4d)
{
    ge::op::FusedBiasLeakyRelu op;
    ge::TensorDesc tensor_x;
    ge::Shape shape({2, 4, 8, 16});
    tensor_x.SetDataType(ge::DT_FLOAT);
    tensor_x.SetShape(shape);
    tensor_x.SetOriginShape(shape);

    ge::TensorDesc tensor_bias;
    tensor_bias.SetDataType(ge::DT_FLOAT);
    tensor_bias.SetShape(shape);
    tensor_bias.SetOriginShape(shape);

    op.UpdateInputDesc("x", tensor_x);
    op.UpdateInputDesc("bias", tensor_bias);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    std::vector<int64_t> expected_y_shape = {2, 4, 8, 16};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(FusedBiasLeakyReluInferShapeTest, infershape_test_case_float16)
{
    ge::op::FusedBiasLeakyRelu op;
    ge::TensorDesc tensor_x;
    ge::Shape shape({8192});
    tensor_x.SetDataType(ge::DT_FLOAT16);
    tensor_x.SetShape(shape);
    tensor_x.SetOriginShape(shape);

    ge::TensorDesc tensor_bias;
    tensor_bias.SetDataType(ge::DT_FLOAT16);
    tensor_bias.SetShape(shape);
    tensor_bias.SetOriginShape(shape);

    op.UpdateInputDesc("x", tensor_x);
    op.UpdateInputDesc("bias", tensor_bias);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    std::vector<int64_t> expected_y_shape = {8192};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(FusedBiasLeakyReluInferShapeTest, infershape_test_case_empty)
{
    ge::op::FusedBiasLeakyRelu op;
    ge::TensorDesc tensor_x;
    ge::Shape shape({0});
    tensor_x.SetDataType(ge::DT_FLOAT);
    tensor_x.SetShape(shape);
    tensor_x.SetOriginShape(shape);

    ge::TensorDesc tensor_bias;
    tensor_bias.SetDataType(ge::DT_FLOAT);
    tensor_bias.SetShape(shape);
    tensor_bias.SetOriginShape(shape);

    op.UpdateInputDesc("x", tensor_x);
    op.UpdateInputDesc("bias", tensor_bias);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

TEST_F(FusedBiasLeakyReluInferShapeTest, ascend950_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950PR_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ge::op::FusedBiasLeakyRelu op;
    ge::TensorDesc tensor_x;
    ge::Shape shape({1, 64, 128});
    tensor_x.SetDataType(ge::DT_FLOAT);
    tensor_x.SetShape(shape);
    tensor_x.SetOriginShape(shape);

    ge::TensorDesc tensor_bias;
    tensor_bias.SetDataType(ge::DT_FLOAT);
    tensor_bias.SetShape(shape);
    tensor_bias.SetOriginShape(shape);

    op.UpdateInputDesc("x", tensor_x);
    op.UpdateInputDesc("bias", tensor_bias);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}