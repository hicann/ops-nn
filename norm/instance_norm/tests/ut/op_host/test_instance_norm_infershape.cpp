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
#include "infershape_test_util.h"
#include "../../../op_graph/instance_norm_proto.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

class InstanceNormTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "InstanceNormTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "InstanceNormTest TearDown" << std::endl;
    }
};

TEST_F(InstanceNormTest, instance_norm_infer_shape_float_nchw_001)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({4, 3, 2, 2});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{-64, 64}, {-64, 64}, {-64, 64}, {-64, 64}};
    auto input_x_dtype = DT_FLOAT;
    // input gamma info
    auto input_gamma_shape = vector<int64_t>({3});
    std::vector<std::pair<int64_t, int64_t>> shape_range_gamma = {{-64, 64}};
    auto input_gamma_dtype = DT_FLOAT;

    std::vector<int64_t> expected_output_y_shape = input_x_shape;
    std::vector<int64_t> expected_output_mean_shape = vector<int64_t>({4, 3, 1, 1});

    auto test_op = op::InstanceNorm("InstanceNorm");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, gamma, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);
    TENSOR_INPUT_WITH_SHAPE(test_op, beta, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_y_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_y_shape);
    auto output_mean_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_output_mean_shape);
    auto output_var_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_output_mean_shape);
}

TEST_F(InstanceNormTest, instance_norm_infer_shape_float_ncdhw_002)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({4, 3, 2, 2, 2});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{-64, 64}, {-64, 64}, {-64, 64}, {-64, 64}, {-64, 64}};
    auto input_x_dtype = DT_FLOAT;
    // input gamma info
    auto input_gamma_shape = vector<int64_t>({3});
    std::vector<std::pair<int64_t, int64_t>> shape_range_gamma = {{64, 64}};
    auto input_gamma_dtype = DT_FLOAT;

    std::vector<int64_t> expected_output_y_shape = input_x_shape;
    std::vector<int64_t> expected_output_mean_shape = vector<int64_t>({4, 3, 1, 1, 1});

    auto test_op = op::InstanceNorm("InstanceNorm");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCDHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, gamma, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);
    TENSOR_INPUT_WITH_SHAPE(test_op, beta, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_y_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_y_shape);
    auto output_mean_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_output_mean_shape);
    auto output_var_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_output_mean_shape);
}

TEST_F(InstanceNormTest, instance_norm_infer_shape_float_nhwc_003)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({4, 2, 2, 3});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{-64, 64}, {-64, 64}, {-64, 64}, {-64, 64}};
    auto input_x_dtype = DT_FLOAT;
    // input gamma info
    auto input_gamma_shape = vector<int64_t>({3});
    std::vector<std::pair<int64_t, int64_t>> shape_range_gamma = {{-64, 64}};
    auto input_gamma_dtype = DT_FLOAT;

    std::vector<int64_t> expected_output_y_shape = input_x_shape;
    std::vector<int64_t> expected_output_mean_shape = vector<int64_t>({4, 1, 1, 3});

    auto test_op = op::InstanceNorm("InstanceNorm");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NHWC, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, gamma, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);
    TENSOR_INPUT_WITH_SHAPE(test_op, beta, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_y_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_y_shape);
    auto output_mean_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_output_mean_shape);
    auto output_var_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_output_mean_shape);
}

TEST_F(InstanceNormTest, instance_norm_infer_shape_float_ndhwc_004)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({4, 2, 2, 2, 3});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{-64, 64}, {-64, 64}, {-64, 64}, {-64, 64}, {-64, 64}};
    auto input_x_dtype = DT_FLOAT;
    // input gamma info
    auto input_gamma_shape = vector<int64_t>({3});
    std::vector<std::pair<int64_t, int64_t>> shape_range_gamma = {{-64, 64}};
    auto input_gamma_dtype = DT_FLOAT;

    std::vector<int64_t> expected_output_y_shape = input_x_shape;
    std::vector<int64_t> expected_output_mean_shape = vector<int64_t>({4, 1, 1, 1, 3});

    auto test_op = op::InstanceNorm("InstanceNorm");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NDHWC, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, gamma, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);
    TENSOR_INPUT_WITH_SHAPE(test_op, beta, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_y_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_y_shape);
    auto output_mean_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_output_mean_shape);
    auto output_var_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_output_mean_shape);
}

TEST_F(InstanceNormTest, instance_norm_infer_shape_float16_nchw_005)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({4, 3, 1024, 1024});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{-64, 64}, {-64, 64}, {-64, 64}, {-64, 64}};
    auto input_x_dtype = DT_FLOAT16;
    // input gamma info
    auto input_gamma_shape = vector<int64_t>({3});
    std::vector<std::pair<int64_t, int64_t>> shape_range_gamma = {{-64, 64}};
    auto input_gamma_dtype = DT_FLOAT16;

    std::vector<int64_t> expected_output_y_shape = input_x_shape;
    std::vector<int64_t> expected_output_mean_shape = vector<int64_t>({4, 3, 1, 1});

    auto test_op = op::InstanceNorm("InstanceNorm");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, gamma, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);
    TENSOR_INPUT_WITH_SHAPE(test_op, beta, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_y_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_y_shape);
    auto output_mean_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_output_mean_shape);
    auto output_var_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_output_mean_shape);
}

TEST_F(InstanceNormTest, instance_norm_infer_shape_float16_ncdhw_006)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({4, 3, 1024, 1024, 1024});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{-64, 64}, {-64, 64}, {-64, 64}, {-64, 64}, {-64, 64}};
    auto input_x_dtype = DT_FLOAT16;
    // input gamma info
    auto input_gamma_shape = vector<int64_t>({3});
    std::vector<std::pair<int64_t, int64_t>> shape_range_gamma = {{64, 64}};
    auto input_gamma_dtype = DT_FLOAT16;

    std::vector<int64_t> expected_output_y_shape = input_x_shape;
    std::vector<int64_t> expected_output_mean_shape = vector<int64_t>({4, 3, 1, 1, 1});

    auto test_op = op::InstanceNorm("InstanceNorm");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCDHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, gamma, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);
    TENSOR_INPUT_WITH_SHAPE(test_op, beta, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_y_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_y_shape);
    auto output_mean_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_output_mean_shape);
    auto output_var_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_output_mean_shape);
}

TEST_F(InstanceNormTest, instance_norm_infer_shape_float16_nhwc_007)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({4, 1024, 1024, 3});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{-64, 64}, {-64, 64}, {-64, 64}, {-64, 64}};
    auto input_x_dtype = DT_FLOAT16;
    // input gamma info
    auto input_gamma_shape = vector<int64_t>({3});
    std::vector<std::pair<int64_t, int64_t>> shape_range_gamma = {{-64, 64}};
    auto input_gamma_dtype = DT_FLOAT16;

    std::vector<int64_t> expected_output_y_shape = input_x_shape;
    std::vector<int64_t> expected_output_mean_shape = vector<int64_t>({4, 1, 1, 3});

    auto test_op = op::InstanceNorm("InstanceNorm");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NHWC, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, gamma, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);
    TENSOR_INPUT_WITH_SHAPE(test_op, beta, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_y_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_y_shape);
    auto output_mean_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_output_mean_shape);
    auto output_var_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_output_mean_shape);
}

TEST_F(InstanceNormTest, instance_norm_infer_shape_float16_ndhwc_008)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({4, 1024, 1024, 1024, 3});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{-64, 64}, {-64, 64}, {-64, 64}, {-64, 64}, {-64, 64}};
    auto input_x_dtype = DT_FLOAT16;
    // input gamma info
    auto input_gamma_shape = vector<int64_t>({3});
    std::vector<std::pair<int64_t, int64_t>> shape_range_gamma = {{-64, 64}};
    auto input_gamma_dtype = DT_FLOAT16;

    std::vector<int64_t> expected_output_y_shape = input_x_shape;
    std::vector<int64_t> expected_output_mean_shape = vector<int64_t>({4, 1, 1, 1, 3});

    auto test_op = op::InstanceNorm("InstanceNorm");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NDHWC, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, gamma, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);
    TENSOR_INPUT_WITH_SHAPE(test_op, beta, input_gamma_shape, input_gamma_dtype, FORMAT_ND, shape_range_gamma);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_y_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_y_shape);
    auto output_mean_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_output_mean_shape);
    auto output_var_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_output_mean_shape);
}

TEST_F(InstanceNormTest, instance_norm_infer_dtype_float16_float16)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950PR_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    ge::op::InstanceNorm op;
    op.UpdateInputDesc("x", create_desc({1, 1152, 64, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({1152}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({1152}, ge::DT_FLOAT16));
    op.SetAttr("data_format", "NCDHW");
    op.SetAttr("epsilon", static_cast<float>(1e-6));
    EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);

    auto output_dtype = op.GetOutputDesc("y").GetDataType();
    auto mean_dtype = op.GetOutputDesc("mean").GetDataType();
    auto var_dtype = op.GetOutputDesc("variance").GetDataType();

    EXPECT_EQ(output_dtype, ge::DT_FLOAT16);
    EXPECT_EQ(mean_dtype, ge::DT_FLOAT16);
    EXPECT_EQ(var_dtype, ge::DT_FLOAT16);
}

TEST_F(InstanceNormTest, instance_norm_infer_dtype_float16_float)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950PR_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    ge::op::InstanceNorm op;
    op.UpdateInputDesc("x", create_desc({1, 1152, 64, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({1152}, ge::DT_FLOAT));
    op.UpdateInputDesc("beta", create_desc({1152}, ge::DT_FLOAT));
    op.SetAttr("data_format", "NCDHW");
    op.SetAttr("epsilon", static_cast<float>(1e-6));
    EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);

    auto output_dtype = op.GetOutputDesc("y").GetDataType();
    auto mean_dtype = op.GetOutputDesc("mean").GetDataType();
    auto var_dtype = op.GetOutputDesc("variance").GetDataType();

    EXPECT_EQ(output_dtype, ge::DT_FLOAT16);
    EXPECT_EQ(mean_dtype, ge::DT_FLOAT);
    EXPECT_EQ(var_dtype, ge::DT_FLOAT);
}
