/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "../../../op_graph/batch_norm_v3_proto.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/common_error_codes.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "log/log.h"
#include "op_desc.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "array_ops.h"

class BatchNormV3Test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "BatchNormV3 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BatchNormV3 TearDown" << std::endl;
    }
};

// REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT().ATTR().OP_END_FACTORY_REG()

TEST_F(BatchNormV3Test, batchnorm_infer_shape_nchw)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({1, 64, 128, 128});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{1, 1}, {64, 64}, {128, 128}, {128, 128}};
    auto input_x_dtype = DT_FLOAT;
    // input weight info
    auto input_weight_shape = vector<int64_t>({64});
    std::vector<std::pair<int64_t, int64_t>> shape_range_weight = {{64, 64}};
    auto input_weight_dtype = DT_FLOAT;

    std::vector<int64_t> expected_output_shape = input_x_shape;
    std::vector<int64_t> expected_output1_shape = input_weight_shape;

    auto test_op = op::BatchNormV3("BatchNormV3");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, weight, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, bias, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(
        test_op, running_mean, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(
        test_op, running_var, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output0_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
    auto output1_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output1_shape);
}

TEST_F(BatchNormV3Test, batchnorm_infer_shape_ncdhw)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({1, 64, 64, 128, 128});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{1, 1}, {64, 64}, {64, 64}, {128, 128}, {128, 128}};
    auto input_x_dtype = DT_FLOAT;
    // input weight info
    auto input_weight_shape = vector<int64_t>({64});
    std::vector<std::pair<int64_t, int64_t>> shape_range_weight = {{64, 64}};
    auto input_weight_dtype = DT_FLOAT;

    std::vector<int64_t> expected_output_shape = input_x_shape;
    std::vector<int64_t> expected_output1_shape = input_weight_shape;

    auto test_op = op::BatchNormV3("BatchNormV3");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCDHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, weight, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, bias, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(
        test_op, running_mean, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(
        test_op, running_var, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output0_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
    auto output1_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output1_shape);
}

TEST_F(BatchNormV3Test, batchnorm_infer_shape_nhwc)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({1, 64, 128, 128});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{1, 1}, {64, 64}, {128, 128}, {128, 128}};
    auto input_x_dtype = DT_FLOAT;
    // input weight info
    auto input_weight_shape = vector<int64_t>({128});
    std::vector<std::pair<int64_t, int64_t>> shape_range_weight = {{128, 128}};
    auto input_weight_dtype = DT_FLOAT;

    std::vector<int64_t> expected_output_shape = input_x_shape;
    std::vector<int64_t> expected_output1_shape = input_weight_shape;

    auto test_op = op::BatchNormV3("BatchNormV3");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NHWC, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, weight, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, bias, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(
        test_op, running_mean, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(
        test_op, running_var, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output0_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
    auto output1_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output1_shape);
}

TEST_F(BatchNormV3Test, batchnorm_infer_shape_ndhwc)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({1, 64, 64, 128, 128});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{1, 1}, {64, 64}, {64, 64}, {128, 128}, {128, 128}};
    auto input_x_dtype = DT_FLOAT;
    // input weight info
    auto input_weight_shape = vector<int64_t>({128});
    std::vector<std::pair<int64_t, int64_t>> shape_range_weight = {{128, 128}};
    auto input_weight_dtype = DT_FLOAT;

    std::vector<int64_t> expected_output_shape = input_x_shape;
    std::vector<int64_t> expected_output1_shape = input_weight_shape;

    auto test_op = op::BatchNormV3("BatchNormV3");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NDHWC, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, weight, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, bias, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(
        test_op, running_mean, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(
        test_op, running_var, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output0_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
    auto output1_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output1_shape);
}
