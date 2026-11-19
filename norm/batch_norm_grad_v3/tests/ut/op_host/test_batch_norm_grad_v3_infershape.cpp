/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../../../op_graph/batch_norm_grad_v3_proto.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

class BatchNormGradV3Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BatchNormGradV3 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BatchNormGradV3 TearDown" << std::endl; }
};

namespace {
void CheckEmptyOutputShape(const std::vector<int64_t>& dyShape, int64_t channels, ge::Format format, int training,
                           const std::vector<int64_t>& expectedParams, ge::DataType dtype = ge::DT_FLOAT)
{
    auto testOp = ge::op::BatchNormGradV3("BatchNormGradV3");
    if (training >= 0) {
        testOp.set_attr_is_training(training != 0);
    }
    const std::vector<int64_t> weightShape = {channels};
    const std::vector<std::pair<int64_t, int64_t>> shapeRange;
    TENSOR_INPUT_WITH_SHAPE(testOp, dy, dyShape, dtype, format, shapeRange);
    TENSOR_INPUT_WITH_SHAPE(testOp, x, dyShape, dtype, format, shapeRange);
    TENSOR_INPUT_WITH_SHAPE(testOp, weight, weightShape, dtype, ge::FORMAT_ND, shapeRange);
    TENSOR_INPUT_WITH_SHAPE(testOp, running_mean, weightShape, dtype, ge::FORMAT_ND, shapeRange);
    TENSOR_INPUT_WITH_SHAPE(testOp, running_var, weightShape, dtype, ge::FORMAT_ND, shapeRange);
    TENSOR_INPUT_WITH_SHAPE(testOp, save_mean, weightShape, ge::DT_FLOAT, ge::FORMAT_ND, shapeRange);
    TENSOR_INPUT_WITH_SHAPE(testOp, save_rstd, weightShape, ge::DT_FLOAT, ge::FORMAT_ND, shapeRange);
    const Runtime2TestParam param{{"is_training"}, {}, {}};
    ASSERT_EQ(InferShapeTest(testOp, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(testOp.GetOutputDesc(0).GetShape().GetDims(), dyShape);
    EXPECT_EQ(testOp.GetOutputDesc(1).GetShape().GetDims(), expectedParams);
    EXPECT_EQ(testOp.GetOutputDesc(2).GetShape().GetDims(), expectedParams);
}

const std::vector<std::pair<ge::Format, std::vector<int64_t>>> emptyTestLayouts = {{ge::FORMAT_NCHW, {2, 3, 4, 4}},
                                                                                   {ge::FORMAT_NHWC, {2, 4, 4, 3}},
                                                                                   {ge::FORMAT_NCDHW, {2, 3, 2, 4, 4}},
                                                                                   {ge::FORMAT_NDHWC, {2, 2, 4, 4, 3}}};
} // namespace

TEST_F(BatchNormGradV3Test, EmptyInferenceMakesAllOutputsEmpty)
{
    for (const auto& layout : emptyTestLayouts) {
        const size_t channelAxis = layout.first == ge::FORMAT_NHWC || layout.first == ge::FORMAT_NDHWC ?
                                       layout.second.size() - 1 :
                                       1;
        for (size_t axis = 0; axis < layout.second.size(); ++axis) {
            auto shape = layout.second;
            shape[axis] = 0;
            for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
                SCOPED_TRACE(testing::Message()
                             << "format=" << layout.first << " empty_axis=" << axis << " dtype=" << dtype);
                CheckEmptyOutputShape(shape, shape[channelAxis], layout.first, 0, {0}, dtype);
            }
        }
    }
}

TEST_F(BatchNormGradV3Test, EmptyTrainingPreservesReductionOutputs)
{
    for (const auto& layout : emptyTestLayouts) {
        const size_t channelAxis = layout.first == ge::FORMAT_NHWC || layout.first == ge::FORMAT_NDHWC ?
                                       layout.second.size() - 1 :
                                       1;
        for (size_t axis = 0; axis < layout.second.size(); ++axis) {
            auto shape = layout.second;
            shape[axis] = 0;
            // 归约轴为空时参数梯度仍为 [C]；只有 C 本身为 0 时参数梯度才为空。
            // 同时验证未提供 is_training 时仍使用默认训练语义。
            for (int training : {-1, 1}) {
                SCOPED_TRACE(testing::Message()
                             << "format=" << layout.first << " empty_axis=" << axis << " training=" << training);
                CheckEmptyOutputShape(shape, shape[channelAxis], layout.first, training, {shape[channelAxis]});
            }
        }
    }
}

TEST_F(BatchNormGradV3Test, NonemptyInferencePreservesParameterShapes)
{
    for (const auto& layout : emptyTestLayouts) {
        CheckEmptyOutputShape(layout.second, 3, layout.first, 0, {3});
    }
}

TEST_F(BatchNormGradV3Test, UnknownDimensionsDoNotImplyEmptyOutputs)
{
    CheckEmptyOutputShape({-1, 3, 4, 4}, 3, ge::FORMAT_NCHW, 0, {3});
    CheckEmptyOutputShape({-1, -1, -1, -1}, -1, ge::FORMAT_NCHW, 0, {-1});
    CheckEmptyOutputShape({-2}, 3, ge::FORMAT_NCHW, 0, {3});
    CheckEmptyOutputShape({-1, 3, 0, 4}, 3, ge::FORMAT_NCHW, 0, {0});
    CheckEmptyOutputShape({-1, 3, 0, 4}, 3, ge::FORMAT_NCHW, 1, {3});
}

TEST_F(BatchNormGradV3Test, EmptyShapeDoesNotRequireDimensionMultiplication)
{
    CheckEmptyOutputShape({0, 3, int64_t{1} << 62, 4}, 3, ge::FORMAT_NCHW, 0, {0});
    CheckEmptyOutputShape({0, 3, int64_t{1} << 62, 4}, 3, ge::FORMAT_NCHW, 1, {3});
}

// REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT().ATTR().OP_END_FACTORY_REG()

TEST_F(BatchNormGradV3Test, batchnorm_infer_shape_nchw)
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

    auto test_op = op::BatchNormGradV3("BatchNormGradV3");
    TENSOR_INPUT_WITH_SHAPE(test_op, dy, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, weight, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, running_mean, input_weight_shape, input_weight_dtype, FORMAT_ND,
                            shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, running_var, input_weight_shape, input_weight_dtype, FORMAT_ND,
                            shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, save_mean, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, save_rstd, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output0_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
    auto output1_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output1_shape);
    auto output2_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output2_desc.GetShape().GetDims(), expected_output1_shape);
}

TEST_F(BatchNormGradV3Test, batchnorm_infer_shape_ncdhw)
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

    auto test_op = op::BatchNormGradV3("BatchNormGradV3");
    TENSOR_INPUT_WITH_SHAPE(test_op, dy, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, weight, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, running_mean, input_weight_shape, input_weight_dtype, FORMAT_ND,
                            shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, running_var, input_weight_shape, input_weight_dtype, FORMAT_ND,
                            shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, save_mean, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, save_rstd, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output0_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
    auto output1_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output1_shape);
    auto output2_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output2_desc.GetShape().GetDims(), expected_output1_shape);
}

TEST_F(BatchNormGradV3Test, batchnorm_infer_shape_nhwc)
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

    auto test_op = op::BatchNormGradV3("BatchNormGradV3");
    TENSOR_INPUT_WITH_SHAPE(test_op, dy, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, weight, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, running_mean, input_weight_shape, input_weight_dtype, FORMAT_ND,
                            shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, running_var, input_weight_shape, input_weight_dtype, FORMAT_ND,
                            shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, save_mean, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, save_rstd, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output0_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
    auto output1_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output1_shape);
    auto output2_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output2_desc.GetShape().GetDims(), expected_output1_shape);
}

TEST_F(BatchNormGradV3Test, batchnorm_infer_shape_ndhwc)
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

    auto test_op = op::BatchNormGradV3("BatchNormGradV3");
    TENSOR_INPUT_WITH_SHAPE(test_op, dy, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);
    TENSOR_INPUT_WITH_SHAPE(test_op, weight, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, running_mean, input_weight_shape, input_weight_dtype, FORMAT_ND,
                            shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, running_var, input_weight_shape, input_weight_dtype, FORMAT_ND,
                            shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, save_mean, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);
    TENSOR_INPUT_WITH_SHAPE(test_op, save_rstd, input_weight_shape, input_weight_dtype, FORMAT_ND, shape_range_weight);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output0_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
    auto output1_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output1_shape);
    auto output2_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output2_desc.GetShape().GetDims(), expected_output1_shape);
}
