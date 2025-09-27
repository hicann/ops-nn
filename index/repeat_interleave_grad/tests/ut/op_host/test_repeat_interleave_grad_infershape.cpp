/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave_grad_proto.h
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_graph/repeat_interleave_grad_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"

class RepeatInterleaveGrad : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "RepeatInterleaveGrad SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RepeatInterleaveGrad TearDown" << std::endl;
    }
};

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test)
{
    ge::op::RepeatInterleaveGrad op;
    auto tensor_desc_x = create_desc_shape_range(
        {-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, {{2, 10}, {3, 10}, {4, 10}});

    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
    constDesc.SetSize(1 * sizeof(int32_t));
    constTensor.SetTensorDesc(constDesc);
    int32_t constData[1] = {2};
    constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
    auto repeats = ge::op::Constant().set_attr_value(constTensor);
    op.set_input_repeats(repeats);
    auto desc = op.GetInputDesc("repeats");
    desc.SetDataType(ge::DT_INT32);
    op.UpdateInputDesc("repeats", desc);

    op.UpdateInputDesc("y_grad", tensor_desc_x);
    op.SetAttr("axis", 0);
    std::vector<int64_t> expected_output_shape = {-2};
    // runtime 2.0
    Runtime2TestParam rt_param;
    rt_param.attrs = {"axis"};
    rt_param.input_const = {false, true};
    EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("x_grad");
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_2)
{
    ge::op::RepeatInterleaveGrad op;

    auto tensor_desc_x = create_desc_shape_range(
        {-1, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, {{2, 10}, {3, 10}, {4, 10}});

    op.UpdateInputDesc("y_grad", tensor_desc_x);
    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
    constDesc.SetSize(1 * sizeof(int32_t));
    constTensor.SetTensorDesc(constDesc);
    int32_t constData[1] = {2};
    constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
    auto repeats = ge::op::Constant().set_attr_value(constTensor);
    op.set_input_repeats(repeats);
    auto desc = op.GetInputDesc("repeats");
    desc.SetDataType(ge::DT_INT32);
    op.UpdateInputDesc("repeats", desc);

    op.SetAttr("axis", 0);
    std::vector<int64_t> expected_output_shape = {-2};
    // runtime 2.0
    Runtime2TestParam rt_param;
    rt_param.attrs = {"axis"};
    rt_param.input_const = {false, true};
    EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("x_grad");
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_3)
{
    ge::op::RepeatInterleaveGrad op;

    auto tensor_desc_x = create_desc_shape_range(
        {4, 32, 64, 32}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 32, 64, 32}, ge::FORMAT_ND, {{2, 10}, {3, 10}, {4, 10}});
    op.UpdateInputDesc("y_grad", tensor_desc_x);

    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
    constDesc.SetSize(1 * sizeof(int32_t));
    constTensor.SetTensorDesc(constDesc);
    int32_t constData[1] = {2};
    constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
    auto repeats = ge::op::Constant().set_attr_value(constTensor);
    op.set_input_repeats(repeats);
    auto desc = op.GetInputDesc("repeats");
    desc.SetDataType(ge::DT_INT32);
    op.UpdateInputDesc("repeats", desc);

    op.SetAttr("axis", 0);
    std::vector<int64_t> expected_output_shape = {2, 32, 64, 32};
    // runtime 2.0
    Runtime2TestParam rt_param;
    rt_param.attrs = {"axis"};
    rt_param.input_const = {false, true};
    EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("x_grad");
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_4)
{
    ge::op::RepeatInterleaveGrad op;

    auto tensor_desc_x = create_desc_shape_range(
        {4, 32, 32, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 32, 32, 64}, ge::FORMAT_ND, {{2, 10}, {3, 10}, {4, 10}});

    op.UpdateInputDesc("y_grad", tensor_desc_x);

    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT64);
    constDesc.SetSize(1 * sizeof(int64_t));
    constTensor.SetTensorDesc(constDesc);
    int64_t constData[1] = {2};
    constTensor.SetData((uint8_t*)constData, 1 * sizeof(int64_t));
    auto repeats = ge::op::Constant().set_attr_value(constTensor);
    op.set_input_repeats(repeats);
    auto desc = op.GetInputDesc("repeats");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("repeats", desc);

    op.SetAttr("axis", -3);
    std::vector<int64_t> expected_output_shape = {4, 16, 32, 64};
    // runtime 2.0
    Runtime2TestParam rt_param;
    rt_param.attrs = {"axis"};
    rt_param.input_const = {false, true};
    EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("x_grad");
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_5)
{
    ge::op::RepeatInterleaveGrad op;

    auto tensor_desc_x = create_desc_shape_range(
        {8, 32, 32, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {8, 32, 32, 64}, ge::FORMAT_ND, {{2, 10}, {3, 10}, {4, 10}});

    op.UpdateInputDesc("y_grad", tensor_desc_x);

    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
    constDesc.SetSize(4 * sizeof(int32_t));
    constTensor.SetTensorDesc(constDesc);
    int32_t constData[4] = {2, 1, 3, 2};
    constTensor.SetData((uint8_t*)constData, 4 * sizeof(int32_t));
    auto repeats = ge::op::Constant().set_attr_value(constTensor);
    op.set_input_repeats(repeats);
    auto desc = op.GetInputDesc("repeats");
    desc.SetDataType(ge::DT_INT32);
    op.UpdateInputDesc("repeats", desc);

    op.SetAttr("axis", 0);
    std::vector<int64_t> expected_output_shape = {4, 32, 32, 64};
    // runtime 2.0
    Runtime2TestParam rt_param;
    rt_param.attrs = {"axis"};
    rt_param.input_const = {false, true};
    EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("x_grad");
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_dy1)
{
    ge::op::RepeatInterleaveGrad op;
    std::vector<int64_t> input_shape = {-2};
    std::vector<int64_t> expect_shape = {-2};
    TENSOR_INPUT_WITH_SHAPE(op, y_grad, input_shape, DT_FLOAT, FORMAT_ND, {});

    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
    constDesc.SetSize(1 * sizeof(int32_t));
    constTensor.SetTensorDesc(constDesc);
    int32_t constData[1] = {2};
    constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
    auto repeats = ge::op::Constant().set_attr_value(constTensor);
    op.set_input_repeats(repeats);
    auto desc = op.GetInputDesc("repeats");
    desc.SetDataType(ge::DT_INT32);
    op.UpdateInputDesc("repeats", desc);

    op.SetAttr("axis", 0);
    // runtime 2.0
    Runtime2TestParam rt_param;
    rt_param.attrs = {"axis"};
    rt_param.input_const = {false, true};
    EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("x_grad");
    EXPECT_EQ(output_desc.GetShape().GetDims(), expect_shape);
}