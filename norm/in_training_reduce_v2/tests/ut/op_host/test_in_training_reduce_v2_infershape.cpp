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
 * \file test_in_training_reduce_v2_infershape.cpp
 * \brief arch35 核心路径 UT —— InferShape / InferDataType
 *   契约（spec.yaml / DESIGN §6.2）：
 *     - InferShape：输入 x [N,C,H,W] → sum/square_sum shape 均为 [N,C,1,1]
 *                   （N,C 取自 x、空间轴置 1）；5D [N,C,D,H,W] → [N,C,1,1,1]。
 *     - InferDataType：输入 fp32/fp16 → 输出 sum/square_sum 恒 DT_FLOAT（不随输入变）。
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "../../../op_graph/in_training_reduce_v2_proto.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

class INTrainingReduceV2InferTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "INTrainingReduceV2InferTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "INTrainingReduceV2InferTest TearDown" << std::endl; }
};

// ---------------------------------------------------------------------------
// InferShape：4D NCHW 典型 shape [4,16,32,32] → sum/square_sum [4,16,1,1]
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2InferTest, infer_shape_float_nchw_4d_001)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({4, 16, 32, 32});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}};
    auto input_x_dtype = DT_FLOAT;

    std::vector<int64_t> expected_output_shape = vector<int64_t>({4, 16, 1, 1});

    auto test_op = op::INTrainingReduceV2("INTrainingReduceV2");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_sum_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_sum_desc.GetShape().GetDims(), expected_output_shape);
    auto output_square_sum_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_square_sum_desc.GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// InferShape：4D NCHW 保留维退化 N=1 / C=1（[1,1,8,8] → [1,1,1,1]）
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2InferTest, infer_shape_float_nchw_4d_keepdim_002)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({1, 1, 8, 8});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}};
    auto input_x_dtype = DT_FLOAT16;

    std::vector<int64_t> expected_output_shape = vector<int64_t>({1, 1, 1, 1});

    auto test_op = op::INTrainingReduceV2("INTrainingReduceV2");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_sum_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_sum_desc.GetShape().GetDims(), expected_output_shape);
    auto output_square_sum_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_square_sum_desc.GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// InferShape：5D NCDHW [N,C,D,H,W] → [N,C,1,1,1]
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2InferTest, infer_shape_float_ncdhw_5d_003)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({2, 3, 4, 5, 6});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}};
    auto input_x_dtype = DT_FLOAT;

    std::vector<int64_t> expected_output_shape = vector<int64_t>({2, 3, 1, 1, 1});

    auto test_op = op::INTrainingReduceV2("INTrainingReduceV2");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCDHW, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_sum_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_sum_desc.GetShape().GetDims(), expected_output_shape);
    auto output_square_sum_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_square_sum_desc.GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// InferDataType：输入 fp32 → 输出恒 DT_FLOAT
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2InferTest, infer_dtype_float32_to_float32)
{
    using namespace ge;
    auto test_op = op::INTrainingReduceV2("INTrainingReduceV2");
    test_op.UpdateInputDesc("x", create_desc({4, 16, 32, 32}, ge::DT_FLOAT));

    EXPECT_EQ(InferDataTypeTest(test_op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(test_op.GetOutputDesc("sum").GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(test_op.GetOutputDesc("square_sum").GetDataType(), ge::DT_FLOAT);
}

// ---------------------------------------------------------------------------
// InferDataType：输入 fp16 → 输出恒 DT_FLOAT（不随输入变）
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2InferTest, infer_dtype_float16_to_float32)
{
    using namespace ge;
    auto test_op = op::INTrainingReduceV2("INTrainingReduceV2");
    test_op.UpdateInputDesc("x", create_desc({4, 16, 32, 32}, ge::DT_FLOAT16));

    EXPECT_EQ(InferDataTypeTest(test_op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(test_op.GetOutputDesc("sum").GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(test_op.GetOutputDesc("square_sum").GetDataType(), ge::DT_FLOAT);
}

// ---------------------------------------------------------------------------
// InferShape：动态 shape -1（部分已知维度）
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2InferTest, infer_shape_dynamic_minus1_004)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({-1, -1, 32, 32});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{1, 16}, {1, 16}, {32, 32}, {32, 32}};
    auto input_x_dtype = DT_FLOAT;

    auto test_op = op::INTrainingReduceV2("INTrainingReduceV2");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NCHW, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_sum_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_sum_desc.GetShape().GetDimNum(), 4U);
    EXPECT_EQ(output_sum_desc.GetShape().GetDim(0), -1);
    EXPECT_EQ(output_sum_desc.GetShape().GetDim(1), -1);
    EXPECT_EQ(output_sum_desc.GetShape().GetDim(2), 1);
    EXPECT_EQ(output_sum_desc.GetShape().GetDim(3), 1);
}

// ---------------------------------------------------------------------------
// InferShape：动态 shape -2（UNKNOWN_RANK）
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2InferTest, infer_shape_dynamic_minus2_005)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({-2});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;
    auto input_x_dtype = DT_FLOAT;

    auto test_op = op::INTrainingReduceV2("INTrainingReduceV2");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_ND, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
}
