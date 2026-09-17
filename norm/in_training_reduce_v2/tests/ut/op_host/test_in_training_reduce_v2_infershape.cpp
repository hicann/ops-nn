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
 * \brief arch35 核心路径 UT —— InferShape
 *   契约（spec.yaml / DESIGN §6.2）：
 *     - InferShape：输入 x [N,C,H,W] → sum/square_sum shape 均为 [N,C,1,1]
 *                   （N,C 取自 x、空间轴置 1）；5D [N,C,D,H,W] → [N,C,1,1,1]。
 *     - C 轴位置由 origin format 决定：channel-first（NCHW/NCDHW/ND）在 dim 1，
 *       channel-last（NHWC/NDHWC）在最后一维。def.cpp 的 Format 列声明的是
 *       storage format，约束不到 origin format —— 后者由用户网络决定（TF 来源
 *       天然是 NHWC），GE 会插 TransData 转排布，但 InferShape 跑在其之前。
 *
 *   InferDataType（输出恒 DT_FLOAT）和静态图 legacy V1 bridge 位于
 *   op_graph/in_training_reduce_v2_graph_infer.cpp，由 op_graph UT 与安装后的
 *   GEIR 用例验证；本文件只验证 runtime2.0 InferShape。
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
    auto output_square_sum_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_square_sum_desc.GetShape().GetDims(), output_sum_desc.GetShape().GetDims());
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
    const std::vector<int64_t> expected_output_shape = {-2};
    auto output_sum_desc = test_op.GetOutputDesc(0);
    auto output_square_sum_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_sum_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_square_sum_desc.GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// InferShape：channel-last origin format —— NHWC [4,32,32,16] → [4,1,1,16]
//   C 在最后一维而非 dim 1；若按 channel-first 硬编码会错推成 [4,32,1,1]。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2InferTest, infer_shape_float_nhwc_4d_channel_last_006)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({4, 32, 32, 16});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}};
    auto input_x_dtype = DT_FLOAT;

    std::vector<int64_t> expected_output_shape = vector<int64_t>({4, 1, 1, 16});

    auto test_op = op::INTrainingReduceV2("INTrainingReduceV2");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NHWC, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_sum_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_sum_desc.GetShape().GetDims(), expected_output_shape);
    auto output_square_sum_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_square_sum_desc.GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// InferShape：channel-last origin format —— NDHWC [2,4,8,8,16] → [2,1,1,1,16]
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2InferTest, infer_shape_float16_ndhwc_5d_channel_last_007)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({2, 4, 8, 8, 16});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}};
    auto input_x_dtype = DT_FLOAT16;

    std::vector<int64_t> expected_output_shape = vector<int64_t>({2, 1, 1, 1, 16});

    auto test_op = op::INTrainingReduceV2("INTrainingReduceV2");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NDHWC, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_sum_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_sum_desc.GetShape().GetDims(), expected_output_shape);
    auto output_square_sum_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output_square_sum_desc.GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// InferShape：channel-last + 动态 shape -1 —— NHWC [-1,32,32,-1] → [-1,1,1,-1]
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2InferTest, infer_shape_nhwc_dynamic_minus1_008)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({-1, 32, 32, -1});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{1, 16}, {32, 32}, {32, 32}, {1, 16}};
    auto input_x_dtype = DT_FLOAT;

    auto test_op = op::INTrainingReduceV2("INTrainingReduceV2");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_NHWC, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    auto output_sum_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output_sum_desc.GetShape().GetDimNum(), 4U);
    EXPECT_EQ(output_sum_desc.GetShape().GetDim(0), -1);
    EXPECT_EQ(output_sum_desc.GetShape().GetDim(1), 1);
    EXPECT_EQ(output_sum_desc.GetShape().GetDim(2), 1);
    EXPECT_EQ(output_sum_desc.GetShape().GetDim(3), -1);
}

// ---------------------------------------------------------------------------
// Runtime2.0 InferShape：ND 是本仓显式支持的 channel-first 布局，不能落入
// built-in 历史实现的“非 NCHW/NCDHW 即 channel-last”兜底分支。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2InferTest, infer_shape_nd_channel_first_009)
{
    using namespace ge;
    const std::vector<int64_t> inputShape = {2, 3, 5, 7};
    const std::vector<std::pair<int64_t, int64_t>> shapeRange(inputShape.size(), {-1, -1});
    const std::vector<int64_t> expectedOutputShape = {2, 3, 1, 1};

    auto testOp = op::INTrainingReduceV2("INTrainingReduceV2");
    TENSOR_INPUT_WITH_SHAPE(testOp, x, inputShape, DT_FLOAT, FORMAT_ND, shapeRange);

    ASSERT_EQ(InferShapeTest(testOp), ge::GRAPH_SUCCESS);
    EXPECT_EQ(testOp.GetOutputDesc(0).GetShape().GetDims(), expectedOutputShape);
    EXPECT_EQ(testOp.GetOutputDesc(1).GetShape().GetDims(), expectedOutputShape);
}

TEST_F(INTrainingReduceV2InferTest, infer_shape_rejects_nd_rank_below_two_010)
{
    using namespace ge;
    const std::vector<int64_t> inputShape = {8};
    const std::vector<std::pair<int64_t, int64_t>> shapeRange(inputShape.size(), {-1, -1});
    auto testOp = op::INTrainingReduceV2("INTrainingReduceV2");
    TENSOR_INPUT_WITH_SHAPE(testOp, x, inputShape, DT_FLOAT, FORMAT_ND, shapeRange);
    EXPECT_EQ(InferShapeTest(testOp), ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2InferTest, infer_shape_rejects_nd_rank_above_eight_011)
{
    using namespace ge;
    const std::vector<int64_t> inputShape(9, 1);
    const std::vector<std::pair<int64_t, int64_t>> shapeRange(inputShape.size(), {-1, -1});
    auto testOp = op::INTrainingReduceV2("INTrainingReduceV2");
    TENSOR_INPUT_WITH_SHAPE(testOp, x, inputShape, DT_FLOAT16, FORMAT_ND, shapeRange);
    EXPECT_EQ(InferShapeTest(testOp), ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2InferTest, infer_shape_rejects_nchw_non_four_dimensional_012)
{
    using namespace ge;
    const std::vector<int64_t> inputShape = {2, 3, 5};
    const std::vector<std::pair<int64_t, int64_t>> shapeRange(inputShape.size(), {-1, -1});
    auto testOp = op::INTrainingReduceV2("INTrainingReduceV2");
    TENSOR_INPUT_WITH_SHAPE(testOp, x, inputShape, DT_FLOAT, FORMAT_NCHW, shapeRange);
    EXPECT_EQ(InferShapeTest(testOp), ge::GRAPH_FAILED);
}
