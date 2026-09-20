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
 * \file test_gn_training_update_infershape.cpp
 * \brief GNTrainingUpdate infershape UT：y <- x，batch_mean/batch_variance <- sum
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "../../../op_graph/gn_training_update_proto.h"

class GnTrainingUpdateInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GnTrainingUpdate Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GnTrainingUpdate Proto Test TearDown" << std::endl; }
};

// NCHW：x[2,4,2,3] fp16，统计量 [2,2,1,1,1]，G=2；y 同 x，统计量输出同 sum
TEST_F(GnTrainingUpdateInferShape, nchw_success)
{
    ge::op::GNTrainingUpdate op;
    op.UpdateInputDesc("x", create_desc({2, 4, 2, 3}, ge::DT_FLOAT16));
    op.UpdateInputDesc("sum", create_desc({2, 2, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("square_sum", create_desc({2, 2, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("scale", create_desc({1, 2, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("offset", create_desc({1, 2, 1, 1, 1}, ge::DT_FLOAT));
    op.SetAttr("num_groups", (int64_t)2);

    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({2, 4, 2, 3}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({2, 2, 1, 1, 1}));
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), std::vector<int64_t>({2, 2, 1, 1, 1}));
}

// NHWC：x[2,2,3,4]（C=4 末维），统计量 [2,1,1,2,1]
TEST_F(GnTrainingUpdateInferShape, nhwc_success)
{
    ge::op::GNTrainingUpdate op;
    op.UpdateInputDesc("x", create_desc({2, 2, 3, 4}, ge::DT_FLOAT));
    op.UpdateInputDesc("sum", create_desc({2, 1, 1, 2, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("square_sum", create_desc({2, 1, 1, 2, 1}, ge::DT_FLOAT));
    op.SetAttr("num_groups", (int64_t)2);

    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({2, 2, 3, 4}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({2, 1, 1, 2, 1}));
}

// 无仿射（scale/offset/mean/variance 全缺席）
TEST_F(GnTrainingUpdateInferShape, no_affine_success)
{
    ge::op::GNTrainingUpdate op;
    op.UpdateInputDesc("x", create_desc({1, 8, 4, 4}, ge::DT_FLOAT));
    op.UpdateInputDesc("sum", create_desc({1, 4, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("square_sum", create_desc({1, 4, 1, 1, 1}, ge::DT_FLOAT));
    op.SetAttr("num_groups", (int64_t)4);

    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({1, 8, 4, 4}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({1, 4, 1, 1, 1}));
}

// GEIR.03：x 未知 rank（{-2}）时，sum 的具体非法形状（rank=4≠5）仍应被拒，
// 不能借动态支持整块放行
TEST_F(GnTrainingUpdateInferShape, unknown_rank_x_rejects_illegal_sum)
{
    ge::op::GNTrainingUpdate op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("sum", create_desc({2, 2, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("square_sum", create_desc({2, 2, 1, 1}, ge::DT_FLOAT));
    op.SetAttr("num_groups", (int64_t)2);

    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

// GEIR.03：x 未知 rank + sum 合法 → 通过，统计量输出取 sum 形状
TEST_F(GnTrainingUpdateInferShape, unknown_rank_x_valid_sum_success)
{
    ge::op::GNTrainingUpdate op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("sum", create_desc({2, 2, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("square_sum", create_desc({2, 2, 1, 1, 1}, ge::DT_FLOAT));
    op.SetAttr("num_groups", (int64_t)2);

    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({2, 2, 1, 1, 1}));
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), std::vector<int64_t>({2, 2, 1, 1, 1}));
}
