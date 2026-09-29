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
 * \file test_group_norm_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "../../../op_graph/group_norm_proto.h"

class GroupNorm : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupNorm Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupNorm Proto Test TearDown" << std::endl; }
};

TEST_F(GroupNorm, group_norm_infershape_test_1)
{
    ge::op::GroupNorm op;
    op.UpdateInputDesc("x", create_desc({8, 16, 15, 15}, ge::DT_FLOAT16));
    op.SetAttr("num_groups", 8);
    std::vector<int64_t> expected_output_shape = {8, 16, 15, 15};
    std::vector<int64_t> expected_mean_shape = {8, 8};
    std::vector<int64_t> expected_variance_shape = {8, 8};

    // run rt 2.0
    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
    auto output1_desc = op.GetOutputDesc(1);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_mean_shape);
    auto output2_desc = op.GetOutputDesc(2);
    EXPECT_EQ(output2_desc.GetShape().GetDims(), expected_variance_shape);
}

TEST_F(GroupNorm, group_norm_infershape_test_2)
{
    ge::op::GroupNorm op;
    op.UpdateInputDesc("x", create_desc({-1, -1, -1, -1}, ge::DT_FLOAT16));
    op.SetAttr("num_groups", 8);
    std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1};
    std::vector<int64_t> expected_mean_shape = {-1, 8};
    std::vector<int64_t> expected_variance_shape = {-1, 8};

    // run rt 2.0
    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
    auto output1_desc = op.GetOutputDesc(1);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_mean_shape);
    auto output2_desc = op.GetOutputDesc(2);
    EXPECT_EQ(output2_desc.GetShape().GetDims(), expected_variance_shape);
}

TEST_F(GroupNorm, group_norm_inferdtype_david_test1)
{
    ge::op::GroupNorm op;
    op.UpdateInputDesc("x", create_desc({1, 1152, 64, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({1152}, ge::DT_FLOAT));
    op.UpdateInputDesc("beta", create_desc({1152}, ge::DT_FLOAT));

    auto ret = InferDataTypeTest(op);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_dtype = op.GetOutputDesc("y").GetDataType();
    auto mean_dtype = op.GetOutputDesc("mean").GetDataType();
    auto variance_dtype = op.GetOutputDesc("variance").GetDataType();

    EXPECT_EQ(output_dtype, ge::DT_FLOAT16);
    EXPECT_EQ(mean_dtype, ge::DT_FLOAT16);
    EXPECT_EQ(variance_dtype, ge::DT_FLOAT16);
}

TEST_F(GroupNorm, group_norm_infershape_unknown_rank)
{
    ge::op::GroupNorm op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT));
    op.SetAttr("num_groups", 4);

    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc("y").GetShape().GetDims(), std::vector<int64_t>({-2}));
    EXPECT_EQ(op.GetOutputDesc("mean").GetShape().GetDims(), std::vector<int64_t>({-2}));
    EXPECT_EQ(op.GetOutputDesc("variance").GetShape().GetDims(), std::vector<int64_t>({-2}));
}

TEST_F(GroupNorm, group_norm_infershape_rejects_rank_one)
{
    ge::op::GroupNorm op;
    op.UpdateInputDesc("x", create_desc({16}, ge::DT_FLOAT));
    op.SetAttr("num_groups", 4);

    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}
