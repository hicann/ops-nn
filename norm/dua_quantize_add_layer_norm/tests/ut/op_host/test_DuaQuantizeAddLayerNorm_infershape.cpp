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
#include "ut_op_common.h"
#include "../../../op_graph/dua_quantize_add_layer_norm_proto.h"

class DuaQuantizeAddLayerNorm : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "DuaQuantizeAddLayerNorm infershape Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DuaQuantizeAddLayerNorm infershape Test TearDown" << std::endl;
    }
};

TEST_F(DuaQuantizeAddLayerNorm, DuaQuantizeAddLayerNorm_infershape_case_0)
{
    ge::op::DuaQuantizeAddLayerNorm op;
    op.UpdateInputDesc("x1", create_desc({4, 1, 8}, ge::DT_BF16));
    op.UpdateInputDesc("x2", create_desc({4, 1, 8}, ge::DT_BF16));
    op.UpdateInputDesc("gamma", create_desc({8}, ge::DT_BF16));
    op.UpdateInputDesc("beta", create_desc({8}, ge::DT_BF16));
    op.UpdateInputDesc("bias", create_desc({8}, ge::DT_BF16));
    op.UpdateInputDesc("scales1", create_desc({8}, ge::DT_BF16));
    op.UpdateInputDesc("scales2", create_desc({8}, ge::DT_BF16));
    op.UpdateInputDesc("zero_points1", create_desc({8}, ge::DT_BF16));
    op.UpdateInputDesc("zero_points2", create_desc({8}, ge::DT_BF16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_x_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y1_shape = {4, 1, 8};
    std::vector<int64_t> expected_y2_shape = {4, 1, 8};
    std::vector<int64_t> expected_x_shape = {4, 1, 8};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y1_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y2_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_x_shape);
}
