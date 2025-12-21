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
#include "../../../op_graph/adaptive_avg_pool3d_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"

using namespace ge;

class AdaptiveAvgPool3dInferShapeTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveAvgPool3d InferShape Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveAvgPool3d InferShape Test TearDown" << std::endl;
    }
};

TEST_F(AdaptiveAvgPool3dInferShapeTest, adaptive_avg_pool3d_infershape_test1)
{
    ge::op::AdaptiveAvgPool3d op;

    ge::DataType x_type = ge::DT_FLOAT;
    ge::Format x_format = ge::FORMAT_ND;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 1, 2, 2, 2}, x_type, x_format, {2, 1, 2, 2, 2}, x_format));

    std::vector<int64_t> out_shape = {1, 1, 1};
    op.SetAttr("output_size", out_shape);
    op.SetAttr("data_format", "NDHWC");

    Runtime2TestParam param{{"output_size", "data_format"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
}

TEST_F(AdaptiveAvgPool3dInferShapeTest, adaptive_avg_pool3d_infershape_test2)
{
    ge::op::AdaptiveAvgPool3d op;

    ge::DataType x_type = ge::DT_FLOAT16;
    ge::Format x_format = ge::FORMAT_ND;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 1, 2, 2, 2}, x_type, x_format, {2, 1, 2, 2, 2}, x_format));

    std::vector<int64_t> out_shape = {1, 1, 1};
    op.SetAttr("output_size", out_shape);
    op.SetAttr("data_format", "NDHWC");

    Runtime2TestParam param{{"output_size", "data_format"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
}

TEST_F(AdaptiveAvgPool3dInferShapeTest, adaptive_avg_pool3d_infershape_test3)
{
    ge::op::AdaptiveAvgPool3d op;

    ge::DataType x_type = ge::DT_BF16;
    ge::Format x_format = ge::FORMAT_ND;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 1, 2, 2, 2}, x_type, x_format, {2, 1, 2, 2, 2}, x_format));

    std::vector<int64_t> out_shape = {1, 1, 1};
    op.SetAttr("output_size", out_shape);
    op.SetAttr("data_format", "NDHWC");

    Runtime2TestParam param{{"output_size", "data_format"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
}

TEST_F(AdaptiveAvgPool3dInferShapeTest, adaptive_avg_pool3d_infershape_test4)
{
    ge::op::AdaptiveAvgPool3d op;

    ge::DataType x_type = ge::DT_FLOAT;
    ge::Format x_format = ge::FORMAT_ND;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 1, 2, 2, 2}, x_type, x_format, {2, 1, 2, 2, 2}, x_format));

    std::vector<int64_t> out_shape = {1, 1, 1};
    op.SetAttr("output_size", out_shape);
    op.SetAttr("data_format", "NDHWC");

    Runtime2TestParam param{{"output_size", "data_format"}, {}, {}};
    EXPECT_EQ(InferDataTypeTest(op, param), ge::GRAPH_SUCCESS);
    auto y_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(y_desc.GetDataType(), x_type);
}