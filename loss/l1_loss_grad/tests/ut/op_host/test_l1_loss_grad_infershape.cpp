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
#include "log/log.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "platform/platform_info.h"
#include "../../../op_graph/l1_loss_grad_proto.h"

class L1LossGradTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "L1LossGrad SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "L1LossGrad TearDown" << std::endl;
    }
};

TEST_F(L1LossGradTest, l1lossgrad_infershape_test1) {
    ge::op::L1LossGrad op;
    op.UpdateInputDesc("predict", create_desc({-1, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("grads", create_desc({-1, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("label", create_desc({-1, 8}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {-1, 8};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    Runtime2TestParam param{{"reduction"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(L1LossGradTest, l1lossgrad_infershape_test2) {
    ge::op::L1LossGrad op;
    op.UpdateInputDesc("predict", create_desc({-1, -1, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("grads", create_desc({-1, -1, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("label", create_desc({-1, -1, 8}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {-1, -1, 8};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    Runtime2TestParam param{{"reduction"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(L1LossGradTest, l1lossgrad_infershape_test3) {
    ge::op::L1LossGrad op;
    op.UpdateInputDesc("predict", create_desc({4, 16, 36}, ge::DT_FLOAT));
    op.UpdateInputDesc("grads", create_desc({4, 16, 36}, ge::DT_FLOAT));
    op.UpdateInputDesc("label", create_desc({4, 16, 36}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {4, 16, 36};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    Runtime2TestParam param{{"reduction"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(L1LossGradTest, l1lossgrad_infershape_test4) {
    ge::op::L1LossGrad op;
    op.UpdateInputDesc("predict", create_desc({4, 16, 36}, ge::DT_FLOAT));
    op.UpdateInputDesc("grads", create_desc({4, 16, 36}, ge::DT_FLOAT16));
    op.UpdateInputDesc("label", create_desc({4, 16, 36}, ge::DT_FLOAT16));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    Runtime2TestParam param{{"reduction"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
}
