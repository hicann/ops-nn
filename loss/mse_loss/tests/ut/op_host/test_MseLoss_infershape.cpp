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
 * \file test_MseLoss_infershape.cpp
 * \brief
 */

#include <iostream>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/mse_loss_proto.h"

class mse_loss : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "mse_loss Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "mse_loss Proto Test TearDown" << std::endl;
    }
};

TEST_F(mse_loss, mse_loss_infershape_diff_test)
{
    ge::op::MseLoss op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{15, 16}, {8, 8}, {375, 375}};
    auto tensor_desc =
        create_desc_shape_range({-1, 8, 375}, ge::DT_FLOAT16, ge::FORMAT_ND, {16, 8, 375}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("predict", tensor_desc);
    op.UpdateInputDesc("label", tensor_desc);
    op.SetAttr("reduction", "mean");
    // auto ret = op.InferShapeAndType();
    // EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y");
    // EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
    EXPECT_EQ(output_shape_range, expected_shape_range);

    Runtime2TestParam param{{"reduction"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}
