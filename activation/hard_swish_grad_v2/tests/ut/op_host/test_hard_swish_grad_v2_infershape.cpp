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
#include "../../../op_graph/hard_swish_grad_v2_proto.h"

class HardSwishGradV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "HardSwishGradV2 Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "HardSwishGradV2 Proto TearDown" << std::endl;
    }
};

TEST_F(HardSwishGradV2, HardSwishGradV2_infershape_case_0)
{
    gert::StorageShape grad_output_shape = {{2, 4}, {2, 4}};
    gert::StorageShape self_shape = {{2, 4}, {2, 4}};
    gert::StorageShape out_shape = {{}, {}};
    
    ge::op::HardSwishGradV2 op;
    op.UpdateInputDesc("gradOutput", create_desc({2, 4}, ge::DT_BF16));
    op.UpdateInputDesc("self", create_desc({2, 4}, ge::DT_BF16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    std::vector<int64_t> expected_output_shape = {2, 4};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
}