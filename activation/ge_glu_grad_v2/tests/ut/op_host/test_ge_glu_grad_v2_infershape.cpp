/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 *
 * @file test_ge_glu_grad_v2_infershape.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/ge_glu_grad_v2_proto.h"

class GEGLUGRADV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GEGLUGRADV2 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GEGLUGRADV2 Proto Test TearDown" << std::endl;
    }
};

TEST_F(GEGLUGRADV2, GEGLUGRADV2_infershape_diff_test_legal_input)
{
    ge::op::GeGluGradV2 op;
    op.UpdateInputDesc("dy", create_desc({4, 4, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({4, 4, 8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gelu", create_desc({4, 4, 4}, ge::DT_FLOAT16));
    op.SetAttr("dim", -1);
    op.SetAttr("approximate", 1);
    op.SetAttr("activate_left", false);
    Runtime2TestParam param{{"dim", "approximate", "activate_left"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_dx_desc = op.GetOutputDesc(0);
    std::vector<int64_t> expected_output_shape = {4, 4, 8};
    EXPECT_EQ(output_dx_desc.GetShape().GetDims(), expected_output_shape);
}
