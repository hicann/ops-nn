/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"
#include "../../../op_graph/logit_proto.h"

class Logit : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "Logit SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Logit TearDown" << std::endl;
    }
};

TEST_F(Logit, Logit_infershape_case_0)
{
    ge::op::Logit op;
    op.UpdateInputDesc("x", create_desc({4, 1, 1280}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}