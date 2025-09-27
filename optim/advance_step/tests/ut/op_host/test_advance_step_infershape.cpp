/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"
#include "../../../op_graph/advance_step_proto.h"

class AdvanceStep : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdvanceStep SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdvanceStep TearDown" << std::endl;
    }
};

TEST_F(AdvanceStep, AdvanceStep_infershape_case_0)
{
    ge::op::AdvanceStep op;
    op.UpdateInputDesc("x1", create_desc({16, 1}, ge::DT_INT64));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);
}