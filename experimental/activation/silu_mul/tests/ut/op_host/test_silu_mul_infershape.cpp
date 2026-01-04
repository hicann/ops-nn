/*
 * Copyright (c) 2025 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/silu_mul_proto.h"
#include "log/log.h"
class SiluMul : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SiluMul SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SiluMul TearDown" << std::endl;
    }
};

TEST_F(SiluMul, SiluMul_infershape_case_0)
{
    ge::op::SiluMul op;
    op.UpdateInputDesc("x", create_desc({4, 1, 1280}, ge::DT_FLOAT16));
    op.UpdateInputDesc("y", create_desc({4, 1, 1280}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);
}