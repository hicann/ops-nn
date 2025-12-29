/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_test_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "ut_op_common.h"
#include "../../../op_graph/fast_gelu_grad_proto.h"

using namespace ge;

class FastGeluGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FastGeluGradTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FastGeluGradTest TearDown" << std::endl;
    }
};

TEST_F(FastGeluGradTest, fast_gelu_grad_infershape_test)
{
    ge::op::FastGeluGrad op;

    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 16}, {1, 16}};

    auto tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {16, 16}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("dy", tensor_desc);
    op.UpdateInputDesc("x", tensor_desc);

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("z");
    auto output_shape = output_desc.GetShape();
    EXPECT_EQ(output_shape.GetDimNum(), 2);

    std::vector<int64_t> expected_output_shape = {-1, -1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

}
