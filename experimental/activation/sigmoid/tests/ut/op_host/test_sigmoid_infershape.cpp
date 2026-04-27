/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../op_graph/sigmoid_proto.h"


class SigmoidInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SigmoidInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SigmoidInfershape TearDown" << std::endl;
    }
};

TEST_F(SigmoidInfershape, sigmoid_infershape_test1)
{

    ge::op::Sigmoid sigmoid_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({2, 2});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);
    XDesc.SetOriginShape(xShape);

    sigmoid_op.UpdateInputDesc("x", XDesc);

    std::vector<int64_t> expected_output1_shape = {2, 2};
 
    EXPECT_EQ(InferShapeTest(sigmoid_op), ge::GRAPH_SUCCESS);
    auto output0_desc = sigmoid_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output1_shape);
}
