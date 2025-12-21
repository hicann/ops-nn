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
#include "../../../op_graph/apply_adam_w_v2_proto.h"

class ApplyAdamWV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyAdamWV2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyAdamWV2 TearDown" << std::endl;
    }
};

TEST_F(ApplyAdamWV2, ApplyAdamWV2_infershape_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWV2"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape varShape = {{96, 256}, {96, 256}};
    gert::StorageShape mShape = {{96, 256}, {96, 256}};
    gert::StorageShape vShape = {{96, 256}, {96, 256}};
    gert::StorageShape gradShape = {{256}, {256}};
    gert::StorageShape stepShape = {{1}, {1}};
    gert::StorageShape max_grad_normShape = {{256}, {256}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(6, 4)
        .IrInstanceNum({1, 1, 1, 1, 1, 1})
        .InputShapes({&varShape, &mShape, &vShape, &gradShape,
                        &stepShape, &max_grad_normShape})
        .OutputShapes({&varShape, &mShape, &vShape, &max_grad_normShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}