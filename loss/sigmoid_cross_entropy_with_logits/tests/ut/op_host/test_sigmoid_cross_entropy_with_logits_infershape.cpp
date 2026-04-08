/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "../../../op_graph/sigmoid_cross_entropy_with_logits_proto.h"

class SigmoidCrossEntropyWithLogits : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SigmoidCrossEntropyWithLogits SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SigmoidCrossEntropyWithLogits TearDown" << std::endl;
    }
};

TEST_F(SigmoidCrossEntropyWithLogits, SigmoidCrossEntropyWithLogits_infershape_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SigmoidCrossEntropyWithLogits"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SigmoidCrossEntropyWithLogits")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape predictShape = {{96, 256}, {96, 256}};
    gert::StorageShape targetShape = {{96, 256}, {96, 256}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .InputShapes({&predictShape, &targetShape})
        .OutputShapes({&predictShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(SigmoidCrossEntropyWithLogits, SigmoidCrossEntropyWithLogits_infershape_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SigmoidCrossEntropyWithLogits"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SigmoidCrossEntropyWithLogits")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape predictShape = {{1024}, {1024}};
    gert::StorageShape targetShape = {{1024}, {1024}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .InputShapes({&predictShape, &targetShape})
        .OutputShapes({&predictShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}
