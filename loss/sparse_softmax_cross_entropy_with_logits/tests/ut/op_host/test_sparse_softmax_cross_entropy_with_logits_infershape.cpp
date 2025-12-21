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
#include "../../../op_graph/sparse_softmax_cross_entropy_with_logits_proto.h"

class tset_sparse_softmax_cross_entropy_with_logits_infershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "sparse_softmax_cross_entropy_with_logits SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "sparse_softmax_cross_entropy_with_logits TearDown" << std::endl;
    }
};

TEST_F(tset_sparse_softmax_cross_entropy_with_logits_infershape, test_infershape_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSoftmaxCrossEntropyWithLogits"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSoftmaxCrossEntropyWithLogits")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape featuresShape = {{96, 256}, {96, 256}};
    gert::StorageShape labelsShape = {{96}, {96}};
    gert::StorageShape lossShape = {{96}, {96}};
    gert::StorageShape backpropShape = {{96, 256}, {96, 256}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1})
        .InputShapes({&featuresShape, &labelsShape})
        .OutputShapes({&lossShape, &backpropShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}