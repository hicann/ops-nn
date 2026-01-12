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
#include <vector>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

class SoftmaxCrossEntropyWithLogitsTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SoftmaxCrossEntropyWithLogitsTest Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SoftmaxCrossEntropyWithLogitsTest Proto Test TearDown" << std::endl;
    }
};

TEST_F(SoftmaxCrossEntropyWithLogitsTest, softmax_cross_entropy_with_logits_test_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape features_shape = {{50, 1}, {50, 1}};
    gert::StorageShape labels_shape = {{50, 1}, {50, 1}};

    std::vector<gert::StorageShape> output_shapes(2);
    std::vector<void*> output_shapes_ref(2);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&features_shape, &labels_shape})
                      .OutputShapes(output_shapes_ref)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto loss_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_loss_shape = {50};
    ASSERT_EQ(Ops::Base::ToString(*loss_desc), Ops::Base::ToString(expected_loss_shape));
    auto backprop_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_backprop_shape = {50, 1};
    ASSERT_EQ(Ops::Base::ToString(*backprop_desc), Ops::Base::ToString(expected_backprop_shape));
}

TEST_F(SoftmaxCrossEntropyWithLogitsTest, softmax_cross_entropy_with_logits_test_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape features_shape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape labels_shape = {{2, 3, 4, 5}, {2, 3, 4, 5}};

    std::vector<gert::StorageShape> output_shapes(2);
    std::vector<void*> output_shapes_ref(2);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&features_shape, &labels_shape})
                      .OutputShapes(output_shapes_ref)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto loss_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_loss_shape = {2, 1, 4, 5};
    ASSERT_EQ(Ops::Base::ToString(*loss_desc), Ops::Base::ToString(expected_loss_shape));
    auto backprop_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_backprop_shape = {2, 3, 4, 5};
    ASSERT_EQ(Ops::Base::ToString(*backprop_desc), Ops::Base::ToString(expected_backprop_shape));
}