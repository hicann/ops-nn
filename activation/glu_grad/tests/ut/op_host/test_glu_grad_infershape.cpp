/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "infershape_test_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"
#include "../../../op_graph/glu_grad_proto.h"

using namespace ge;

class GluGradInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GluGradInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GluGradInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(GluGradInferShapeTest, infershape_legal_negative_dim)
{
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GLUGrad")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::Shape grad_out_shape = {2, 3, 4};
    gert::Shape self_shape = {2, 6, 4};
    gert::Shape output_shape = {};
    int64_t dim = -2;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&grad_out_shape, &self_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dim)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    std::vector<int64_t> expected_output_shape = {2, 6, 4};
    auto actual_output_shape = context->GetOutputShape(0);
    ASSERT_NE(actual_output_shape, nullptr);
    ASSERT_EQ(actual_output_shape->GetDimNum(), expected_output_shape.size());
    for (size_t i = 0; i < expected_output_shape.size(); ++i) {
        EXPECT_EQ(actual_output_shape->GetDim(i), expected_output_shape[i]);
    }
}

TEST_F(GluGradInferShapeTest, gert_infershape_legal_positive_dim)
{
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GLUGrad")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::Shape grad_out_shape = {2, 3, 4};
    gert::Shape self_shape = {2, 6, 4};
    gert::Shape output_shape = {};
    int64_t dim = 1;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&grad_out_shape, &self_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dim)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    std::vector<int64_t> expected_output_shape = {2, 6, 4};
    auto actual_output_shape = context->GetOutputShape(0);
    ASSERT_NE(actual_output_shape, nullptr);
    ASSERT_EQ(actual_output_shape->GetDimNum(), expected_output_shape.size());
    for (size_t i = 0; i < expected_output_shape.size(); ++i) {
        EXPECT_EQ(actual_output_shape->GetDim(i), expected_output_shape[i]);
    }
}

TEST_F(GluGradInferShapeTest, gert_infershape_dim_out_of_range)
{
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GLUGrad")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::Shape grad_out_shape = {2, 3, 4};
    gert::Shape self_shape = {2, 6, 4};
    gert::Shape output_shape = {};
    int64_t dim = 3;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&grad_out_shape, &self_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dim)}})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(GluGradInferShapeTest, gert_infershape_scalar_self)
{
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GLUGrad")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::Shape grad_out_shape = {};
    gert::Shape self_shape = {};
    gert::Shape output_shape = {};
    int64_t dim = 0;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&grad_out_shape, &self_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dim)}})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
