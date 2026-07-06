/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 */

/*!
 * \file test_apply_adam_with_amsgrad_v2_infershape.cpp
 * \brief InferShape UT for ApplyAdamWithAmsgradV2 operator.
 *
 * Schema (aligned to REG_OP(ApplyAdamWithAmsgradV2)):
 *   Inputs  (11): var, m, v, vhat, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
 *     - var / m / v / vhat / grad: share var.shape
 *     - beta1_power / beta2_power / lr / beta1 / beta2 / epsilon: scalar tensors (shape == {1})
 *   Outputs (4): var_out, m_out, v_out, vhat_out (all inferred == var/m/v/vhat shape == var.shape)
 *   Attrs   (0)
 *
 * Test coverage:
 *   case_0  2D shape         {96, 256}
 *   case_1  1D shape         {16}
 *   case_2  4D shape         {2, 4, 8, 16}
 *   case_3  Empty tensor     {0}                (shape with 0 dim)
 *   case_4  8D shape         {2,2,2,2,2,2,2,2}  (README rank=1-8 boundary)
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/apply_adam_with_amsgrad_v2_proto.h"

class ApplyAdamWithAmsgradV2 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyAdamWithAmsgradV2 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyAdamWithAmsgradV2 TearDown" << std::endl; }
};

// case_0: 2D shape, four-output (var/m/v/vhat) shape == var.shape inference.
TEST_F(ApplyAdamWithAmsgradV2, ApplyAdamWithAmsgradV2_infershape_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWithAmsgradV2"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWithAmsgradV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape varShape = {{96, 256}, {96, 256}};
    gert::StorageShape mShape = {{96, 256}, {96, 256}};
    gert::StorageShape vShape = {{96, 256}, {96, 256}};
    gert::StorageShape vhatShape = {{96, 256}, {96, 256}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape gradShape = {{96, 256}, {96, 256}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &mShape, &vShape, &vhatShape, &scalarShape, &scalarShape, &scalarShape,
                                    &scalarShape, &scalarShape, &scalarShape, &gradShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &vhatShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// case_1: 1D shape (small vector).
TEST_F(ApplyAdamWithAmsgradV2, ApplyAdamWithAmsgradV2_infershape_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWithAmsgradV2"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWithAmsgradV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape varShape = {{16}, {16}};
    gert::StorageShape mShape = {{16}, {16}};
    gert::StorageShape vShape = {{16}, {16}};
    gert::StorageShape vhatShape = {{16}, {16}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape gradShape = {{16}, {16}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &mShape, &vShape, &vhatShape, &scalarShape, &scalarShape, &scalarShape,
                                    &scalarShape, &scalarShape, &scalarShape, &gradShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &vhatShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// case_2: 4D shape (common NN tensor shape).
TEST_F(ApplyAdamWithAmsgradV2, ApplyAdamWithAmsgradV2_infershape_case_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWithAmsgradV2"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWithAmsgradV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape varShape = {{2, 4, 8, 16}, {2, 4, 8, 16}};
    gert::StorageShape mShape = {{2, 4, 8, 16}, {2, 4, 8, 16}};
    gert::StorageShape vShape = {{2, 4, 8, 16}, {2, 4, 8, 16}};
    gert::StorageShape vhatShape = {{2, 4, 8, 16}, {2, 4, 8, 16}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape gradShape = {{2, 4, 8, 16}, {2, 4, 8, 16}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &mShape, &vShape, &vhatShape, &scalarShape, &scalarShape, &scalarShape,
                                    &scalarShape, &scalarShape, &scalarShape, &gradShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &vhatShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// case_3: Empty Tensor (one dim is 0). InferShape should propagate the shape
// as-is (pure shape copy implementation).
TEST_F(ApplyAdamWithAmsgradV2, ApplyAdamWithAmsgradV2_infershape_case_3_empty)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWithAmsgradV2"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWithAmsgradV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape varShape = {{0}, {0}};
    gert::StorageShape mShape = {{0}, {0}};
    gert::StorageShape vShape = {{0}, {0}};
    gert::StorageShape vhatShape = {{0}, {0}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape gradShape = {{0}, {0}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &mShape, &vShape, &vhatShape, &scalarShape, &scalarShape, &scalarShape,
                                    &scalarShape, &scalarShape, &scalarShape, &gradShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &vhatShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// case_4: 8D shape (README declares rank in [1, 8]; this exercises the upper
// boundary).
TEST_F(ApplyAdamWithAmsgradV2, ApplyAdamWithAmsgradV2_infershape_case_4_8d)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWithAmsgradV2"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWithAmsgradV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape varShape = {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}};
    gert::StorageShape mShape = {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}};
    gert::StorageShape vShape = {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}};
    gert::StorageShape vhatShape = {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape gradShape = {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &mShape, &vShape, &vhatShape, &scalarShape, &scalarShape, &scalarShape,
                                    &scalarShape, &scalarShape, &scalarShape, &gradShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &vhatShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}
