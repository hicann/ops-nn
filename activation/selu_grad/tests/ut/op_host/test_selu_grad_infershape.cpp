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

#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

class SeluGradInferShapeTest : public testing::Test {};

TEST_F(SeluGradInferShapeTest, SameShapeSucceeds)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SeluGrad")->infer_shape;
    gert::Shape gradientsShape = {2, 3, 4};
    gert::Shape outputsShape = {2, 3, 4};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradientsShape, &outputsShape})
                      .OutputShapes({&yShape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    const gert::Shape* inferredYShape = context->GetOutputShape(0);
    ASSERT_NE(inferredYShape, nullptr);
    EXPECT_EQ(*inferredYShape, gradientsShape);
}

TEST_F(SeluGradInferShapeTest, BroadcastableShapeFails)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SeluGrad")->infer_shape;
    gert::Shape gradientsShape = {2, 3, 4};
    gert::Shape outputsShape = {1, 3, 1};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradientsShape, &outputsShape})
                      .OutputShapes({&yShape})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(SeluGradInferShapeTest, CompatibleDynamicDimensionSucceeds)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SeluGrad")->infer_shape;
    gert::Shape gradientsShape = {-1, 3, 4};
    gert::Shape outputsShape = {2, 3, 4};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradientsShape, &outputsShape})
                      .OutputShapes({&yShape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    const gert::Shape* inferredYShape = context->GetOutputShape(0);
    ASSERT_NE(inferredYShape, nullptr);
    EXPECT_EQ(*inferredYShape, gradientsShape);
}

TEST_F(SeluGradInferShapeTest, DynamicDimensionDoesNotHideKnownMismatch)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SeluGrad")->infer_shape;
    gert::Shape gradientsShape = {-1, 3, 4};
    gert::Shape outputsShape = {2, 5, 4};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradientsShape, &outputsShape})
                      .OutputShapes({&yShape})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(SeluGradInferShapeTest, UnknownGradientsRankDoesNotBypassOutputsRankLimit)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SeluGrad")->infer_shape;
    gert::Shape gradientsShape = {-2};
    gert::Shape outputsShape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradientsShape, &outputsShape})
                      .OutputShapes({&yShape})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
