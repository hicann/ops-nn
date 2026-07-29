/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_apply_gradient_descent_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/apply_gradient_descent_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class TestApplyGradientDescentInfershape : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

static void RunInferShapeCase(gert::Shape varShape, gert::Shape alphaShape, gert::Shape deltaShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyGradientDescent")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape outShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&varShape, &alphaShape, &deltaShape})
                      .OutputShapes({&outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto* ctx = holder.GetContext<gert::InferShapeContext>();
    auto* out = ctx->GetOutputShape(0);
    ASSERT_NE(out, nullptr);
    ASSERT_EQ(out->GetDimNum(), varShape.GetDimNum());
    for (size_t i = 0; i < varShape.GetDimNum(); i++) {
        EXPECT_EQ(out->GetDim(i), varShape.GetDim(i));
    }
}

TEST_F(TestApplyGradientDescentInfershape, apply_gradient_descent_infershape_2d)
{
    gert::Shape varShape = {4, 3};
    gert::Shape alphaShape = {1};
    gert::Shape deltaShape = {4, 3};
    RunInferShapeCase(varShape, alphaShape, deltaShape);
}

TEST_F(TestApplyGradientDescentInfershape, apply_gradient_descent_infershape_3d)
{
    gert::Shape varShape = {2, 3, 4};
    gert::Shape alphaShape = {1};
    gert::Shape deltaShape = {2, 3, 4};
    RunInferShapeCase(varShape, alphaShape, deltaShape);
}
