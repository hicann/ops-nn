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
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace {
TEST(GaussianNllLossGradInferShape, OutputsMatchInputAndVar)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("GaussianNllLossGrad");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    gert::Shape gradOutputShape({1});
    gert::Shape inputShape({2, 3, 4});
    gert::Shape targetShape({2, 1, 4});
    gert::Shape varShape({2, 3});
    gert::Shape gradInputShape;
    gert::Shape gradVarShape;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GaussianNllLossGrad")
                      .NodeIoNum(4, 2)
                      .InputShapes({&gradOutputShape, &inputShape, &targetShape, &varShape})
                      .OutputShapes({&gradInputShape, &gradVarShape})
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->infer_shape(context), ge::GRAPH_SUCCESS);
    const auto* inferredGradInput = context->GetOutputShape(0);
    const auto* inferredGradVar = context->GetOutputShape(1);
    ASSERT_NE(inferredGradInput, nullptr);
    ASSERT_NE(inferredGradVar, nullptr);
    ASSERT_EQ(inferredGradInput->GetDimNum(), 3);
    EXPECT_EQ(inferredGradInput->GetDim(0), 2);
    EXPECT_EQ(inferredGradInput->GetDim(1), 3);
    EXPECT_EQ(inferredGradInput->GetDim(2), 4);
    ASSERT_EQ(inferredGradVar->GetDimNum(), 2);
    EXPECT_EQ(inferredGradVar->GetDim(0), 2);
    EXPECT_EQ(inferredGradVar->GetDim(1), 3);
}
} // namespace
