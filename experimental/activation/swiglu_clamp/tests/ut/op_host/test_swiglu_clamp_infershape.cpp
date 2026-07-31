/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_swiglu_clamp_infershape.cpp
 * \brief SwigluClamp InferShape UT: y shape = x shape with last dim halved.
 *        New-style (OpDef) registration comes from swiglu_clamp_infershape.cpp linked via
 *        add_modules_ut_sources(OP_INFERSHAPE_MODULE_NAME); no proto.h needed.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class TestSwigluClampInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwigluClamp Infershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SwigluClamp Infershape TearDown" << std::endl; }
};

// x [..., 2N] -> y [..., N]: verify success and that the last dim is halved.
TEST_F(TestSwigluClampInfershape, swiglu_clamp_infershape_basic_test)
{
    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SwigluClamp");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape input_shape = {4, 3, 16}; // x last dim 16 = 2N, N=8
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    gert::InferShapeContext* ctx = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(ctx), ge::GRAPH_SUCCESS);
    // InferShapeContextFaker 把 output_shape 值拷贝进 context(TensorDescription.shape_ 是值成员),
    // infershape 改的是 context 内部那份,本地 output_shape 不会被回写 —— 必须从 context 读回。
    const gert::Shape* resultShape = ctx->GetOutputShape(0);
    ASSERT_NE(resultShape, nullptr);
    // y last dim must equal x last dim / 2 = 8
    EXPECT_EQ(resultShape->GetDim(resultShape->GetDimNum() - 1), 8);
    // leading dims preserved
    EXPECT_EQ(resultShape->GetDim(0), 4);
    EXPECT_EQ(resultShape->GetDim(1), 3);
}

// odd last dim must be rejected (SwigluClamp requires x last dim even).
TEST_F(TestSwigluClampInfershape, swiglu_clamp_infershape_odd_lastdim_test)
{
    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SwigluClamp");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape input_shape = {4, 3, 15}; // odd last dim -> must fail
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
