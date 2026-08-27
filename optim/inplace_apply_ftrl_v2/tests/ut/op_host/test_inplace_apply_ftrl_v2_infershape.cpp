/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/inplace_apply_ftrl_v2_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class InplaceApplyFtrlV2InferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InplaceApplyFtrlV2 InferShape Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "InplaceApplyFtrlV2 InferShape Test TearDown" << std::endl; }
};

static void DoInferShapeTest(gert::StorageShape& inputShape, ge::graphStatus expectedStatus)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyFtrlV2")->infer_shape;
    gert::StorageShape outVarShape = {{}, {}};
    gert::StorageShape outAccumShape = {{}, {}};
    gert::StorageShape outLinearShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(9, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&inputShape, &inputShape, &inputShape, &inputShape, &inputShape, &inputShape,
                                    &inputShape, &inputShape, &inputShape})
                      .OutputShapes({&outVarShape, &outAccumShape, &outLinearShape})
                      .Build();

    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), inputShape.GetOriginShape().GetDimNum());
        EXPECT_EQ(context->GetOutputShape(1)->GetDimNum(), inputShape.GetOriginShape().GetDimNum());
        EXPECT_EQ(context->GetOutputShape(2)->GetDimNum(), inputShape.GetOriginShape().GetDimNum());
    }
}

TEST_F(InplaceApplyFtrlV2InferShapeTest, infershape_1d_fp32_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyFtrlV2")->infer_shape;

    gert::StorageShape varShape = {{128}, {128}};
    gert::StorageShape outVarShape = {{}, {}};
    gert::StorageShape outAccumShape = {{}, {}};
    gert::StorageShape outLinearShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(9, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &varShape, &varShape, &varShape, &varShape, &varShape, &varShape,
                                    &varShape, &varShape})
                      .OutputShapes({&outVarShape, &outAccumShape, &outLinearShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    const auto* actualVarShape = context->GetOutputShape(0);
    const auto* actualAccumShape = context->GetOutputShape(1);
    const auto* actualLinearShape = context->GetOutputShape(2);
    ASSERT_NE(actualVarShape, nullptr);
    ASSERT_NE(actualAccumShape, nullptr);
    ASSERT_NE(actualLinearShape, nullptr);
    EXPECT_EQ(actualVarShape->GetDimNum(), 1);
    EXPECT_EQ(actualVarShape->GetDim(0), 128);
    EXPECT_EQ(actualAccumShape->GetDimNum(), 1);
    EXPECT_EQ(actualAccumShape->GetDim(0), 128);
    EXPECT_EQ(actualLinearShape->GetDimNum(), 1);
    EXPECT_EQ(actualLinearShape->GetDim(0), 128);
}

TEST_F(InplaceApplyFtrlV2InferShapeTest, infershape_rank0_test)
{
    gert::StorageShape rank0Shape = {{}, {}};
    DoInferShapeTest(rank0Shape, ge::GRAPH_SUCCESS);
}

TEST_F(InplaceApplyFtrlV2InferShapeTest, infershape_reject_rank9_test)
{
    gert::StorageShape rank9Shape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
    DoInferShapeTest(rank9Shape, ge::GRAPH_FAILED);
}
