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
#include "array_ops.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../../../../tests/ut/common/any_value.h"

namespace {
void RunArch35InferShapeCase(gert::StorageShape& var, gert::StorageShape& lr)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape varOutput = {{}, {}};
    gert::StorageShape accumOutput = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4, 1})
                      .InputShapes({&var, &var, &lr, &var})
                      .OutputShapes({&varOutput, &accumOutput})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"update_slots", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto outputShape = context->GetOutputShape(0);
    auto accumShape = context->GetOutputShape(1);
    ASSERT_NE(outputShape, nullptr);
    ASSERT_NE(accumShape, nullptr);
    ASSERT_EQ(outputShape->GetDimNum(), 2U);
    ASSERT_EQ(accumShape->GetDimNum(), 2U);
    EXPECT_EQ(outputShape->GetDim(0), 8);
    EXPECT_EQ(outputShape->GetDim(1), 16);
    EXPECT_EQ(accumShape->GetDim(0), 8);
    EXPECT_EQ(accumShape->GetDim(1), 16);
}
} // namespace

TEST(InplaceApplyAdagradArch35InferShapeTest, lr_unknown_dim)
{
    gert::StorageShape var = {{8, 16}, {8, 16}};
    gert::StorageShape lr = {{-1}, {-1}};
    RunArch35InferShapeCase(var, lr);
}

TEST(InplaceApplyAdagradArch35InferShapeTest, lr_unknown_rank)
{
    gert::StorageShape var = {{8, 16}, {8, 16}};
    gert::StorageShape lr = {{-2}, {-2}};
    RunArch35InferShapeCase(var, lr);
}
