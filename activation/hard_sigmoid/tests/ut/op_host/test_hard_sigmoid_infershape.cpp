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
void RunInferShapeCase(gert::Shape inputShape)
{
    gert::Shape outputShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&inputShape})
                      .OutputShapes({&outputShape})
                      .Build();

    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoid");
    ASSERT_NE(impl, nullptr);
    auto inferShape = impl->infer_shape;
    ASSERT_NE(inferShape, nullptr);
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShape(context), ge::GRAPH_SUCCESS);
    const auto* inferredShape = context->GetOutputShape(0);
    ASSERT_NE(inferredShape, nullptr);
    EXPECT_EQ(*inferredShape, inputShape);
}

void RunInferDataTypeCase(ge::DataType dtype)
{
    ge::DataType inputRef = dtype;
    ge::DataType outputRef = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(1)
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&inputRef})
                      .OutputDataTypes({&outputRef})
                      .Build();

    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoid");
    ASSERT_NE(impl, nullptr);
    auto inferDataType = impl->infer_datatype;
    ASSERT_NE(inferDataType, nullptr);
    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDataType(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), dtype);
}
} // namespace

TEST(HardSigmoidInferShapeTest, TensorShape) { RunInferShapeCase({2, 3, 5, 7}); }

TEST(HardSigmoidInferShapeTest, ScalarShape) { RunInferShapeCase({}); }

TEST(HardSigmoidInferShapeTest, EmptyShape) { RunInferShapeCase({0, 4}); }

TEST(HardSigmoidInferShapeTest, InferDataType)
{
    RunInferDataTypeCase(ge::DT_FLOAT);
    RunInferDataTypeCase(ge::DT_FLOAT16);
    RunInferDataTypeCase(ge::DT_BF16);
    RunInferDataTypeCase(ge::DT_INT32);
}
