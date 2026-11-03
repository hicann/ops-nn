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
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataType4Relu6Grad(gert::InferDataTypeContext* context);
} // namespace ops

namespace {
void CheckDtype(ge::DataType gradients, ge::DataType features, ge::graphStatus expected)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Relu6Grad").OpName("relu6_grad_dtype").IONum(2, 1);
    builder.InputTensorDesc(0, gradients, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, features, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ops::InferDataType4Relu6Grad(holder.GetContext()), expected);
    if (expected == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(holder.GetContext()->GetOutputDataType(0), gradients);
    }
}
} // namespace

TEST(Relu6GradGraphInfer, PreservesAllSupportedDataTypes)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}) {
        CheckDtype(dtype, dtype, ge::GRAPH_SUCCESS);
    }
}

TEST(Relu6GradGraphInfer, RejectsMismatchedInputsWithoutLegacyVerifier)
{
    CheckDtype(ge::DT_FLOAT16, ge::DT_FLOAT, ge::GRAPH_FAILED);
    CheckDtype(ge::DT_BF16, ge::DT_FLOAT16, ge::GRAPH_FAILED);
}

TEST(Relu6GradGraphInfer, RejectsUnsupportedMatchingDataTypes)
{
    CheckDtype(ge::DT_INT32, ge::DT_INT32, ge::GRAPH_FAILED);
    CheckDtype(ge::DT_BOOL, ge::DT_BOOL, ge::GRAPH_FAILED);
    CheckDtype(ge::DT_DOUBLE, ge::DT_DOUBLE, ge::GRAPH_FAILED);
}
