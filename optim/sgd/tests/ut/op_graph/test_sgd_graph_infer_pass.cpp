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
#include <array>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForSgd(gert::InferDataTypeContext* context);
} // namespace ops

namespace {
void CheckSgdDtype(const std::array<ge::DataType, 6>& types, ge::graphStatus expected)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("SGD").OpName("sgd_dtype").IONum(6, 1);
    for (size_t input = 0; input < types.size(); ++input) {
        builder.InputTensorDesc(input, types[input], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_EQ(ops::InferDataTypeForSgd(holder.GetContext()), expected);
    if (expected == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(holder.GetContext()->GetOutputDataType(0), types[0]);
    }
}
} // namespace

TEST(SgdGraphInfer, PreservesSupportedTypes)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}) {
        std::array<ge::DataType, 6> types;
        types.fill(dtype);
        CheckSgdDtype(types, ge::GRAPH_SUCCESS);
    }
}

TEST(SgdGraphInfer, RejectsEachMismatchedInput)
{
    for (size_t input = 1; input < 6; ++input) {
        std::array<ge::DataType, 6> types;
        types.fill(ge::DT_FLOAT);
        types[input] = ge::DT_FLOAT16;
        CheckSgdDtype(types, ge::GRAPH_FAILED);
    }
}

TEST(SgdGraphInfer, RejectsUnsupportedMatchingTypes)
{
    for (auto dtype : {ge::DT_INT32, ge::DT_BOOL, ge::DT_DOUBLE}) {
        std::array<ge::DataType, 6> types;
        types.fill(dtype);
        CheckSgdDtype(types, ge::GRAPH_FAILED);
    }
}
