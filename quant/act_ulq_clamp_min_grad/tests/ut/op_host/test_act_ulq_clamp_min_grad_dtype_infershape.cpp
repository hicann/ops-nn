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
#include "../../../op_graph/act_ulq_clamp_min_grad_graph_infer.cpp"

// The Host UT library collects op_host inference files only; include the graph
// dtype implementation here to exercise its validation without SDK registry precedence.

namespace {
ge::graphStatus InferDtype(ge::DataType y, ge::DataType mask, ge::DataType loss, ge::DataType& output)
{
    auto holder = gert::OpInferDataTypeContextBuilder()
                      .OpType("ActULQClampMinGrad")
                      .OpName("ActULQClampMinGrad")
                      .IONum(3, 1)
                      .InputTensorDesc(0, y, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, mask, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(2, loss, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto* context = holder.GetContext();
    const auto status = ops::InferDataType4ActULQClampMinGrad(context);
    if (status == ge::GRAPH_SUCCESS) {
        output = context->GetOutputDataType(0);
    }
    return status;
}

TEST(ActULQClampMinGradInferDtype, LegalCombinationsPreserveOutputType)
{
    for (auto y : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        for (auto mask : {y, ge::DT_BOOL}) {
            ge::DataType output = ge::DT_UNDEFINED;
            ASSERT_EQ(InferDtype(y, mask, y, output), ge::GRAPH_SUCCESS);
            EXPECT_EQ(output, y);
        }
    }
}

TEST(ActULQClampMinGradInferDtype, RejectUnsupportedMaskTypes)
{
    for (auto y : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        for (auto mask : {ge::DT_DOUBLE, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8}) {
            ge::DataType output = ge::DT_UNDEFINED;
            EXPECT_EQ(InferDtype(y, mask, y, output), ge::GRAPH_FAILED);
        }
    }
}

TEST(ActULQClampMinGradInferDtype, RejectUnsupportedAndMixedFloatingInputs)
{
    ge::DataType output = ge::DT_UNDEFINED;
    EXPECT_EQ(InferDtype(ge::DT_DOUBLE, ge::DT_BOOL, ge::DT_DOUBLE, output), ge::GRAPH_FAILED);
    EXPECT_EQ(InferDtype(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, output), ge::GRAPH_FAILED);
    EXPECT_EQ(InferDtype(ge::DT_FLOAT16, ge::DT_BOOL, ge::DT_FLOAT, output), ge::GRAPH_FAILED);
}
} // namespace
