/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <gtest/gtest.h>
#include <vector>

#include "../../../op_graph/inplace_apply_adagrad_proto.h"
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeInplaceApplyAdagrad(gert::InferDataTypeContext* context);
}

namespace {
ge::graphStatus RunRegisteredVerifyCase(const std::array<ge::DataType, 4>& inputDtypes)
{
    ge::op::InplaceApplyAdagrad op("inplace_apply_adagrad_verify_ut");
    const ge::Shape tensorShape({8, 16});
    const ge::Shape scalarShape(std::vector<int64_t>{});
    const std::array<const char*, 4> inputNames = {"var", "accum", "lr", "grad"};
    for (size_t inputIdx = 0; inputIdx < inputDtypes.size(); ++inputIdx) {
        const ge::Shape& shape = inputIdx == 2 ? scalarShape : tensorShape;
        EXPECT_EQ(op.UpdateInputDesc(inputNames[inputIdx], ge::TensorDesc(shape, ge::FORMAT_ND, inputDtypes[inputIdx])),
                  ge::GRAPH_SUCCESS);
    }
    EXPECT_EQ(op.UpdateOutputDesc("var", ge::TensorDesc(tensorShape, ge::FORMAT_ND, inputDtypes[0])),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateOutputDesc("accum", ge::TensorDesc(tensorShape, ge::FORMAT_ND, inputDtypes[0])),
              ge::GRAPH_SUCCESS);
    return op.VerifyAllAttr(true);
}

void RunInferDataTypeInplaceApplyAdagrad(ge::DataType inDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("InplaceApplyAdagrad");
    builder.OpName("InplaceApplyAdagrad");
    builder.IONum(4, 2);
    for (int32_t i = 0; i < 4; ++i) {
        builder.InputTensorDesc(i, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < 2; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeInplaceApplyAdagrad(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inDtype);
    EXPECT_EQ(context->GetOutputDataType(1), inDtype);
}

void RunInferDataTypeMismatchCase(size_t mismatchInputIdx)
{
    std::array<ge::DataType, 4> inputDtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    inputDtypes[mismatchInputIdx] = ge::DT_FLOAT16;
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("InplaceApplyAdagrad");
    builder.OpName("InplaceApplyAdagrad");
    builder.IONum(4, 2);
    for (int32_t i = 0; i < 4; ++i) {
        builder.InputTensorDesc(i, inputDtypes[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < 2; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(ops::InferDataTypeInplaceApplyAdagrad(context), ge::GRAPH_FAILED);
}
} // namespace

TEST(InplaceApplyAdagradGraphVerifyTest, accepts_matching_inputs)
{
    EXPECT_EQ(RunRegisteredVerifyCase({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunRegisteredVerifyCase({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunRegisteredVerifyCase({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16}), ge::GRAPH_SUCCESS);
}

TEST(InplaceApplyAdagradGraphVerifyTest, rejects_each_mismatched_input)
{
    EXPECT_EQ(RunRegisteredVerifyCase({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT}), ge::GRAPH_FAILED);
    EXPECT_EQ(RunRegisteredVerifyCase({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16}), ge::GRAPH_FAILED);
    EXPECT_EQ(RunRegisteredVerifyCase({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT}), ge::GRAPH_FAILED);
}

TEST(InplaceApplyAdagradGraphInfer, InferDataTypeFloat) { RunInferDataTypeInplaceApplyAdagrad(ge::DT_FLOAT); }

TEST(InplaceApplyAdagradGraphInfer, InferDataTypeFloat16) { RunInferDataTypeInplaceApplyAdagrad(ge::DT_FLOAT16); }

TEST(InplaceApplyAdagradGraphInfer, InferDataTypeBFloat16) { RunInferDataTypeInplaceApplyAdagrad(ge::DT_BF16); }

TEST(InplaceApplyAdagradGraphInfer, rejects_mismatched_inputs)
{
    RunInferDataTypeMismatchCase(1);
    RunInferDataTypeMismatchCase(2);
    RunInferDataTypeMismatchCase(3);
}
