/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_transpose_batch_mat_mul_graph_infer_pass.cpp
 * \brief TransposeBatchMatMul InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForTransposeBatchMatMul(gert::InferDataTypeContext* context);
}

namespace {
constexpr size_t kTbmmIrInputNum = 4; // x1/x2/.../scale, scale为OPTIONAL且位于index 3
constexpr size_t kTbmmOutputNum = 1;
constexpr size_t kTbmmScaleIdx = 3;

gert::ContextHolder<gert::InferDataTypeContext> BuildTbmmDtypeContext(ge::DataType x1Dtype, ge::DataType x2Dtype,
                                                                      bool withScale = false,
                                                                      ge::DataType scaleDtype = ge::DT_UNDEFINED)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("TransposeBatchMatMul").OpName("TransposeBatchMatMul");
    builder.IONum(kTbmmIrInputNum, kTbmmOutputNum);
    builder.InputTensorDesc(0, x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(kTbmmScaleIdx, withScale ? scaleDtype : ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    // builder不提供输出dtype设置,输出槽残留DT_MAX;归一为DT_UNDEFINED模拟真实GE未声明输出的语义
    EXPECT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    return holder;
}
} // namespace

TEST(TransposeBatchMatMulGraphInfer, InferDataTypeFp16OutFp16)
{
    auto holder = BuildTbmmDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForTransposeBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(TransposeBatchMatMulGraphInfer, InferDataTypeFp32OutFp32)
{
    auto holder = BuildTbmmDtypeContext(ge::DT_FLOAT, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForTransposeBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(TransposeBatchMatMulGraphInfer, InferDataTypeBf16OutBf16)
{
    auto holder = BuildTbmmDtypeContext(ge::DT_BF16, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForTransposeBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

// scale(INT64)实例化时为量化输出,y固定INT8
TEST(TransposeBatchMatMulGraphInfer, ScaleInt64QuantOutInt8)
{
    auto holder = BuildTbmmDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, true, ge::DT_INT64);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForTransposeBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
}

// scale(UINT64)同样触发量化输出
TEST(TransposeBatchMatMulGraphInfer, ScaleUint64QuantOutInt8)
{
    auto holder = BuildTbmmDtypeContext(ge::DT_BF16, ge::DT_BF16, true, ge::DT_UINT64);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForTransposeBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
}

// x1/x2必须成对同dtype
TEST(TransposeBatchMatMulGraphInfer, MismatchedX1X2Failed)
{
    auto holder = BuildTbmmDtypeContext(ge::DT_FLOAT16, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForTransposeBatchMatMul(context), ge::GRAPH_FAILED);
}

TEST(TransposeBatchMatMulGraphInfer, InvalidX1DtypeFailed)
{
    auto holder = BuildTbmmDtypeContext(ge::DT_INT8, ge::DT_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForTransposeBatchMatMul(context), ge::GRAPH_FAILED);
}

// scale仅允许int64/uint64,其余dtype实例化scale时失败
TEST(TransposeBatchMatMulGraphInfer, InvalidScaleDtypeFailed)
{
    auto holder = BuildTbmmDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, true, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForTransposeBatchMatMul(context), ge::GRAPH_FAILED);
}

TEST(TransposeBatchMatMulGraphInfer, RejectsNullContext)
{
    EXPECT_EQ(ops::InferDataTypeForTransposeBatchMatMul(nullptr), ge::GRAPH_FAILED);
}
