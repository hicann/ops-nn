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
 * \file test_fused_mat_mul_graph_infer_pass.cpp
 * \brief FusedMatMul InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForFusedMatMul(gert::InferDataTypeContext* context);
}

namespace {
constexpr size_t kFusedIrInputNum = 4; // x1/x2/bias/x3, bias与x3均为OPTIONAL
constexpr size_t kFusedOutputNum = 1;
constexpr size_t kFusedBiasIdx = 2;
constexpr size_t kFusedX3Idx = 3;

// attrs按def声明顺序追加: 0=transpose_x1(Bool) 1=transpose_x2(Bool) 2=enable_hf32(Bool) 3=fused_op_type(String)
gert::ContextHolder<gert::InferDataTypeContext> BuildFusedDtypeContext(ge::DataType aDtype, ge::DataType bDtype,
                                                                       const char* fusedOpType, bool withBias = false,
                                                                       ge::DataType biasDtype = ge::DT_UNDEFINED,
                                                                       bool withX3 = false,
                                                                       ge::DataType x3Dtype = ge::DT_UNDEFINED)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("FusedMatMul").OpName("FusedMatMul");
    builder.IONum(kFusedIrInputNum, kFusedOutputNum);
    builder.InputTensorDesc(0, aDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, bDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(kFusedBiasIdx, withBias ? biasDtype : ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(kFusedX3Idx, withX3 ? x3Dtype : ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(false).AppendAttr(false).AppendAttr(false).AppendAttr(ge::AscendString(fusedOpType));
    auto holder = builder.Build();
    // builder不提供输出dtype设置,输出槽残留DT_MAX;归一为DT_UNDEFINED模拟真实GE未声明输出的语义
    EXPECT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    return holder;
}
} // namespace

// 默认模式(""): 无bias无x3, y跟随a
TEST(FusedMatMulGraphInfer, DefaultModeFp16OutFp16)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "");
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(FusedMatMulGraphInfer, DefaultModeBf16OutBf16)
{
    auto holder = BuildFusedDtypeContext(ge::DT_BF16, ge::DT_BF16, "");
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

// 默认模式bias: 与x1同dtype或fp32均接受
TEST(FusedMatMulGraphInfer, DefaultModeBiasFp32)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "", true, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// a/b必须同dtype
TEST(FusedMatMulGraphInfer, MismatchedABFailed)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_BF16, "");
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_FAILED);
}

// quant模式: 仅fp16 x1/x2 + uint64 x3, y固定INT8
TEST(FusedMatMulGraphInfer, QuantModeOutInt8)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "quant", false, ge::DT_UNDEFINED, true,
                                         ge::DT_UINT64);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
}

TEST(FusedMatMulGraphInfer, QuantModeRejectsFp32X1)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT, ge::DT_FLOAT, "quant", false, ge::DT_UNDEFINED, true,
                                         ge::DT_UINT64);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_FAILED);
}

TEST(FusedMatMulGraphInfer, QuantModeRejectsInt64X3)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "quant", false, ge::DT_UNDEFINED, true,
                                         ge::DT_INT64);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_FAILED);
}

// quant模式bias仅允许fp16
TEST(FusedMatMulGraphInfer, QuantModeBiasFp32Failed)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "quant", true, ge::DT_FLOAT, true,
                                         ge::DT_UINT64);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_FAILED);
}

// relu_quant模式同quant输出INT8
TEST(FusedMatMulGraphInfer, ReluQuantModeOutInt8)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "relu_quant", false, ge::DT_UNDEFINED, true,
                                         ge::DT_UINT64);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
}

// 16cast32模式: fp16/bf16输入(不允许fp32), 不支持x3, y固定FP32
TEST(FusedMatMulGraphInfer, SixteenCast32OutFp32)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "16cast32");
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(FusedMatMulGraphInfer, SixteenCast32Bf16OutFp32)
{
    auto holder = BuildFusedDtypeContext(ge::DT_BF16, ge::DT_BF16, "16cast32");
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(FusedMatMulGraphInfer, SixteenCast32RejectsFp32X1)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT, ge::DT_FLOAT, "16cast32");
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_FAILED);
}

TEST(FusedMatMulGraphInfer, SixteenCast32RejectsX3)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "16cast32", false, ge::DT_UNDEFINED, true,
                                         ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_FAILED);
}

// add模式: x3必填且与x1同dtype
TEST(FusedMatMulGraphInfer, AddModeWithX3)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "add", false, ge::DT_UNDEFINED, true,
                                         ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(FusedMatMulGraphInfer, AddModeWithoutX3Failed)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "add");
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_FAILED);
}

TEST(FusedMatMulGraphInfer, AddModeX3MismatchFailed)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "add", false, ge::DT_UNDEFINED, true,
                                         ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_FAILED);
}

// mul模式与add同规则
TEST(FusedMatMulGraphInfer, MulModeWithX3)
{
    auto holder = BuildFusedDtypeContext(ge::DT_BF16, ge::DT_BF16, "mul", false, ge::DT_UNDEFINED, true, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

// scale_add模式: x3必填且不支持bias
TEST(FusedMatMulGraphInfer, ScaleAddModeWithX3)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "scale_add", false, ge::DT_UNDEFINED, true,
                                         ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(FusedMatMulGraphInfer, ScaleAddModeWithBiasFailed)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "scale_add", true, ge::DT_FLOAT16, true,
                                         ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_FAILED);
}

// gelu等后处理模式不支持x3
TEST(FusedMatMulGraphInfer, GeluModeRejectsX3)
{
    auto holder = BuildFusedDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, "gelu_erf", false, ge::DT_UNDEFINED, true,
                                         ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForFusedMatMul(context), ge::GRAPH_FAILED);
}

TEST(FusedMatMulGraphInfer, RejectsNullContext)
{
    EXPECT_EQ(ops::InferDataTypeForFusedMatMul(nullptr), ge::GRAPH_FAILED);
}
