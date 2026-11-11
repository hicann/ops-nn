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
 * \file test_mat_mul_v3_graph_infer_pass.cpp
 * \brief MatMulV3 InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeMatMulV3(gert::InferDataTypeContext* context);
}

namespace {
constexpr size_t kMatMulV3IrInputNum = 3; // x1/x2/bias, bias为OPTIONAL
constexpr size_t kMatMulV3OutputNum = 1;

gert::ContextHolder<gert::InferDataTypeContext> BuildMatMulV3DtypeContext(ge::DataType x1Dtype, ge::DataType x2Dtype,
                                                                          bool withBias = false,
                                                                          ge::DataType biasDtype = ge::DT_UNDEFINED,
                                                                          bool declareY = false,
                                                                          ge::DataType declaredY = ge::DT_UNDEFINED)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("MatMulV3").OpName("MatMulV3");
    builder.IONum(kMatMulV3IrInputNum, kMatMulV3OutputNum);
    builder.InputTensorDesc(0, x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, withBias ? biasDtype : ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    if (declareY) {
        EXPECT_EQ(holder.GetContext()->SetOutputDataType(0, declaredY), ge::GRAPH_SUCCESS);
    } else {
        // builder不提供输出dtype设置,输出槽残留DT_MAX;归一为DT_UNDEFINED模拟真实GE未声明输出的语义
        EXPECT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    }
    return holder;
}
} // namespace

TEST(MatMulV3GraphInfer, InferDataTypeFp16OutFp16)
{
    auto holder = BuildMatMulV3DtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeMatMulV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(MatMulV3GraphInfer, InferDataTypeFp32OutFp32)
{
    auto holder = BuildMatMulV3DtypeContext(ge::DT_FLOAT, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeMatMulV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(MatMulV3GraphInfer, InferDataTypeBf16OutBf16)
{
    auto holder = BuildMatMulV3DtypeContext(ge::DT_BF16, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeMatMulV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

TEST(MatMulV3GraphInfer, BiasSameDtypeAsX1)
{
    auto holder = BuildMatMulV3DtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, true, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeMatMulV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(MatMulV3GraphInfer, BiasFp32WithFp16Input)
{
    auto holder = BuildMatMulV3DtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, true, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeMatMulV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// bias仅允许与x1同dtype或fp32
TEST(MatMulV3GraphInfer, BiasInvalidDtypeFailed)
{
    auto holder = BuildMatMulV3DtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, true, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeMatMulV3(context), ge::GRAPH_FAILED);
}

TEST(MatMulV3GraphInfer, X1X2MismatchFailed)
{
    auto holder = BuildMatMulV3DtypeContext(ge::DT_FLOAT16, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeMatMulV3(context), ge::GRAPH_FAILED);
}

// 显式声明y: fp16输入声明fp32(16进32出)被尊重
TEST(MatMulV3GraphInfer, DeclaredFp32OutWithFp16Input)
{
    auto holder = BuildMatMulV3DtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, false, ge::DT_UNDEFINED, true,
                                            ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeMatMulV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

// 显式声明y: 与x1一致时被尊重
TEST(MatMulV3GraphInfer, DeclaredYSameAsX1)
{
    auto holder = BuildMatMulV3DtypeContext(ge::DT_BF16, ge::DT_BF16, false, ge::DT_UNDEFINED, true, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeMatMulV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

// 显式声明y: fp16输入声明INT8既非同dtype也非16进32出,失败
TEST(MatMulV3GraphInfer, DeclaredInvalidYFailed)
{
    auto holder = BuildMatMulV3DtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, false, ge::DT_UNDEFINED, true, ge::DT_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeMatMulV3(context), ge::GRAPH_FAILED);
}

// 声明为UNDEFINED视同未声明,y跟随x1
TEST(MatMulV3GraphInfer, DeclaredUndefinedYFollowsX1)
{
    auto holder = BuildMatMulV3DtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, false, ge::DT_UNDEFINED, true,
                                            ge::DT_UNDEFINED);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeMatMulV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// 输出槽残留DT_MAX(builder未归一)时按未声明处理,y跟随x1
TEST(MatMulV3GraphInfer, ResidualDtMaxTreatedAsUndeclared)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("MatMulV3").OpName("MatMulV3");
    builder.IONum(kMatMulV3IrInputNum, kMatMulV3OutputNum);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->GetOutputDataType(0), ge::DT_MAX);
    ASSERT_EQ(ops::InferDataTypeMatMulV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(MatMulV3GraphInfer, RejectsNullContext) { EXPECT_EQ(ops::InferDataTypeMatMulV3(nullptr), ge::GRAPH_FAILED); }
