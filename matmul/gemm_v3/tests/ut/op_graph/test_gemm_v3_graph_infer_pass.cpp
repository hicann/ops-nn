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
 * \file test_gemm_v3_graph_infer_pass.cpp
 * \brief GemmV3 InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForGemmV3(gert::InferDataTypeContext* context);
}

namespace {
constexpr size_t kGemmV3IrInputNum = 3; // a/b/c, c为OPTIONAL
constexpr size_t kGemmV3OutputNum = 1;

gert::ContextHolder<gert::InferDataTypeContext> BuildGemmV3DtypeContext(ge::DataType aDtype, ge::DataType bDtype,
                                                                        bool withC = false,
                                                                        ge::DataType cDtype = ge::DT_UNDEFINED)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("GemmV3").OpName("GemmV3");
    builder.IONum(kGemmV3IrInputNum, kGemmV3OutputNum);
    builder.InputTensorDesc(0, aDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, bDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    // 未实例化的可选输入显式填DT_UNDEFINED(与真实GE未实例化语义一致),避免builder残留值干扰
    builder.InputTensorDesc(2, withC ? cDtype : ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    // builder不提供输出dtype设置,输出槽残留DT_MAX;归一为DT_UNDEFINED模拟真实GE未声明输出的语义
    EXPECT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    return holder;
}
} // namespace

TEST(GemmV3GraphInfer, InferDataTypeFp16OutFp16)
{
    auto holder = BuildGemmV3DtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForGemmV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(GemmV3GraphInfer, InferDataTypeFp32OutFp32)
{
    auto holder = BuildGemmV3DtypeContext(ge::DT_FLOAT, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForGemmV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(GemmV3GraphInfer, InferDataTypeBf16OutBf16)
{
    auto holder = BuildGemmV3DtypeContext(ge::DT_BF16, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForGemmV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

// c已实例化时y跟随c: fp32 c覆盖16进16出(16进32出)
TEST(GemmV3GraphInfer, CFp32OverridesToFp32)
{
    auto holder = BuildGemmV3DtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, true, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForGemmV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

// y跟随c而非a: fp32输入配bf16 c时y为bf16
TEST(GemmV3GraphInfer, YFollowsCBf16)
{
    auto holder = BuildGemmV3DtypeContext(ge::DT_FLOAT, ge::DT_FLOAT, true, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForGemmV3(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

TEST(GemmV3GraphInfer, InvalidADtypeFailed)
{
    auto holder = BuildGemmV3DtypeContext(ge::DT_INT8, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForGemmV3(context), ge::GRAPH_FAILED);
}

TEST(GemmV3GraphInfer, InvalidCDtypeFailed)
{
    auto holder = BuildGemmV3DtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, true, ge::DT_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForGemmV3(context), ge::GRAPH_FAILED);
}

TEST(GemmV3GraphInfer, RejectsNullContext) { EXPECT_EQ(ops::InferDataTypeForGemmV3(nullptr), ge::GRAPH_FAILED); }
