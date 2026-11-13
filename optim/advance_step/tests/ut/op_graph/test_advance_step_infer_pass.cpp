/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_advance_step_infer_pass.cpp
 * \brief AdvanceStep 算子 RT2.0 InferDataType UT（按转测布局规则自 op_host 拆分后随 op_graph 测试）
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataType4AdvanceStep(gert::InferDataTypeContext* context);
}

// 4 个 inplace 状态张量输出与 input_tokens 同 dtype
TEST(AdvanceStepGraphInfer, InferDataTypeInt64)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("AdvanceStep").OpName("AdvanceStep");
    builder.IONum(8, 4);
    for (size_t i = 0; i < 8; i++) {
        builder.InputTensorDesc(i, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (size_t i = 0; i < 4; i++) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    ASSERT_EQ(ops::InferDataType4AdvanceStep(context), ge::GRAPH_SUCCESS);
    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(context->GetOutputDataType(i), ge::DT_INT64);
    }
}

TEST(AdvanceStepGraphInfer, RejectsNullContext)
{
    EXPECT_EQ(ops::InferDataType4AdvanceStep(nullptr), ge::GRAPH_FAILED);
}

// Host dtype 门负例：必选输入非 INT64 直接拒绝
TEST(AdvanceStepGraphInfer, InferDataTypeRejectsNonInt64)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("AdvanceStep").OpName("AdvanceStep");
    builder.IONum(8, 4);
    builder.InputTensorDesc(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    for (size_t i = 1; i < 8; i++) {
        builder.InputTensorDesc(i, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (size_t i = 0; i < 4; i++) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(ops::InferDataType4AdvanceStep(context), ge::GRAPH_FAILED);
}

// Host dtype 门与 tiling 宽容度对齐：可选输入 dtype 不在门内校验
// （tiling 按存在性 numel>0 判定后才查，空张量任意 dtype 视为未传入回落 V1）
TEST(AdvanceStepGraphInfer, InferDataTypeDefersOptionalDtypeToTiling)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("AdvanceStep").OpName("AdvanceStep");
    builder.IONum(8, 4);
    for (size_t i = 0; i < 6; i++) {
        builder.InputTensorDesc(i, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    builder.InputTensorDesc(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(7, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND);
    for (size_t i = 0; i < 4; i++) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(ops::InferDataType4AdvanceStep(context), ge::GRAPH_SUCCESS);
}
