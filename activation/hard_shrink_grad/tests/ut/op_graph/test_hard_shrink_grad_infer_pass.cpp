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
 * \file test_hard_shrink_grad_infer_pass.cpp
 * \brief HardShrinkGrad InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "log/log.h"

// 直连函数原型（模式 = mse_loss test_mse_loss_infer_pass.cpp：op_graph UT 骨架的
// SetSpaceRegistry 重建默认 registry，IMPL_OP 注册不可经 registry 查询触达）
namespace ops {
ge::graphStatus InferDataType4HardShrinkGrad(gert::InferDataTypeContext* context);
}

namespace {
void RunHardShrinkGradInferDataType(ge::DataType gradientsDtype, ge::DataType featuresDtype,
                                    ge::graphStatus expectedStatus, ge::DataType expectedOutputDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("HardShrinkGrad");
    builder.OpName("HardShrinkGrad");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, gradientsDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, featuresDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataType4HardShrinkGrad(context), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(context->GetOutputDataType(0), expectedOutputDtype);
    }
}
} // namespace

// Positive: supported dtypes propagate to the output with a consistent input pair.
TEST(HardShrinkGradGraphInfer, InferDataTypeFloat)
{
    RunHardShrinkGradInferDataType(ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, ge::DT_FLOAT);
}

TEST(HardShrinkGradGraphInfer, InferDataTypeFloat16)
{
    RunHardShrinkGradInferDataType(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, ge::DT_FLOAT16);
}

TEST(HardShrinkGradGraphInfer, InferDataTypeBfloat16)
{
    RunHardShrinkGradInferDataType(ge::DT_BF16, ge::DT_BF16, ge::GRAPH_SUCCESS, ge::DT_BF16);
}

// Negative: dtype outside the supported list must be rejected.
TEST(HardShrinkGradGraphInfer, InferDataTypeUnsupportedRejected)
{
    RunHardShrinkGradInferDataType(ge::DT_BOOL, ge::DT_BOOL, ge::GRAPH_FAILED, ge::DT_UNDEFINED);
    RunHardShrinkGradInferDataType(ge::DT_COMPLEX128, ge::DT_COMPLEX128, ge::GRAPH_FAILED, ge::DT_UNDEFINED);
}

// Negative: gradients and features dtypes differ, the consistency check must fail.
TEST(HardShrinkGradGraphInfer, InferDataTypeMismatchRejected)
{
    RunHardShrinkGradInferDataType(ge::DT_FLOAT16, ge::DT_FLOAT, ge::GRAPH_FAILED, ge::DT_UNDEFINED);
    RunHardShrinkGradInferDataType(ge::DT_FLOAT, ge::DT_BF16, ge::GRAPH_FAILED, ge::DT_UNDEFINED);
}
