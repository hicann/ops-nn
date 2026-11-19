/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_adaptive_avg_pool2d_grad_infer_pass.cpp
 * \brief AdaptiveAvgPool2dGrad InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace ops {
ge::graphStatus InferDataTypeAdaptiveAvgPool2dGrad(gert::InferDataTypeContext* context);
}

namespace {
void RunAdaptiveAvgPool2dGradInferDataType(ge::DataType inDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("AdaptiveAvgPool2dGrad");
    builder.OpName("AdaptiveAvgPool2dGrad");
    builder.IONum(1, 1);
    for (int32_t i = 0; i < 1; ++i) {
        builder.InputTensorDesc(i, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < 1; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeAdaptiveAvgPool2dGrad(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inDtype); // out0=grad
}
} // namespace

TEST(AdaptiveAvgPool2dGradGraphInfer, InferDataTypeBfloat16) { RunAdaptiveAvgPool2dGradInferDataType(ge::DT_BF16); }

TEST(AdaptiveAvgPool2dGradGraphInfer, InferDataTypeFloat16) { RunAdaptiveAvgPool2dGradInferDataType(ge::DT_FLOAT16); }

TEST(AdaptiveAvgPool2dGradGraphInfer, InferDataTypeFloat) { RunAdaptiveAvgPool2dGradInferDataType(ge::DT_FLOAT); }

TEST(AdaptiveAvgPool2dGradGraphInfer, InferDataTypeInvalidDtypeRejected)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("AdaptiveAvgPool2dGrad");
    builder.OpName("AdaptiveAvgPool2dGrad");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    EXPECT_EQ(ops::InferDataTypeAdaptiveAvgPool2dGrad(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST(AdaptiveAvgPool2dGradGraphInfer, RegisteredInferDataType)
{
    auto registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::OppSoDesc graphSo({ge::AscendString(ADAPTIVE_AVG_POOL2D_GRAD_GRAPH_REGISTRY_SO)},
                            "adaptive_avg_pool2d_grad_graph_ut");
    ASSERT_EQ(registry->AddSoToRegistry(graphSo), ge::GRAPH_SUCCESS);
    const auto* impl = registry->GetOpImpl("AdaptiveAvgPool2dGrad");
    ASSERT_NE(impl, nullptr);
    EXPECT_NE(impl->infer_datatype, nullptr);
}
