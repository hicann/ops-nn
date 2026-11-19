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
 * \file test_avg_pool_grad_infer_pass.cpp
 * \brief AvgPoolGrad InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace ops {
ge::graphStatus InferDataTypeAvgPoolGrad(gert::InferDataTypeContext* context);
}

namespace {
void RunAvgPoolGradInferDataType(ge::DataType gradDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("AvgPoolGrad");
    builder.OpName("AvgPoolGrad");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    for (int32_t i = 0; i < 1; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeAvgPoolGrad(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), gradDtype); // out0=grad(input1)
}
} // namespace

TEST(AvgPoolGradGraphInfer, InferDataTypeBfloat16) { RunAvgPoolGradInferDataType(ge::DT_BF16); }

TEST(AvgPoolGradGraphInfer, InferDataTypeFloat16) { RunAvgPoolGradInferDataType(ge::DT_FLOAT16); }

TEST(AvgPoolGradGraphInfer, InferDataTypeFloat) { RunAvgPoolGradInferDataType(ge::DT_FLOAT); }

TEST(AvgPoolGradGraphInfer, InferDataTypeInt32) { RunAvgPoolGradInferDataType(ge::DT_INT32); }

TEST(AvgPoolGradGraphInfer, RegisteredInferDataType)
{
    auto registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::OppSoDesc graphSo({ge::AscendString(AVG_POOL_GRAD_GRAPH_REGISTRY_SO)}, "avg_pool_grad_graph_ut");
    ASSERT_EQ(registry->AddSoToRegistry(graphSo), ge::GRAPH_SUCCESS);
    const auto* impl = registry->GetOpImpl("AvgPoolGrad");
    ASSERT_NE(impl, nullptr);
    EXPECT_NE(impl->infer_datatype, nullptr);
}
