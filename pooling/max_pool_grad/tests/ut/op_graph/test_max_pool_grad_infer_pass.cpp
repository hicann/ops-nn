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
 * \file test_max_pool_grad_infer_pass.cpp
 * \brief MaxPoolGrad InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace ops {
ge::graphStatus InferDataTypeMaxPoolGrad(gert::InferDataTypeContext* context);
}

namespace {
void RunMaxPoolGradInferDataType(ge::DataType inDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("MaxPoolGrad");
    builder.OpName("MaxPoolGrad");
    builder.IONum(3, 1);
    for (int32_t i = 0; i < 3; ++i) {
        builder.InputTensorDesc(i, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < 1; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeMaxPoolGrad(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inDtype); // out0=x1
}
} // namespace

TEST(MaxPoolGradGraphInfer, InferDataTypeBfloat16) { RunMaxPoolGradInferDataType(ge::DT_BF16); }

TEST(MaxPoolGradGraphInfer, InferDataTypeFloat16) { RunMaxPoolGradInferDataType(ge::DT_FLOAT16); }

TEST(MaxPoolGradGraphInfer, InferDataTypeFloat) { RunMaxPoolGradInferDataType(ge::DT_FLOAT); }

TEST(MaxPoolGradGraphInfer, RegisteredInferDataType)
{
    auto registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::OppSoDesc graphSo({ge::AscendString(MAX_POOL_GRAD_GRAPH_REGISTRY_SO)}, "max_pool_grad_graph_ut");
    ASSERT_EQ(registry->AddSoToRegistry(graphSo), ge::GRAPH_SUCCESS);
    const auto* impl = registry->GetOpImpl("MaxPoolGrad");
    ASSERT_NE(impl, nullptr);
    EXPECT_NE(impl->infer_datatype, nullptr);
}
