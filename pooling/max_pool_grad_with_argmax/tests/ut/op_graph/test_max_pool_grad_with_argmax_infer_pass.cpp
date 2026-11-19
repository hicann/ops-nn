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
 * \file test_max_pool_grad_with_argmax_infer_pass.cpp
 * \brief MaxPoolGradWithArgmax InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace ops {
ge::graphStatus InferDataTypeMaxPoolGradWithArgmax(gert::InferDataTypeContext* context);
}

namespace {
void RunMaxPoolGradWithArgmaxInferDataType(ge::DataType xDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("MaxPoolGradWithArgmax");
    builder.OpName("MaxPoolGradWithArgmax");
    builder.IONum(3, 1);
    builder.InputTensorDesc(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    for (int32_t i = 0; i < 1; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeMaxPoolGradWithArgmax(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), xDtype); // out0=x
}
} // namespace

TEST(MaxPoolGradWithArgmaxGraphInfer, InferDataTypeBfloat16) { RunMaxPoolGradWithArgmaxInferDataType(ge::DT_BF16); }

TEST(MaxPoolGradWithArgmaxGraphInfer, InferDataTypeFloat16) { RunMaxPoolGradWithArgmaxInferDataType(ge::DT_FLOAT16); }

TEST(MaxPoolGradWithArgmaxGraphInfer, InferDataTypeFloat) { RunMaxPoolGradWithArgmaxInferDataType(ge::DT_FLOAT); }

TEST(MaxPoolGradWithArgmaxGraphInfer, RegisteredInferDataType)
{
    auto registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::OppSoDesc graphSo({ge::AscendString(MAX_POOL_GRAD_WITH_ARGMAX_GRAPH_REGISTRY_SO)},
                            "max_pool_grad_with_argmax_graph_ut");
    ASSERT_EQ(registry->AddSoToRegistry(graphSo), ge::GRAPH_SUCCESS);
    const auto* impl = registry->GetOpImpl("MaxPoolGradWithArgmax");
    ASSERT_NE(impl, nullptr);
    EXPECT_NE(impl->infer_datatype, nullptr);
}
