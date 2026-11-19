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
 * \file test_fake_quant_with_min_max_vars_gradient_infer_pass.cpp
 * \brief FakeQuantWithMinMaxVarsGradient InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace ops {
ge::graphStatus InferDataTypeFakeQuantWithMinMaxVarsGradient(gert::InferDataTypeContext* context);
}

namespace {
void RunFakeQuantWithMinMaxVarsGradientInferDataType(ge::DataType inDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("FakeQuantWithMinMaxVarsGradient");
    builder.OpName("FakeQuantWithMinMaxVarsGradient");
    builder.IONum(4, 3);
    for (int32_t i = 0; i < 4; ++i) {
        builder.InputTensorDesc(i, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < 3; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeFakeQuantWithMinMaxVarsGradient(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT); // out0..2=FLOAT
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT); // out0..2=FLOAT
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT); // out0..2=FLOAT
}
} // namespace

TEST(FakeQuantWithMinMaxVarsGradientGraphInfer, InferDataTypeFloat)
{
    RunFakeQuantWithMinMaxVarsGradientInferDataType(ge::DT_FLOAT);
}

TEST(FakeQuantWithMinMaxVarsGradientGraphInfer, RegisteredInferDataType)
{
    auto registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::OppSoDesc graphSo({ge::AscendString(FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_GRAPH_REGISTRY_SO)},
                            "fake_quant_with_min_max_vars_gradient_graph_ut");
    ASSERT_EQ(registry->AddSoToRegistry(graphSo), ge::GRAPH_SUCCESS);
    const auto* impl = registry->GetOpImpl("FakeQuantWithMinMaxVarsGradient");
    ASSERT_NE(impl, nullptr);
    EXPECT_NE(impl->infer_datatype, nullptr);
}
