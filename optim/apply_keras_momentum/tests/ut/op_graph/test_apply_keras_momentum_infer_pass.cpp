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
 * \file test_apply_keras_momentum_infer_pass.cpp
 * \brief ApplyKerasMomentum InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace ops {
ge::graphStatus InferDataTypeApplyKerasMomentum(gert::InferDataTypeContext* context);
}

namespace {
void RunApplyKerasMomentumInferDataType(ge::DataType inDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("ApplyKerasMomentum");
    builder.OpName("ApplyKerasMomentum");
    builder.IONum(5, 1);
    builder.InputTensorDesc(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);      // var
    builder.InputTensorDesc(1, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);      // accum
    builder.InputTensorDesc(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND); // lr 恒为FLOAT
    builder.InputTensorDesc(3, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);      // grad
    builder.InputTensorDesc(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND); // momentum 恒为FLOAT
    for (int32_t i = 0; i < 1; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeApplyKerasMomentum(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inDtype); // out0=var
}
} // namespace

TEST(ApplyKerasMomentumGraphInfer, InferDataTypeBfloat16) { RunApplyKerasMomentumInferDataType(ge::DT_BF16); }

TEST(ApplyKerasMomentumGraphInfer, InferDataTypeFloat16) { RunApplyKerasMomentumInferDataType(ge::DT_FLOAT16); }

TEST(ApplyKerasMomentumGraphInfer, InferDataTypeFloat) { RunApplyKerasMomentumInferDataType(ge::DT_FLOAT); }

TEST(ApplyKerasMomentumGraphInfer, RegisteredInferDataType)
{
    auto registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::OppSoDesc graphSo({ge::AscendString(APPLY_KERAS_MOMENTUM_GRAPH_REGISTRY_SO)},
                            "apply_keras_momentum_graph_ut");
    ASSERT_EQ(registry->AddSoToRegistry(graphSo), ge::GRAPH_SUCCESS);
    const auto* impl = registry->GetOpImpl("ApplyKerasMomentum");
    ASSERT_NE(impl, nullptr);
    EXPECT_NE(impl->infer_datatype, nullptr);
}
