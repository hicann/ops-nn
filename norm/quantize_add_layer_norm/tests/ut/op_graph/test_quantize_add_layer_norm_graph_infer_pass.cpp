/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace ops {
ge::graphStatus InferDataType4QuantizeAddLayerNorm(gert::InferDataTypeContext* context);
}

TEST(QuantizeAddLayerNormGraphInfer, RegistersDataTypeCallback)
{
    auto registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::OppSoDesc graphSo({ge::AscendString(QUANTIZE_ADD_LAYER_NORM_GRAPH_REGISTRY_SO)},
                            "quantize_add_layer_norm_graph_ut");
    ASSERT_EQ(registry->AddSoToRegistry(graphSo), ge::GRAPH_SUCCESS);
    auto* opImpl = registry->GetOpImpl("QuantizeAddLayerNorm");
    ASSERT_NE(opImpl, nullptr);
    EXPECT_NE(opImpl->infer_datatype, nullptr);
}

TEST(QuantizeAddLayerNormGraphInfer, SetsDeclaredOutputTypes)
{
    for (auto inputType : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("QuantizeAddLayerNorm").OpName("quantize_add_layer_norm_dtype").IONum(7, 2);
        for (size_t i = 0; i < 5; ++i) {
            builder.InputTensorDesc(i, inputType, ge::FORMAT_ND, ge::FORMAT_ND);
        }
        builder.InputTensorDesc(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);

        auto holder = builder.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(ops::InferDataType4QuantizeAddLayerNorm(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
        EXPECT_EQ(context->GetOutputDataType(1), inputType);
    }
}
