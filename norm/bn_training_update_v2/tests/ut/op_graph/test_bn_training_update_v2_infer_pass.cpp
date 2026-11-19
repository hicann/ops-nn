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
 * \file test_bn_training_update_v2_infer_pass.cpp
 * \brief BNTrainingUpdateV2 InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace ops {
ge::graphStatus InferDataTypeBNTrainingUpdateV2(gert::InferDataTypeContext* context);
}

namespace {
void RunBNTrainingUpdateV2InferDataType(ge::DataType inDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("BNTrainingUpdateV2");
    builder.OpName("BNTrainingUpdateV2");
    builder.IONum(5, 3);
    for (int32_t i = 0; i < 3; ++i) {
        builder.InputTensorDesc(i, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 3; i < 5; ++i) {
        builder.InputTensorDesc(i, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < 3; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBNTrainingUpdateV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inDtype);      // out0=x
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT); // out1,2=scale
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT); // out1,2=scale
}
} // namespace

TEST(BNTrainingUpdateV2GraphInfer, InferDataTypeBfloat16) { RunBNTrainingUpdateV2InferDataType(ge::DT_BF16); }

TEST(BNTrainingUpdateV2GraphInfer, InferDataTypeFloat16) { RunBNTrainingUpdateV2InferDataType(ge::DT_FLOAT16); }

TEST(BNTrainingUpdateV2GraphInfer, InferDataTypeFloat) { RunBNTrainingUpdateV2InferDataType(ge::DT_FLOAT); }

TEST(BNTrainingUpdateV2GraphInfer, RegisteredInferDataType)
{
    auto registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::OppSoDesc graphSo({ge::AscendString(BN_TRAINING_UPDATE_V2_GRAPH_REGISTRY_SO)},
                            "bn_training_update_v2_graph_ut");
    ASSERT_EQ(registry->AddSoToRegistry(graphSo), ge::GRAPH_SUCCESS);
    const auto* impl = registry->GetOpImpl("BNTrainingUpdateV2");
    ASSERT_NE(impl, nullptr);
    EXPECT_NE(impl->infer_datatype, nullptr);
}
