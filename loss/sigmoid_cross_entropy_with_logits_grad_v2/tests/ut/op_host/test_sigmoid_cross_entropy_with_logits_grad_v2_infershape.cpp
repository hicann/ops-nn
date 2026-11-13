/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {

void CheckGradientShape(const gert::Shape& predict, const std::vector<int64_t>& expected, const std::string& reduction,
                        bool withWeights = false, const gert::Shape& dout = {})
{
    gert::StorageShape predictShape;
    predictShape.MutableOriginShape() = predict;
    predictShape.MutableStorageShape() = predict;
    gert::StorageShape doutShape;
    doutShape.MutableOriginShape() = dout;
    doutShape.MutableStorageShape() = dout;
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {{predictShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                                          {predictShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                                          {doutShape, ge::DT_FLOAT, ge::FORMAT_ND}};
    if (withWeights) {
        inputs.emplace_back(gert::StorageShape{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND);
        inputs.emplace_back(gert::StorageShape{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    }
    gert::InfershapeContextPara para("SigmoidCrossEntropyWithLogitsGradV2", inputs,
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}},
                                     {1, 1, 1, withWeights ? 1U : 0U, withWeights ? 1U : 0U}, {1});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {expected});
}

TEST(SigmoidCrossEntropyWithLogitsGradV2Infer, AllReductionsKeepPredictShape)
{
    for (const auto* reduction : {"none", "mean", "sum"}) {
        SCOPED_TRACE(reduction);
        CheckGradientShape({2, 3, 4}, {2, 3, 4}, reduction);
    }
}

TEST(SigmoidCrossEntropyWithLogitsGradV2Infer, AllReductionsWithSameShapeDout)
{
    for (const auto* reduction : {"none", "mean", "sum"}) {
        SCOPED_TRACE(reduction);
        CheckGradientShape({2, 3, 4}, {2, 3, 4}, reduction, false, {2, 3, 4});
    }
}

TEST(SigmoidCrossEntropyWithLogitsGradV2Infer, AllReductionsWithBroadcastDout)
{
    for (const auto* reduction : {"none", "mean", "sum"}) {
        SCOPED_TRACE(reduction);
        CheckGradientShape({2, 3, 4}, {2, 3, 4}, reduction, false, {4});
    }
}

TEST(SigmoidCrossEntropyWithLogitsGradV2Infer, OptionalWeights)
{
    CheckGradientShape({2, 3, 4}, {2, 3, 4}, "mean", true);
}

TEST(SigmoidCrossEntropyWithLogitsGradV2Infer, DynamicShape) { CheckGradientShape({-1, 3, 4}, {-1, 3, 4}, "mean"); }

TEST(SigmoidCrossEntropyWithLogitsGradV2Infer, UnknownRank) { CheckGradientShape({-2}, {-2}, "sum"); }

TEST(SigmoidCrossEntropyWithLogitsGradV2Infer, ScalarAndEmpty)
{
    CheckGradientShape({}, {}, "mean");
    CheckGradientShape({0, 4}, {0, 4}, "none");
}

TEST(SigmoidCrossEntropyWithLogitsGradV2Infer, InferDataType)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* impl = registry->GetOpImpl("SigmoidCrossEntropyWithLogitsGradV2");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("SigmoidCrossEntropyWithLogitsGradV2").OpName("SigmoidCrossEntropyWithLogitsGradV2").IONum(3, 1);
        builder.InputTensorDesc(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
        auto holder = builder.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
        ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), dtype);
    }
}
} // namespace
