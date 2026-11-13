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

void CheckRenormShape(const gert::Shape& input, int64_t dim, const std::vector<int64_t>& expected)
{
    gert::StorageShape inputShape;
    inputShape.MutableOriginShape() = input;
    inputShape.MutableStorageShape() = input;
    gert::InfershapeContextPara para("Renorm", {{inputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{"p", Ops::NN::AnyValue::CreateFrom<float>(2.0f)},
                                      {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dim)},
                                      {"maxnorm", Ops::NN::AnyValue::CreateFrom<float>(3.0f)}});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {expected});
}

TEST(RenormInfer, PositiveAxis)
{
    CheckRenormShape({2, 3, 4}, 0, {2, 3, 4});
    CheckRenormShape({2, 3, 4}, 1, {2, 3, 4});
    CheckRenormShape({2, 3, 4}, 2, {2, 3, 4});
}

TEST(RenormInfer, NegativeAxis)
{
    CheckRenormShape({2, 3, 4}, -1, {2, 3, 4});
    CheckRenormShape({2, 3, 4}, -3, {2, 3, 4});
}

TEST(RenormInfer, DynamicDimensions)
{
    CheckRenormShape({2, -1, 4}, 1, {2, -1, 4});
    CheckRenormShape({-1, 3, -1}, -2, {-1, 3, -1});
}

TEST(RenormInfer, UnknownRank)
{
    CheckRenormShape({-2}, 2, {-2});
    CheckRenormShape({-2}, -3, {-2});
}

TEST(RenormInfer, OneDimensionAndEmptyDimension)
{
    CheckRenormShape({5}, -1, {5});
    CheckRenormShape({2, 0, 4}, 1, {2, 0, 4});
}

TEST(RenormInfer, InferDataType)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* impl = registry->GetOpImpl("Renorm");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("Renorm").OpName("Renorm").IONum(1, 1);
        builder.InputTensorDesc(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
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
