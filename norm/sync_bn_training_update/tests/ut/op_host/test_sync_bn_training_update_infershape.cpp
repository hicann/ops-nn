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

void CheckSyncBNShape(const gert::Shape& runningMean, const std::vector<int64_t>& expected)
{
    // Distinguish running_mean from mean to catch an incorrect input index.
    gert::StorageShape runningMeanShape;
    runningMeanShape.MutableOriginShape() = runningMean;
    runningMeanShape.MutableStorageShape() = runningMean;
    gert::InfershapeContextPara para(
        "SyncBNTrainingUpdate",
        {{{{2, 7}, {2, 7}}, ge::DT_FLOAT, ge::FORMAT_ND}, {runningMeanShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {expected});
}

TEST(SyncBNTrainingUpdateInfer, RunningMeanShape) { CheckSyncBNShape({7}, {7}); }

TEST(SyncBNTrainingUpdateInfer, DynamicShape) { CheckSyncBNShape({-1}, {-1}); }

TEST(SyncBNTrainingUpdateInfer, UnknownRank) { CheckSyncBNShape({-2}, {-2}); }

TEST(SyncBNTrainingUpdateInfer, ScalarAndEmpty)
{
    CheckSyncBNShape({}, {});
    CheckSyncBNShape({0}, {0});
}

TEST(SyncBNTrainingUpdateInfer, InferDataType)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* impl = registry->GetOpImpl("SyncBNTrainingUpdate");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("SyncBNTrainingUpdate").OpName("SyncBNTrainingUpdate").IONum(2, 1);
        builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
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
