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
 * \file test_max_pool3d_grad_graph_infer_pass.cpp
 * \brief MaxPool3DGrad InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace ops {
ge::graphStatus InferDataTypeMaxPool3DGrad(gert::InferDataTypeContext* context);
}

class MaxPool3DGradInferDataTypeTest : public testing::TestWithParam<ge::DataType> {};

TEST_P(MaxPool3DGradInferDataTypeTest, max_pool3d_grad_inferdatatype_test)
{
    auto inferDataTypeFunc = ops::InferDataTypeMaxPool3DGrad;

    ge::DataType inputDtype = GetParam();
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("MaxPool3DGrad");
    builder.OpName("max_pool3d_grad_dtype");
    builder.IONum(3, 1);
    builder.InputTensorDesc(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inferDataTypeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputDataType(0), inputDtype);
}

INSTANTIATE_TEST_SUITE_P(MaxPool3DGradInferDataTypeSuite, MaxPool3DGradInferDataTypeTest,
                         testing::Values(ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT));

TEST(MaxPool3DGradGraphInfer, RegisteredInferDataType)
{
    auto registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::OppSoDesc graphSo({ge::AscendString(MAX_POOL3D_GRAD_GRAPH_REGISTRY_SO)}, "max_pool3d_grad_graph_ut");
    ASSERT_EQ(registry->AddSoToRegistry(graphSo), ge::GRAPH_SUCCESS);
    const auto* impl = registry->GetOpImpl("MaxPool3DGrad");
    ASSERT_NE(impl, nullptr);
    EXPECT_NE(impl->infer_datatype, nullptr);
}
