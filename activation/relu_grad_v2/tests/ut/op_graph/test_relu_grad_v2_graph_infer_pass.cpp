/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_relu_grad_v2_graph_infer_pass.cpp
 * \brief ReluGradV2 graph InferDataType UT
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeReluGradV2(gert::InferDataTypeContext* context);
} // namespace ops

class ReluGradV2InferDataTypeTest : public testing::TestWithParam<std::tuple<ge::DataType, ge::DataType>> {};

TEST_P(ReluGradV2InferDataTypeTest, relu_grad_v2_inferdatatype_test)
{
    auto inferDataTypeFunc = ops::InferDataTypeReluGradV2;

    ge::DataType gradDtype = std::get<0>(GetParam());
    ge::DataType maskDtype = std::get<1>(GetParam());
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("ReluGradV2").OpName("relu_grad_v2_dtype").IONum(2, 1);
    builder.InputTensorDesc(0, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, maskDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inferDataTypeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputDataType(0), gradDtype);
}

INSTANTIATE_TEST_SUITE_P(
    ReluGradV2InferDataTypeSuite, ReluGradV2InferDataTypeTest,
    testing::Values(std::make_tuple(ge::DT_BF16, ge::DT_UINT8), std::make_tuple(ge::DT_FLOAT16, ge::DT_UINT8),
                    std::make_tuple(ge::DT_FLOAT, ge::DT_UINT8), std::make_tuple(ge::DT_BF16, ge::DT_UINT1),
                    std::make_tuple(ge::DT_FLOAT16, ge::DT_UINT1), std::make_tuple(ge::DT_FLOAT, ge::DT_UINT1)));
