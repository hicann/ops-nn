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
 * \file test_leaky_relu_grad_graph_infer_pass.cpp
 * \brief LeakyReluGrad graph InferDataType UT
 */

#include <tuple>
#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeLeakyReluGrad(gert::InferDataTypeContext* context);
} // namespace ops

class LeakyReluGradInferDataTypeTest : public testing::TestWithParam<std::tuple<ge::DataType, ge::DataType>> {};

TEST_P(LeakyReluGradInferDataTypeTest, leaky_relu_grad_inferdatatype_test)
{
    auto inferDataTypeFunc = ops::InferDataTypeLeakyReluGrad;

    auto param = GetParam();
    ge::DataType firstDtype = std::get<0>(param);
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("LeakyReluGrad").OpName("leaky_relu_grad_dtype").IONum(2, 1);
    builder.InputTensorDesc(0, firstDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, std::get<1>(param), ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inferDataTypeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputDataType(0), firstDtype);
}

INSTANTIATE_TEST_SUITE_P(LeakyReluGradInferDataTypeSuite, LeakyReluGradInferDataTypeTest,
                         testing::Values(std::make_tuple(ge::DT_BF16, ge::DT_BF16),
                                         std::make_tuple(ge::DT_FLOAT16, ge::DT_FLOAT16),
                                         std::make_tuple(ge::DT_FLOAT, ge::DT_FLOAT),
                                         std::make_tuple(ge::DT_BF16, ge::DT_FLOAT16),
                                         std::make_tuple(ge::DT_FLOAT16, ge::DT_BF16)));
