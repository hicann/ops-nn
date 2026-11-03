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
 * \file test_gelu_grad_v2_graph_infer_pass.cpp
 * \brief GeluGradV2 graph InferDataType UT (注册于 op_graph/gelu_grad_v2_graph_infer.cpp 的 IMPL_OP 独立链)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeGeluGradV2(gert::InferDataTypeContext* context);
} // namespace ops

class GeluGradV2InferDataTypeTest
    : public testing::TestWithParam<std::tuple<ge::DataType, ge::DataType, ge::DataType>> {};

TEST_P(GeluGradV2InferDataTypeTest, gelu_grad_v2_inferdatatype_test)
{
    auto inferDataTypeFunc = ops::InferDataTypeGeluGradV2;

    auto paramTuple = GetParam();
    ge::DataType dyDtype = std::get<0>(paramTuple);
    ge::DataType xDtype = std::get<1>(paramTuple);
    ge::DataType expectedDtype = std::get<2>(paramTuple);
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    auto holder = gert::OpInferDataTypeContextBuilder()
                      .OpType("GeluGradV2")
                      .OpName("gelu_grad_v2_dtype")
                      .IONum(2, 1)
                      .InputTensorDesc(0, dyDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), expectedDtype);
}

INSTANTIATE_TEST_SUITE_P(GeluGradV2InferDataTypeSuite, GeluGradV2InferDataTypeTest,
                         testing::Values(std::make_tuple(ge::DT_BF16, ge::DT_BF16, ge::DT_BF16),
                                         std::make_tuple(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16),
                                         std::make_tuple(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT),
                                         std::make_tuple(ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT),
                                         std::make_tuple(ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT)));
