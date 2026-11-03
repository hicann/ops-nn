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
 * \file test_quant_update_scatter_graph_infer_pass.cpp
 * \brief QuantUpdateScatter graph InferDataType UT
 */

#include <tuple>
#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeQuantUpdateScatter(gert::InferDataTypeContext* context);
} // namespace ops

// 按真实契约参数化: var/indices/updates/quant_scales/quant_zero_points 各路独立 dtype 集
class QuantUpdateScatterInferDataTypeTest
    : public testing::TestWithParam<std::tuple<ge::DataType, ge::DataType, ge::DataType, ge::DataType, ge::DataType>> {
};

TEST_P(QuantUpdateScatterInferDataTypeTest, quant_update_scatter_inferdatatype_test)
{
    auto inferDataTypeFunc = ops::InferDataTypeQuantUpdateScatter;

    auto param = GetParam();
    ge::DataType varDtype = std::get<0>(param);
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("QuantUpdateScatter").OpName("quant_update_scatter_dtype").IONum(5, 1);
    builder.InputTensorDesc(0, varDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, std::get<1>(param), ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, std::get<2>(param), ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(3, std::get<3>(param), ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(4, std::get<4>(param), ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inferDataTypeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputDataType(0), varDtype);
}

INSTANTIATE_TEST_SUITE_P(
    QuantUpdateScatterInferDataTypeSuite, QuantUpdateScatterInferDataTypeTest,
    testing::Values(std::make_tuple(ge::DT_INT8, ge::DT_INT32, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16),
                    std::make_tuple(ge::DT_INT8, ge::DT_INT64, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32),
                    std::make_tuple(ge::DT_HIFLOAT8, ge::DT_INT32, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16),
                    std::make_tuple(ge::DT_FLOAT8_E4M3FN, ge::DT_INT64, ge::DT_BF16, ge::DT_BF16, ge::DT_INT32),
                    std::make_tuple(ge::DT_FLOAT8_E5M2, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16)));
