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
 * \file test_ge_glu_v2_graph_infer_pass.cpp
 * \brief GeGluV2 graph InferDataType UT
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeGeGluV2(gert::InferDataTypeContext* context);
} // namespace ops

class GeGluV2InferDataTypeTest : public testing::TestWithParam<ge::DataType> {};

TEST_P(GeGluV2InferDataTypeTest, ge_glu_v2_inferdatatype_test)
{
    auto inferDataTypeFunc = ops::InferDataTypeGeGluV2;

    ge::DataType inputDtype = GetParam();
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("GeGluV2").OpName("ge_glu_v2_dtype").IONum(1, 2);
    builder.InputTensorDesc(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    ASSERT_EQ(holder.GetContext()->SetOutputDataType(1, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inferDataTypeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputDataType(0), inputDtype);
    EXPECT_EQ(holder.GetContext()->GetOutputDataType(1), inputDtype);
}

INSTANTIATE_TEST_SUITE_P(GeGluV2InferDataTypeSuite, GeGluV2InferDataTypeTest,
                         testing::Values(ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT));
