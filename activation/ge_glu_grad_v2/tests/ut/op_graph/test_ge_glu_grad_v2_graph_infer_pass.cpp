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
 * \file test_ge_glu_grad_v2_graph_infer_pass.cpp
 * \brief GeGluGradV2 graph InferDataType UT (注册于 op_graph/ge_glu_grad_v2_graph_infer.cpp 的 IMPL_OP 独立链)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeGeGluGradV2(gert::InferDataTypeContext* context);
} // namespace ops

class GeGluGradV2InferDataTypeTest : public testing::TestWithParam<ge::DataType> {};

TEST_P(GeGluGradV2InferDataTypeTest, ge_glu_grad_v2_inferdatatype_test)
{
    auto inferDataTypeFunc = ops::InferDataTypeGeGluGradV2;

    ge::DataType xDtype = GetParam();
    // dy 刻意取与 x 不同的合法 dtype，验证输出取的是输入 x 而非第一个输入 dy
    ge::DataType dyDtype = (xDtype == ge::DT_FLOAT16) ? ge::DT_BF16 : ge::DT_FLOAT16;
    ge::DataType geluDtype = xDtype;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    auto holder = gert::OpInferDataTypeContextBuilder()
                      .OpType("GeGluGradV2")
                      .OpName("ge_glu_grad_v2_dtype")
                      .IONum(3, 1)
                      .InputTensorDesc(0, dyDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(2, geluDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), xDtype);
}

INSTANTIATE_TEST_SUITE_P(GeGluGradV2InferDataTypeSuite, GeGluGradV2InferDataTypeTest,
                         testing::Values(ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT));
