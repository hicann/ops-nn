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
 * \file test_glu_grad_graph_infer_pass.cpp
 * \brief GLUGrad graph InferDataType UT (注册于 op_graph/glu_grad_graph_infer.cpp 的 IMPL_OP 独立链)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeGLUGrad(gert::InferDataTypeContext* context);
} // namespace ops

class GLUGradInferDataTypeTest : public testing::TestWithParam<ge::DataType> {};

TEST_P(GLUGradInferDataTypeTest, glu_grad_inferdatatype_test)
{
    auto inferDataTypeFunc = ops::InferDataTypeGLUGrad;

    ge::DataType selfDtype = GetParam();
    // grad_out 刻意取与 self 不同的合法 dtype，验证输出取的是输入 self 而非第一个输入 grad_out
    ge::DataType gradOutDtype = (selfDtype == ge::DT_FLOAT16) ? ge::DT_BF16 : ge::DT_FLOAT16;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    auto holder = gert::OpInferDataTypeContextBuilder()
                      .OpType("GLUGrad")
                      .OpName("glu_grad_dtype")
                      .IONum(2, 1)
                      .InputTensorDesc(0, gradOutDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, selfDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), selfDtype);
}

INSTANTIATE_TEST_SUITE_P(GLUGradInferDataTypeSuite, GLUGradInferDataTypeTest,
                         testing::Values(ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT));
