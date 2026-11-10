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
 * \file test_max_pool_with_argmax_v3_graph_infer_pass.cpp
 * \brief MaxPoolWithArgmaxV3 graph InferDataType UT (注册于 op_graph/max_pool_with_argmax_v3_graph_infer.cpp 的
 *        IMPL_OP 独立链)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeMaxPoolWithArgmaxV3(gert::InferDataTypeContext* context);
} // namespace ops

class MaxPoolWithArgmaxV3InferDataTypeTest
    : public testing::TestWithParam<std::tuple<ge::DataType, int64_t, ge::DataType>> {};

TEST_P(MaxPoolWithArgmaxV3InferDataTypeTest, max_pool_with_argmax_v3_inferdatatype_test)
{
    auto inferDataTypeFunc = ops::InferDataTypeMaxPoolWithArgmaxV3;

    auto paramTuple = GetParam();
    ge::DataType inputDtype = std::get<0>(paramTuple);
    int64_t dtypeAttr = std::get<1>(paramTuple);
    ge::DataType expectedArgmaxDtype = std::get<2>(paramTuple);

    auto holder = gert::OpInferDataTypeContextBuilder()
                      .OpType("MaxPoolWithArgmaxV3")
                      .OpName("max_pool_with_argmax_v3_dtype")
                      .IONum(1, 2)
                      .InputTensorDesc(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .AppendAttr(std::vector<int64_t>{1, 1})
                      .AppendAttr(std::vector<int64_t>{1, 1})
                      .AppendAttr(std::vector<int64_t>{0, 0})
                      .AppendAttr(dtypeAttr)
                      .AppendAttr(std::vector<int64_t>{1, 1})
                      .AppendAttr(false)
                      .AppendAttr(ge::AscendString("NCHW"))
                      .Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    ASSERT_EQ(context->SetOutputDataType(1, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inputDtype);
    EXPECT_EQ(context->GetOutputDataType(1), expectedArgmaxDtype);
}

INSTANTIATE_TEST_SUITE_P(MaxPoolWithArgmaxV3InferDataTypeSuite, MaxPoolWithArgmaxV3InferDataTypeTest,
                         testing::Values(std::make_tuple(ge::DT_BF16, static_cast<int64_t>(3), ge::DT_INT32),
                                         std::make_tuple(ge::DT_FLOAT16, static_cast<int64_t>(3), ge::DT_INT32),
                                         std::make_tuple(ge::DT_FLOAT, static_cast<int64_t>(3), ge::DT_INT32),
                                         std::make_tuple(ge::DT_BF16, static_cast<int64_t>(9), ge::DT_INT64),
                                         std::make_tuple(ge::DT_FLOAT16, static_cast<int64_t>(9), ge::DT_INT64),
                                         std::make_tuple(ge::DT_FLOAT, static_cast<int64_t>(9), ge::DT_INT64)));
