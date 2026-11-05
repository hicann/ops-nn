/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_map_index_graph_infer_pass.cpp
 * \brief MapIndex graph InferDataType UT
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeMapIndex(gert::InferDataTypeContext* context);
} // namespace ops

class MapIndexInferDataTypeTest : public testing::TestWithParam<ge::DataType> {};

TEST_P(MapIndexInferDataTypeTest, map_index_inferdatatype_test)
{
    auto inferDataTypeFunc = ops::InferDataTypeMapIndex;

    ge::DataType inputDtype = GetParam();
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("MapIndex").OpName("map_index_dtype").IONum(3, 1);
    builder.InputTensorDesc(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inferDataTypeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    // 输出 y 为映射索引结果，dtype 固定 INT32，不随输入变化
    EXPECT_EQ(holder.GetContext()->GetOutputDataType(0), ge::DT_INT32);
}

INSTANTIATE_TEST_SUITE_P(MapIndexInferDataTypeSuite, MapIndexInferDataTypeTest,
                         testing::Values(ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT64));
