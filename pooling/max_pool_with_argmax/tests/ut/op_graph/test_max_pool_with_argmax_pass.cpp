/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>

#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForMaxPoolWithArgmaxGraph(gert::InferDataTypeContext* context);
}

namespace {
void RunMaxPoolWithArgmaxGraphInferDataType(ge::DataType inDtype, int64_t targmax, ge::DataType expectArgmax)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("MaxPoolWithArgmax");
    builder.OpName("MaxPoolWithArgmax");
    builder.IONum(1, 2);
    builder.InputTensorDesc(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(std::vector<int64_t>{1, 3, 3, 1});
    builder.AppendAttr(std::vector<int64_t>{1, 2, 2, 1});
    builder.AppendAttr(ge::AscendString("SAME"));
    builder.AppendAttr(targmax);
    builder.AppendAttr(false);
    builder.AppendAttr(ge::AscendString("NHWC"));
    builder.AppendAttr(false);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForMaxPoolWithArgmaxGraph(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inDtype);
    EXPECT_EQ(context->GetOutputDataType(1), expectArgmax);
}

TEST(MaxPoolWithArgmaxGraphInfer, InferDataTypeFloat16TargmaxInt32)
{
    RunMaxPoolWithArgmaxGraphInferDataType(ge::DT_FLOAT16, 3, ge::DT_INT32);
}

TEST(MaxPoolWithArgmaxGraphInfer, InferDataTypeFloatTargmaxInt64)
{
    RunMaxPoolWithArgmaxGraphInferDataType(ge::DT_FLOAT, 9, ge::DT_INT64);
}
} // namespace
