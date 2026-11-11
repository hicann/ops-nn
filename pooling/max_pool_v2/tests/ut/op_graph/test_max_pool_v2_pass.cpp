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

#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForMaxPoolV2Graph(gert::InferDataTypeContext* context);
}

namespace {
void RunMaxPoolV2GraphInferDataType(ge::DataType inDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("MaxPoolV2");
    builder.OpName("MaxPoolV2");
    builder.IONum(3, 1);
    builder.InputTensorDesc(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForMaxPoolV2Graph(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inDtype);
}

TEST(MaxPoolV2GraphInfer, InferDataTypeFloat16) { RunMaxPoolV2GraphInferDataType(ge::DT_FLOAT16); }

TEST(MaxPoolV2GraphInfer, InferDataTypeFloat) { RunMaxPoolV2GraphInferDataType(ge::DT_FLOAT); }
} // namespace
