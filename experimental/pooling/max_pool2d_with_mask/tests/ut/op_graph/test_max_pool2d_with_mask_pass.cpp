/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeMaxPool2dWithMask(gert::InferDataTypeContext* context);
}

namespace {
TEST(MaxPool2dWithMaskInferDataType, PropagatesOutputDtypes)
{
    for (const ge::DataType dtype : {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("MaxPool2dWithMask").OpName("MaxPool2dWithMask");
        builder.IONum(1, 2);
        builder.InputTensorDesc(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
        auto holder = builder.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(ops::InferDataTypeMaxPool2dWithMask(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), dtype);
        EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT8);
    }
}
} // namespace
