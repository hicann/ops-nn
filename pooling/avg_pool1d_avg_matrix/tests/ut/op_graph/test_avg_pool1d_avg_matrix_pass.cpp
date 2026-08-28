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
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForAvgPool1DAvgMatrix(gert::InferDataTypeContext* context);
}

TEST(AvgPool1DAvgMatrixGraphInfer, PreservesInputDataType)
{
    constexpr ge::DataType inputDataType = ge::DT_INT16;
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("AvgPool1DAvgMatrix").OpName("AvgPool1DAvgMatrix");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, inputDataType, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.OutputTensorDesc(0, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    ASSERT_EQ(ops::InferDataTypeForAvgPool1DAvgMatrix(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inputDataType);
}

TEST(AvgPool1DAvgMatrixGraphInfer, RejectsNullContext)
{
    EXPECT_EQ(ops::InferDataTypeForAvgPool1DAvgMatrix(nullptr), ge::GRAPH_FAILED);
}
