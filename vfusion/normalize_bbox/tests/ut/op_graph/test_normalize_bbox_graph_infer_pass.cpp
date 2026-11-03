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
ge::graphStatus NormalizeBBoxInferDataType(gert::InferDataTypeContext* context);
}

TEST(NormalizeBBoxGraphInfer, validates_both_input_dtypes)
{
    struct Case {
        ge::DataType boxes;
        ge::DataType shapeHw;
        bool valid;
    };
    const Case cases[] = {
        {ge::DT_FLOAT16, ge::DT_INT32, true}, {ge::DT_FLOAT, ge::DT_INT32, true},  {ge::DT_BF16, ge::DT_INT32, false},
        {ge::DT_BOOL, ge::DT_INT32, false},   {ge::DT_INT32, ge::DT_INT32, false}, {ge::DT_FLOAT, ge::DT_BF16, false},
        {ge::DT_FLOAT, ge::DT_BOOL, false},   {ge::DT_FLOAT, ge::DT_FLOAT, false},
    };
    for (const auto& item : cases) {
        SCOPED_TRACE(static_cast<int>(item.boxes));
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("NormalizeBBox").OpName("normalize_bbox_dtype_check").IONum(2, 1);
        builder.InputTensorDesc(0, item.boxes, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(1, item.shapeHw, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
        auto holder = builder.Build();
        ASSERT_NE(holder.GetContext(), nullptr);
        EXPECT_EQ(ops::NormalizeBBoxInferDataType(holder.GetContext()),
                  item.valid ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED);
        if (item.valid)
            EXPECT_EQ(holder.GetContext()->GetOutputDataType(0), item.boxes);
    }
}
