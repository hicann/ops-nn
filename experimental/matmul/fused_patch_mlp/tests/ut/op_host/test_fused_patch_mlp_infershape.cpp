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

#include "../../../op_graph/fused_patch_mlp_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"

namespace {

void RunInferCase(ge::DataType dtype)
{
    ge::op::FusedPatchMlp op;
    const ge::DataType biasDtype = dtype == ge::DT_BF16 ? ge::DT_FLOAT : dtype;
    op.UpdateInputDesc("x", create_desc({4, 1, 64}, dtype));
    op.UpdateInputDesc("weights", create_desc({147456}, dtype));
    op.UpdateInputDesc("biases", create_desc({768}, biasDtype));
    op.set_attr_num_layers(3);

    Runtime2TestParam param;
    param.attrs = {"num_layers"};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);
}

TEST(FusedPatchMlpInferShapeTest, SupportsAllDtypes)
{
    RunInferCase(ge::DT_FLOAT16);
    RunInferCase(ge::DT_BF16);
    RunInferCase(ge::DT_FLOAT);
}

} // namespace
