/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <gtest/gtest.h>
#include <vector>

#include "../../../op_graph/apply_adagrad_proto.h"

namespace {
ge::graphStatus RunRegisteredVerifyCase(const std::array<ge::DataType, 4>& inputDtypes)
{
    ge::op::ApplyAdagrad op("apply_adagrad_verify_ut");
    const ge::Shape tensorShape({8, 16});
    const ge::Shape scalarShape(std::vector<int64_t>{});
    const std::array<const char*, 4> inputNames = {"var", "accum", "lr", "grad"};
    for (size_t inputIdx = 0; inputIdx < inputDtypes.size(); ++inputIdx) {
        const ge::Shape& shape = inputIdx == 2 ? scalarShape : tensorShape;
        EXPECT_EQ(op.UpdateInputDesc(inputNames[inputIdx], ge::TensorDesc(shape, ge::FORMAT_ND, inputDtypes[inputIdx])),
                  ge::GRAPH_SUCCESS);
    }
    EXPECT_EQ(op.UpdateOutputDesc("var", ge::TensorDesc(tensorShape, ge::FORMAT_ND, inputDtypes[0])),
              ge::GRAPH_SUCCESS);
    return op.VerifyAllAttr(true);
}
} // namespace

TEST(ApplyAdagradGraphVerifyTest, accepts_matching_inputs)
{
    EXPECT_EQ(RunRegisteredVerifyCase({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunRegisteredVerifyCase({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunRegisteredVerifyCase({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16}), ge::GRAPH_SUCCESS);
}

TEST(ApplyAdagradGraphVerifyTest, rejects_each_mismatched_input)
{
    EXPECT_EQ(RunRegisteredVerifyCase({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT}), ge::GRAPH_FAILED);
    EXPECT_EQ(RunRegisteredVerifyCase({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16}), ge::GRAPH_FAILED);
    EXPECT_EQ(RunRegisteredVerifyCase({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT}), ge::GRAPH_FAILED);
}
