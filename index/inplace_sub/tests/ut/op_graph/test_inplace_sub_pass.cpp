/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>

#include "../../../op_graph/inplace_sub_proto.h"

namespace {
ge::graphStatus RunRegisteredVerifyCase(const ge::Shape& xShape)
{
    ge::op::InplaceSub op("inplace_sub_verify_ut");
    const ge::Shape indicesShape({1});
    const ge::Shape valueShape({1});
    EXPECT_EQ(op.UpdateInputDesc("x", ge::TensorDesc(xShape, ge::FORMAT_ND, ge::DT_FLOAT)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateInputDesc("indices", ge::TensorDesc(indicesShape, ge::FORMAT_ND, ge::DT_INT32)),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateInputDesc("v", ge::TensorDesc(valueShape, ge::FORMAT_ND, ge::DT_FLOAT)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateOutputDesc("y", ge::TensorDesc(xShape, ge::FORMAT_ND, ge::DT_FLOAT)), ge::GRAPH_SUCCESS);
    return op.VerifyAllAttr(true);
}
} // namespace

TEST(InplaceSubGraphVerifyTest, rejects_scalar_x)
{
    EXPECT_EQ(RunRegisteredVerifyCase(ge::Shape(std::vector<int64_t>{})), ge::GRAPH_FAILED);
}

TEST(InplaceSubGraphVerifyTest, accepts_rank_one_x)
{
    EXPECT_EQ(RunRegisteredVerifyCase(ge::Shape({8})), ge::GRAPH_SUCCESS);
}

TEST(InplaceSubGraphVerifyTest, accepts_unknown_rank_x)
{
    EXPECT_EQ(RunRegisteredVerifyCase(ge::Shape(ge::UNKNOWN_RANK)), ge::GRAPH_SUCCESS);
}
