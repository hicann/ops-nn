/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <gtest/gtest.h>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/swiglu_group_quant_with_dual_axis_proto.h"

namespace {
const Runtime2TestParam kV2RuntimeParam{{"dst_type", "quant_mode", "clamp_limit", "output_origin", "alpha", "bias"}};

void SetInput(ge::op::SwigluGroupQuantWithDualAxis& op, const char* name, const std::vector<int64_t>& dims,
              ge::DataType dtype)
{
    ge::TensorDesc desc;
    ge::Shape shape(dims);
    desc.SetDataType(dtype);
    desc.SetShape(shape);
    desc.SetOriginShape(shape);
    op.UpdateInputDesc(name, desc);
}

TEST(SwigluGroupQuantWithDualAxisInferShape, RejectsRankThree)
{
    ge::op::SwigluGroupQuantWithDualAxis op;
    SetInput(op, "x", {2, 6, 256}, ge::DT_FLOAT16);
    op.SetAttr("quant_mode", static_cast<int64_t>(1));

    EXPECT_EQ(InferShapeTest(op, kV2RuntimeParam), ge::GRAPH_FAILED);
}

TEST(SwigluGroupQuantWithDualAxisInferShape, GroupedOutputsStillCoverAllRows)
{
    ge::op::SwigluGroupQuantWithDualAxis op;
    SetInput(op, "x", {128, 128}, ge::DT_BF16);
    // InferShapeTest compacts disconnected optional descriptors. Connect the
    // preceding optional input so group_index keeps its production IR index 2.
    SetInput(op, "weight", {128}, ge::DT_FLOAT);
    SetInput(op, "group_index", {3}, ge::DT_INT64);

    ASSERT_EQ(InferShapeTest(op, kV2RuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({128, 64}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({128, 1, 2}));
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), std::vector<int64_t>({128, 64}));
    EXPECT_EQ(op.GetOutputDesc(3).GetShape().GetDims(), std::vector<int64_t>({5, 64, 2}));
}

TEST(SwigluGroupQuantWithDualAxisInferShape, NonGroupExactBlockHasNoExtraScaleRow)
{
    ge::op::SwigluGroupQuantWithDualAxis op;
    SetInput(op, "x", {128, 128}, ge::DT_FLOAT16);

    ASSERT_EQ(InferShapeTest(op, kV2RuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(3).GetShape().GetDims(), std::vector<int64_t>({2, 64, 2}));
}

TEST(SwigluGroupQuantWithDualAxisInferShape, RejectsRetiredModeFive)
{
    ge::op::SwigluGroupQuantWithDualAxis op;
    SetInput(op, "x", {65, 128}, ge::DT_FLOAT16);
    op.SetAttr("quant_mode", static_cast<int64_t>(5));

    EXPECT_EQ(InferShapeTest(op, kV2RuntimeParam), ge::GRAPH_FAILED);
}

TEST(SwigluGroupQuantWithDualAxisInferShape, RejectsNonCanonicalLastDimension)
{
    ge::op::SwigluGroupQuantWithDualAxis op;
    SetInput(op, "x", {4, 96}, ge::DT_FLOAT16);
    EXPECT_EQ(InferShapeTest(op, kV2RuntimeParam), ge::GRAPH_FAILED);
}

TEST(SwigluGroupQuantWithDualAxisInferShape, RejectsRankOutsideContract)
{
    ge::op::SwigluGroupQuantWithDualAxis op;
    SetInput(op, "x", {128}, ge::DT_FLOAT16);
    EXPECT_EQ(InferShapeTest(op, kV2RuntimeParam), ge::GRAPH_FAILED);
}

TEST(SwigluGroupQuantWithDualAxisInferShape, RejectsScalarWeight)
{
    ge::op::SwigluGroupQuantWithDualAxis op;
    SetInput(op, "x", {1, 128}, ge::DT_FLOAT16);
    SetInput(op, "weight", {}, ge::DT_FLOAT);
    SetInput(op, "group_index", {1}, ge::DT_INT64);
    EXPECT_EQ(InferShapeTest(op, kV2RuntimeParam), ge::GRAPH_FAILED);
}
TEST(SwigluGroupQuantWithDualAxisInferShape, LargeRowCountDoesNotOverflowCeilDiv)
{
    ge::op::SwigluGroupQuantWithDualAxis op;
    const int64_t rows = std::numeric_limits<int64_t>::max();
    SetInput(op, "x", {rows, 128}, ge::DT_FLOAT16);
    ASSERT_EQ(InferShapeTest(op, kV2RuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(3).GetShape().GetDims(), std::vector<int64_t>({rows / 64 + 1, 64, 2}));
}

TEST(SwigluGroupQuantWithDualAxisInferShape, RejectsGroupedScaleRowOverflow)
{
    ge::op::SwigluGroupQuantWithDualAxis op;
    SetInput(op, "x", {64, 128}, ge::DT_FLOAT16);
    SetInput(op, "weight", {64}, ge::DT_FLOAT);
    SetInput(op, "group_index", {std::numeric_limits<int64_t>::max()}, ge::DT_INT64);
    EXPECT_EQ(InferShapeTest(op, kV2RuntimeParam), ge::GRAPH_FAILED);
}
} // namespace
