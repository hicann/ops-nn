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
#include "register/op_impl_registry.h"
#include "infershape_test_util.h"
#include "infer_shaperange_context_faker.h"
#include "op_common/op_host/util/shape_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/centralization_proto.h"

TEST(CentralizationInferShape, CopyInputShape)
{
    ge::op::Centralization op;
    auto input = create_desc_with_ori({2, 3, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND);
    op.UpdateInputDesc("x", input);
    op.SetAttr("axes", std::vector<int64_t>{1});
    ASSERT_EQ(InferShapeTest(op, Runtime2TestParam{{"axes"}}), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDescByName("y").GetShape().GetDims(), std::vector<int64_t>({2, 3, 4}));
}

TEST(CentralizationInferShape, CopyInputShapeAndTypeFp16)
{
    ge::op::Centralization op;
    auto input = create_desc_with_ori({2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND);
    op.UpdateInputDesc("x", input);
    op.SetAttr("axes", std::vector<int64_t>{1});
    ASSERT_EQ(InferShapeTest(op, Runtime2TestParam{{"axes"}}), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDescByName("y").GetShape().GetDims(), std::vector<int64_t>({2, 3, 4}));
    ASSERT_EQ(InferDataTypeTest(op, Runtime2TestParam{{"axes"}}), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDescByName("y").GetDataType(), ge::DT_FLOAT16);
}

TEST(CentralizationInferShape, CopyDynamicDimensionAndShapeRange)
{
    ge::op::Centralization op;
    const std::vector<std::pair<int64_t, int64_t>> shapeRange = {{2, 8}, {100, 200}, {4, 16}};
    auto input = create_desc_shape_range({-1, 100, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 100, 4}, ge::FORMAT_ND,
                                         shapeRange);
    op.UpdateInputDesc("x", input);
    op.SetAttr("axes", std::vector<int64_t>{1});

    ASSERT_EQ(InferShapeTest(op, Runtime2TestParam{{"axes"}}), ge::GRAPH_SUCCESS);
    const auto output = op.GetOutputDescByName("y");
    EXPECT_EQ(output.GetShape().GetDims(), std::vector<int64_t>({-1, 100, 4}));
    ASSERT_EQ(InferDataTypeTest(op, Runtime2TestParam{{"axes"}}), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDescByName("y").GetDataType(), ge::DT_FLOAT);
}

TEST(CentralizationInferShape, CopyUnknownRank)
{
    ge::op::Centralization op;
    auto input = create_desc_with_ori({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {-2}, ge::FORMAT_ND);
    op.UpdateInputDesc("x", input);
    op.SetAttr("axes", std::vector<int64_t>{-1});

    ASSERT_EQ(InferShapeTest(op, Runtime2TestParam{{"axes"}}), ge::GRAPH_SUCCESS);
    ASSERT_EQ(InferDataTypeTest(op, Runtime2TestParam{{"axes"}}), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDescByName("y").GetShape().GetDims(), std::vector<int64_t>({-2}));
    EXPECT_EQ(op.GetOutputDescByName("y").GetDataType(), ge::DT_FLOAT16);
}

TEST(CentralizationInferShape, RejectRankZero)
{
    ge::op::Centralization op;
    auto input = create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND);
    op.UpdateInputDesc("x", input);
    op.SetAttr("axes", std::vector<int64_t>{-1});

    EXPECT_EQ(InferShapeTest(op, Runtime2TestParam{{"axes"}}), ge::GRAPH_FAILED);
}

TEST(CentralizationInferShape, RejectRankGreaterThanEight)
{
    ge::op::Centralization op;
    auto input = create_desc_with_ori({1, 1, 1, 1, 1, 1, 1, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_ND,
                                      {1, 1, 1, 1, 1, 1, 1, 1, 1}, ge::FORMAT_ND);
    op.UpdateInputDesc("x", input);
    op.SetAttr("axes", std::vector<int64_t>{-1});

    EXPECT_EQ(InferShapeTest(op, Runtime2TestParam{{"axes"}}), ge::GRAPH_FAILED);
}

TEST(CentralizationInferShape, RejectInvalidAxes)
{
    ge::op::Centralization op;
    auto input = create_desc_with_ori({2, 3, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND);
    op.UpdateInputDesc("x", input);
    op.SetAttr("axes", std::vector<int64_t>{3});

    EXPECT_EQ(InferShapeTest(op, Runtime2TestParam{{"axes"}}), ge::GRAPH_FAILED);
}

TEST(CentralizationInferShape, RejectDuplicateAxes)
{
    ge::op::Centralization op;
    auto input = create_desc_with_ori({2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND);
    op.UpdateInputDesc("x", input);
    op.SetAttr("axes", std::vector<int64_t>{1, -2});

    EXPECT_EQ(InferShapeTest(op, Runtime2TestParam{{"axes"}}), ge::GRAPH_FAILED);
}

TEST(CentralizationInferShape, RejectUnsupportedDtype)
{
    ge::op::Centralization op;
    auto input = create_desc_with_ori({2, 3, 4}, ge::DT_BF16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND);
    op.UpdateInputDesc("x", input);
    op.SetAttr("axes", std::vector<int64_t>{1});

    EXPECT_EQ(InferDataTypeTest(op, Runtime2TestParam{{"axes"}}), ge::GRAPH_FAILED);
}

TEST(CentralizationInferShape, CopyShapeRange)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Centralization")->infer_shape_range;
    ASSERT_NE(inferShapeRangeFunc, nullptr);

    gert::Shape inputMin{2, 100, 4};
    gert::Shape inputMax{8, 200, 16};
    gert::Range<gert::Shape> inputRange(&inputMin, &inputMax);
    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapeRanges({&inputRange})
                      .OutputShapeRanges({nullptr})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeRangeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeRangeFunc(context), ge::GRAPH_SUCCESS);
    const auto* outputRange = context->GetOutputShapeRange(0);
    ASSERT_NE(outputRange, nullptr);
    ASSERT_NE(outputRange->GetMin(), nullptr);
    ASSERT_NE(outputRange->GetMax(), nullptr);
    ASSERT_EQ(outputRange->GetMin()->GetDimNum(), inputMin.GetDimNum());
    ASSERT_EQ(outputRange->GetMax()->GetDimNum(), inputMax.GetDimNum());
    for (size_t dim = 0; dim < inputMin.GetDimNum(); ++dim) {
        EXPECT_EQ(outputRange->GetMin()->GetDim(dim), inputMin.GetDim(dim));
        EXPECT_EQ(outputRange->GetMax()->GetDim(dim), inputMax.GetDim(dim));
    }
}

TEST(CentralizationInferShape, InferDataTypeRegistered)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Centralization"), nullptr);
    EXPECT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Centralization")->infer_datatype, nullptr);
}
