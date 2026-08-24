/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_graph/tensor_scatter_update_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ge;

namespace {
constexpr int64_t kUnknownDim = -1;
constexpr int64_t kUnknownRank = -2;
constexpr size_t kInputNum = 3;
constexpr size_t kOutputNum = 1;
constexpr int32_t kXInputIdx = 0;
constexpr int32_t kIndicesInputIdx = 1;
constexpr int32_t kUpdatesInputIdx = 2;
constexpr int32_t kYOutputIdx = 0;

// The source proto derives y from x alone, so every case only varies x while keeping indices and
// updates well formed.
gert::KernelRunContextHolder BuildContext(gert::StorageShape& xShape, gert::StorageShape& indicesShape,
                                          gert::StorageShape& updatesShape, gert::Shape& yShape)
{
    return gert::InferShapeContextFaker()
        .NodeIoNum(kInputNum, kOutputNum)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(kXInputIdx, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(kIndicesInputIdx, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(kUpdatesInputIdx, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(kYOutputIdx, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputShapes({&xShape, &indicesShape, &updatesShape})
        .OutputShapes({&yShape})
        .Build();
}
} // namespace

class TensorScatterUpdateInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TensorScatterUpdateInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TensorScatterUpdateInfershapeTest TearDown" << std::endl; }
};

TEST_F(TensorScatterUpdateInfershapeTest, InferShapeFollowsXSuccess)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TensorScatterUpdate")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{4, 5, 6}, {4, 5, 6}};
    gert::StorageShape indicesShape = {{3, 2}, {3, 2}};
    gert::StorageShape updatesShape = {{3, 6}, {3, 6}};
    gert::Shape yShape;

    auto holder = BuildContext(xShape, indicesShape, updatesShape, yShape);
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(CreateShape({4, 5, 6})));
}

// x with an unknown rank (-2) must be passed through, not rejected: dynamic graph compilation
// hands this shape in before the real rank is known.
TEST_F(TensorScatterUpdateInfershapeTest, InferShapeXUnknownRankSuccess)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TensorScatterUpdate")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{kUnknownRank}, {kUnknownRank}};
    gert::StorageShape indicesShape = {{3, 2}, {3, 2}};
    gert::StorageShape updatesShape = {{3, 6}, {3, 6}};
    gert::Shape yShape;

    auto holder = BuildContext(xShape, indicesShape, updatesShape, yShape);
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(CreateShape({kUnknownRank})));
}

// Partially dynamic x: the unknown dims stay unknown, the known ones are carried over.
TEST_F(TensorScatterUpdateInfershapeTest, InferShapeXUnknownDimSuccess)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TensorScatterUpdate")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{kUnknownDim, 5, kUnknownDim}, {kUnknownDim, 5, kUnknownDim}};
    gert::StorageShape indicesShape = {{kUnknownDim, 2}, {kUnknownDim, 2}};
    gert::StorageShape updatesShape = {{kUnknownDim, kUnknownDim}, {kUnknownDim, kUnknownDim}};
    gert::Shape yShape;

    auto holder = BuildContext(xShape, indicesShape, updatesShape, yShape);
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(CreateShape({kUnknownDim, 5, kUnknownDim})));
}

// The source proto has no rank guard on x, so a rank 0 x yields a rank 0 y and still succeeds.
// Kept as-is on purpose: the migration target is behavioural equivalence, not tightening the source.
TEST_F(TensorScatterUpdateInfershapeTest, InferShapeScalarXSuccess)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TensorScatterUpdate")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{}, {}};
    gert::StorageShape indicesShape = {{1}, {1}};
    gert::StorageShape updatesShape = {{}, {}};
    gert::Shape yShape;

    auto holder = BuildContext(xShape, indicesShape, updatesShape, yShape);
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(output->GetDimNum(), 0U);
}

TEST_F(TensorScatterUpdateInfershapeTest, InferDataTypeFollowsX)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("TensorScatterUpdate"), nullptr);
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TensorScatterUpdate")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    ge::DataType xDtype = ge::DT_DOUBLE;
    ge::DataType indicesDtype = ge::DT_INT64;
    ge::DataType updatesDtype = ge::DT_DOUBLE;
    ge::DataType yDtype = ge::DT_FLOAT;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(kInputNum, kOutputNum)
                             .NodeInputTd(kXInputIdx, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(kIndicesInputIdx, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(kUpdatesInputIdx, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(kYOutputIdx, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&xDtype, &indicesDtype, &updatesDtype})
                             .OutputDataTypes({&yDtype})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_DOUBLE);
}
