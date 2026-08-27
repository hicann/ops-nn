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
#include <initializer_list>
#include <iostream>
#include <limits>

#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_range_context.h"
#include "../../../op_graph/sparse_fill_empty_rows_proto.h"

namespace {
constexpr size_t kInputNum = 4U;
constexpr size_t kOutputNum = 4U;
constexpr int32_t kYIndicesOutputIdx = 0;
constexpr int32_t kYValuesOutputIdx = 1;
constexpr int32_t kEmptyRowOutputIdx = 2;
constexpr int32_t kReverseIndexMapOutputIdx = 3;

class SparseFillEmptyRowsInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SparseFillEmptyRowsInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SparseFillEmptyRowsInfershapeTest TearDown" << std::endl; }
};

struct SparseFillEmptyRowsRangeCase {
    gert::Tensor indicesMinTensor = {
        {{2, 2}, {2, 2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost, ge::DT_INT64, nullptr};
    gert::Tensor indicesMaxTensor = {
        {{5, 2}, {5, 2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost, ge::DT_INT64, nullptr};
    gert::Range<gert::Tensor> indicesTensorRange = {&indicesMinTensor, &indicesMaxTensor};

    gert::Tensor valuesMinTensor = {
        {{2}, {2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost, ge::DT_FLOAT, nullptr};
    gert::Tensor valuesMaxTensor = {
        {{5}, {5}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost, ge::DT_FLOAT, nullptr};
    gert::Range<gert::Tensor> valuesTensorRange = {&valuesMinTensor, &valuesMaxTensor};

    int64_t denseShapeData[gert::Shape::kMaxDimNum + 1U] = {3, 4};
    gert::Tensor denseShapeTensor = {
        {{2}, {2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost, ge::DT_INT64, denseShapeData};
    gert::Range<gert::Tensor> denseShapeTensorRange = {&denseShapeTensor, &denseShapeTensor};

    gert::Tensor defaultValueTensor = {
        {{}, {}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost, ge::DT_FLOAT, nullptr};
    gert::Range<gert::Tensor> defaultValueTensorRange = {&defaultValueTensor, &defaultValueTensor};

    ge::DataType indicesDtype = ge::DT_INT64;
    ge::DataType valuesDtype = ge::DT_FLOAT;
    ge::DataType denseShapeDtype = ge::DT_INT64;
    ge::DataType defaultValueDtype = ge::DT_FLOAT;

    void SetIndicesShape(std::initializer_list<int64_t> minShape, std::initializer_list<int64_t> maxShape)
    {
        indicesMinTensor.MutableOriginShape() = gert::Shape(minShape);
        indicesMinTensor.MutableStorageShape() = gert::Shape(minShape);
        indicesMaxTensor.MutableOriginShape() = gert::Shape(maxShape);
        indicesMaxTensor.MutableStorageShape() = gert::Shape(maxShape);
    }

    void SetDenseShapeElementNum(int64_t elementNum)
    {
        denseShapeTensor.MutableOriginShape().SetDim(0, elementNum);
        denseShapeTensor.MutableStorageShape().SetDim(0, elementNum);
    }

    void SetDenseShapeTensorShape(std::initializer_list<int64_t> shape)
    {
        denseShapeTensor.MutableOriginShape() = gert::Shape(shape);
        denseShapeTensor.MutableStorageShape() = gert::Shape(shape);
    }

    gert::ContextHolder<gert::InferShapeRangeContext> BuildContext()
    {
        indicesMinTensor.SetDataType(indicesDtype);
        indicesMaxTensor.SetDataType(indicesDtype);
        valuesMinTensor.SetDataType(valuesDtype);
        valuesMaxTensor.SetDataType(valuesDtype);
        denseShapeTensor.SetDataType(denseShapeDtype);
        defaultValueTensor.SetDataType(defaultValueDtype);

        gert::InferShapeRangeContextFaker faker;
        faker.SetOpType("SparseFillEmptyRows")
            .IrInputNum(kInputNum)
            .NodeIoNum(kInputNum, kOutputNum)
            .NodeInputTd(0, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, valuesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, denseShapeDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, defaultValueDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, valuesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputTensorsRange(
                {&indicesTensorRange, &valuesTensorRange, &denseShapeTensorRange, &defaultValueTensorRange});
        return std::move(static_cast<gert::OpInferShapeRangeContextBuilder&>(faker)).Build();
    }
};

decltype(gert::OpImplRegistry::GetInstance().GetOpImpl("SparseFillEmptyRows")->infer_shape_range) GetInferShapeRange()
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseFillEmptyRows");
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr) {
        return nullptr;
    }
    EXPECT_NE(opImpl->infer_shape_range, nullptr);
    return opImpl->infer_shape_range;
}
} // namespace

TEST_F(SparseFillEmptyRowsInfershapeTest, InferShapeRangeWithConstDenseShape)
{
    auto inferShapeRangeFunc = GetInferShapeRange();
    ASSERT_NE(inferShapeRangeFunc, nullptr);
    SparseFillEmptyRowsRangeCase rangeCase;
    auto contextHolder = rangeCase.BuildContext();
    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    ASSERT_EQ(inferShapeRangeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(0), 1);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(1), 2);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(0), 12);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(1), 2);
    EXPECT_EQ(context->GetOutputShapeRange(1)->GetMin()->GetDim(0), 1);
    EXPECT_EQ(context->GetOutputShapeRange(1)->GetMax()->GetDim(0), 12);
    EXPECT_EQ(context->GetOutputShapeRange(2)->GetMin()->GetDim(0), 3);
    EXPECT_EQ(context->GetOutputShapeRange(2)->GetMax()->GetDim(0), 3);
    EXPECT_EQ(context->GetOutputShapeRange(3)->GetMin()->GetDim(0), 5);
    EXPECT_EQ(context->GetOutputShapeRange(3)->GetMax()->GetDim(0), 5);
}

TEST_F(SparseFillEmptyRowsInfershapeTest, InferShapeRangeFailedWhenIndicesDtypeInvalid)
{
    auto inferShapeRangeFunc = GetInferShapeRange();
    ASSERT_NE(inferShapeRangeFunc, nullptr);
    SparseFillEmptyRowsRangeCase rangeCase;
    rangeCase.indicesDtype = ge::DT_INT32;
    auto contextHolder = rangeCase.BuildContext();
    EXPECT_EQ(inferShapeRangeFunc(contextHolder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SparseFillEmptyRowsInfershapeTest, InferShapeRangeFailedWhenDenseShapeDtypeInvalid)
{
    auto inferShapeRangeFunc = GetInferShapeRange();
    ASSERT_NE(inferShapeRangeFunc, nullptr);
    SparseFillEmptyRowsRangeCase rangeCase;
    rangeCase.denseShapeDtype = ge::DT_INT32;
    auto contextHolder = rangeCase.BuildContext();
    EXPECT_EQ(inferShapeRangeFunc(contextHolder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SparseFillEmptyRowsInfershapeTest, InferShapeRangeFailedWhenValuesAndDefaultValueDtypeMismatch)
{
    auto inferShapeRangeFunc = GetInferShapeRange();
    ASSERT_NE(inferShapeRangeFunc, nullptr);
    SparseFillEmptyRowsRangeCase rangeCase;
    rangeCase.defaultValueDtype = ge::DT_DOUBLE;
    auto contextHolder = rangeCase.BuildContext();
    EXPECT_EQ(inferShapeRangeFunc(contextHolder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SparseFillEmptyRowsInfershapeTest, InferShapeRangeFailedWhenIndicesAre1D)
{
    auto inferShapeRangeFunc = GetInferShapeRange();
    ASSERT_NE(inferShapeRangeFunc, nullptr);
    SparseFillEmptyRowsRangeCase rangeCase;
    rangeCase.SetIndicesShape({2}, {5});
    auto contextHolder = rangeCase.BuildContext();
    EXPECT_EQ(inferShapeRangeFunc(contextHolder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SparseFillEmptyRowsInfershapeTest, InferShapeRangeFailedWhenIndicesAre3D)
{
    auto inferShapeRangeFunc = GetInferShapeRange();
    ASSERT_NE(inferShapeRangeFunc, nullptr);
    SparseFillEmptyRowsRangeCase rangeCase;
    rangeCase.SetIndicesShape({2, 2, 1}, {5, 2, 1});
    auto contextHolder = rangeCase.BuildContext();
    EXPECT_EQ(inferShapeRangeFunc(contextHolder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SparseFillEmptyRowsInfershapeTest, InferShapeRangeFailedWhenIndicesRankIsUnknown)
{
    auto inferShapeRangeFunc = GetInferShapeRange();
    ASSERT_NE(inferShapeRangeFunc, nullptr);
    SparseFillEmptyRowsRangeCase rangeCase;
    rangeCase.SetIndicesShape({-2}, {-2});
    auto contextHolder = rangeCase.BuildContext();
    EXPECT_EQ(inferShapeRangeFunc(contextHolder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SparseFillEmptyRowsInfershapeTest, InferShapeRangeFailedWhenDenseShapeIsEmpty)
{
    auto inferShapeRangeFunc = GetInferShapeRange();
    ASSERT_NE(inferShapeRangeFunc, nullptr);
    SparseFillEmptyRowsRangeCase rangeCase;
    rangeCase.SetDenseShapeElementNum(0);
    auto contextHolder = rangeCase.BuildContext();
    EXPECT_EQ(inferShapeRangeFunc(contextHolder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SparseFillEmptyRowsInfershapeTest, InferShapeRangeFailedWhenDenseShapeIsNot1D)
{
    auto inferShapeRangeFunc = GetInferShapeRange();
    ASSERT_NE(inferShapeRangeFunc, nullptr);
    SparseFillEmptyRowsRangeCase rangeCase;
    rangeCase.SetDenseShapeTensorShape({1, 2});
    auto contextHolder = rangeCase.BuildContext();
    EXPECT_EQ(inferShapeRangeFunc(contextHolder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SparseFillEmptyRowsInfershapeTest, InferShapeRangeFailedWhenDenseShapeHasNegativeDim)
{
    auto inferShapeRangeFunc = GetInferShapeRange();
    ASSERT_NE(inferShapeRangeFunc, nullptr);
    SparseFillEmptyRowsRangeCase rangeCase;
    rangeCase.denseShapeData[0] = -1;
    auto contextHolder = rangeCase.BuildContext();
    EXPECT_EQ(inferShapeRangeFunc(contextHolder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SparseFillEmptyRowsInfershapeTest, InferShapeRangeFailedWhenDenseShapeExceedsMaxRank)
{
    auto inferShapeRangeFunc = GetInferShapeRange();
    ASSERT_NE(inferShapeRangeFunc, nullptr);
    SparseFillEmptyRowsRangeCase rangeCase;
    rangeCase.SetDenseShapeElementNum(static_cast<int64_t>(gert::Shape::kMaxDimNum + 1U));
    auto contextHolder = rangeCase.BuildContext();
    EXPECT_EQ(inferShapeRangeFunc(contextHolder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SparseFillEmptyRowsInfershapeTest, InferShapeRangeFailedWhenDenseShapeSizeOverflows)
{
    auto inferShapeRangeFunc = GetInferShapeRange();
    ASSERT_NE(inferShapeRangeFunc, nullptr);
    SparseFillEmptyRowsRangeCase rangeCase;
    rangeCase.denseShapeData[0] = std::numeric_limits<int64_t>::max();
    rangeCase.denseShapeData[1] = 2;
    auto contextHolder = rangeCase.BuildContext();
    EXPECT_EQ(inferShapeRangeFunc(contextHolder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SparseFillEmptyRowsInfershapeTest, InferShapeRangeSupportsZeroDenseDimension)
{
    auto inferShapeRangeFunc = GetInferShapeRange();
    ASSERT_NE(inferShapeRangeFunc, nullptr);
    SparseFillEmptyRowsRangeCase rangeCase;
    rangeCase.SetIndicesShape({0, 2}, {0, 2});
    rangeCase.denseShapeData[0] = 0;
    auto contextHolder = rangeCase.BuildContext();
    auto context = contextHolder.GetContext();

    ASSERT_EQ(inferShapeRangeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShapeRange(kYIndicesOutputIdx)->GetMin()->GetDim(0), 0);
    EXPECT_EQ(context->GetOutputShapeRange(kYIndicesOutputIdx)->GetMax()->GetDim(0), 0);
    EXPECT_EQ(context->GetOutputShapeRange(kYValuesOutputIdx)->GetMin()->GetDim(0), 0);
    EXPECT_EQ(context->GetOutputShapeRange(kYValuesOutputIdx)->GetMax()->GetDim(0), 0);
    EXPECT_EQ(context->GetOutputShapeRange(kEmptyRowOutputIdx)->GetMin()->GetDim(0), 0);
    EXPECT_EQ(context->GetOutputShapeRange(kEmptyRowOutputIdx)->GetMax()->GetDim(0), 0);
}
