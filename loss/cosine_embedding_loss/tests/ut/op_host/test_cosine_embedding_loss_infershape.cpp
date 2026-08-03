/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_cosine_embedding_loss_infershape.cpp
 * \brief CosineEmbeddingLoss inference UT.
 */

#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

namespace {
void AppendShape(gert::StorageShape& shape, const std::vector<int64_t>& dims)
{
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
}

void RunInferShapeCase(const std::vector<int64_t>& x1Dims, const std::vector<int64_t>& x2Dims,
                       const std::vector<int64_t>& targetDims, const char* reduction, ge::graphStatus expectedStatus,
                       const std::vector<int64_t>& expectedOutputDims = {})
{
    gert::StorageShape x1Shape;
    gert::StorageShape x2Shape;
    gert::StorageShape targetShape;
    gert::StorageShape outputShape;
    AppendShape(x1Shape, x1Dims);
    AppendShape(x2Shape, x2Dims);
    AppendShape(targetShape, targetDims);

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"margin", Ops::NN::AnyValue::CreateFrom<float>(0.0f)}};
    if (reduction != nullptr) {
        attrs.emplace_back("reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction));
    }

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("CosineEmbeddingLoss")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x1Shape, &x2Shape, &targetShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(attrs)
                      .Build();

    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("CosineEmbeddingLoss");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(opImpl->infer_shape(context), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        ASSERT_NE(context->GetOutputShape(0), nullptr);
        EXPECT_EQ(ut_util::ToVector(*context->GetOutputShape(0)), expectedOutputDims);
    }
}
} // namespace

class CosineEmbeddingLossInferShape : public testing::Test {};

TEST_F(CosineEmbeddingLossInferShape, none_broadcast)
{
    RunInferShapeCase({1, 3, 4}, {2, 3, 1}, {4}, "none", ge::GRAPH_SUCCESS, {2, 4});
}

TEST_F(CosineEmbeddingLossInferShape, default_reduction)
{
    RunInferShapeCase({2, 3, 4}, {2, 3, 4}, {2, 4}, nullptr, ge::GRAPH_SUCCESS, {1});
}

TEST_F(CosineEmbeddingLossInferShape, none_unknown_dimension)
{
    RunInferShapeCase({1, 3, -1}, {2, 3, 1}, {4}, "none", ge::GRAPH_SUCCESS, {2, -1});
}

TEST_F(CosineEmbeddingLossInferShape, none_unknown_rank)
{
    RunInferShapeCase({-2}, {2, 3, 4}, {2, 4}, "none", ge::GRAPH_SUCCESS, {-2});
}

TEST_F(CosineEmbeddingLossInferShape, invalid_x_broadcast)
{
    RunInferShapeCase({2, 3, 4}, {2, 4, 4}, {2, 4}, "none", ge::GRAPH_FAILED);
}

TEST_F(CosineEmbeddingLossInferShape, invalid_reduction)
{
    RunInferShapeCase({2, 3}, {2, 3}, {2}, "bogus", ge::GRAPH_FAILED);
}

TEST_F(CosineEmbeddingLossInferShape, invalid_zero_dimension)
{
    RunInferShapeCase({2, 0, 4}, {2, 1, 4}, {2, 4}, "none", ge::GRAPH_FAILED);
}

TEST_F(CosineEmbeddingLossInferShape, invalid_dimension_less_than_negative_one)
{
    RunInferShapeCase({2, -3, 4}, {2, 1, 4}, {2, 4}, "none", ge::GRAPH_FAILED);
}

TEST_F(CosineEmbeddingLossInferShape, invalid_rank_zero_target)
{
    RunInferShapeCase({2, 3}, {2, 3}, {}, "none", ge::GRAPH_FAILED);
}

TEST_F(CosineEmbeddingLossInferShape, output_dtype_is_float)
{
    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("CosineEmbeddingLoss");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    ge::DataType x1Dtype = ge::DT_FLOAT16;
    ge::DataType x2Dtype = ge::DT_FLOAT16;
    ge::DataType targetDtype = ge::DT_INT32;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, targetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&x1Dtype, &x2Dtype, &targetDtype})
                      .OutputDataTypes({&outputDtype})
                      .Build();

    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}
