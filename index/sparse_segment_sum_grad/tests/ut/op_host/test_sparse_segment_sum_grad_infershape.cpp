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
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_graph/sparse_segment_sum_grad_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ge;

namespace {
constexpr int64_t kUnknownDim = -1;
constexpr int64_t kUnknownRank = -2;

std::unique_ptr<uint8_t[]> BuildOutputDim0Tensor(const std::vector<int64_t>& shapeDims, int32_t value,
                                                 gert::Tensor*& tensor)
{
    size_t totalSize = 0;
    auto holder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, totalSize);
    tensor = reinterpret_cast<gert::Tensor*>(holder.get());
    for (const int64_t dim : shapeDims) {
        tensor->MutableStorageShape().AppendDim(dim);
        tensor->MutableOriginShape().AppendDim(dim);
    }
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    *(tensor->GetData<int32_t>()) = value;
    return holder;
}

void ExpectShapeEq(const std::vector<int64_t>& expected, const gert::Shape& actual)
{
    ASSERT_EQ(actual.GetDimNum(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual.GetDim(i), expected[i]);
    }
}
} // namespace

class SparseSegmentSumGradInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SparseSegmentSumGradInfershapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SparseSegmentSumGradInfershapeTest TearDown" << std::endl; }
};

TEST_F(SparseSegmentSumGradInfershapeTest, InferShape2dSuccess)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSumGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradShape = {{2, 3}, {2, 3}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape segmentIdsShape = {{3}, {3}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, 3, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&gradShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ExpectShapeEq({kUnknownDim, 3}, *output);
}

TEST_F(SparseSegmentSumGradInfershapeTest, InferShape3dSuccess)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSumGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape segmentIdsShape = {{3}, {3}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, 5, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&gradShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ExpectShapeEq({kUnknownDim, 3, 4}, *output);
}

TEST_F(SparseSegmentSumGradInfershapeTest, InferShapeGradUnknownRank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSumGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradShape = {{kUnknownRank}, {kUnknownRank}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape segmentIdsShape = {{3}, {3}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, 3, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&gradShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ExpectShapeEq({kUnknownRank}, *output);
}

TEST_F(SparseSegmentSumGradInfershapeTest, InferShapeGradUnknownDim)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSumGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradShape = {{kUnknownDim, 3, 4}, {kUnknownDim, 3, 4}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape segmentIdsShape = {{3}, {3}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, 7, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&gradShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ExpectShapeEq({kUnknownDim, 3, 4}, *output);
}

TEST_F(SparseSegmentSumGradInfershapeTest, InferShapeIndicesUnknownRank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSumGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradShape = {{2, 3}, {2, 3}};
    gert::StorageShape indicesShape = {{kUnknownRank}, {kUnknownRank}};
    gert::StorageShape segmentIdsShape = {{5}, {5}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, 3, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&gradShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ExpectShapeEq({kUnknownDim, 3}, *output);
}

TEST_F(SparseSegmentSumGradInfershapeTest, InferShapeIndicesUnknownDim)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSumGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradShape = {{2, 3}, {2, 3}};
    gert::StorageShape indicesShape = {{kUnknownDim}, {kUnknownDim}};
    gert::StorageShape segmentIdsShape = {{5}, {5}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, 3, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&gradShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ExpectShapeEq({kUnknownDim, 3}, *output);
}

TEST_F(SparseSegmentSumGradInfershapeTest, InferShapeIndicesDimMismatchFailed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSumGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradShape = {{2, 3}, {2, 3}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape segmentIdsShape = {{4}, {4}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, 3, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&gradShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(SparseSegmentSumGradInfershapeTest, InferShapeIndicesNot1dFailed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSumGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradShape = {{2, 3}, {2, 3}};
    gert::StorageShape indicesShape = {{3, 1}, {3, 1}};
    gert::StorageShape segmentIdsShape = {{3}, {3}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, 3, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&gradShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
