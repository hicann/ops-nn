/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_graph/sorted_sparse_segment_mean_grad_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ge;
using namespace op;

class SortedSparseSegmentMeanGradProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SortedSparseSegmentMeanGradProtoTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SortedSparseSegmentMeanGradProtoTest TearDown" << std::endl; }
};

TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_shape_test_1d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_shape;

    gert::StorageShape xShape = {{8}, {8}};
    gert::StorageShape sortedIndicesShape = {{8}, {8}};
    gert::StorageShape preLocationIndicesShape = {{8}, {8}};
    gert::StorageShape segmentIdsShape = {{8}, {8}};
    gert::Shape yShape;

    vector<int32_t> outputDim0Data{4};
    size_t totalSize = 0;
    auto outputDim0TensorHolder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, totalSize);
    auto outputDim0Tensor = reinterpret_cast<gert::Tensor*>(outputDim0TensorHolder.get());
    outputDim0Tensor->MutableStorageShape().AppendDim(1);
    outputDim0Tensor->MutableOriginShape().AppendDim(1);
    outputDim0Tensor->SetOriginFormat(ge::FORMAT_ND);
    outputDim0Tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(outputDim0Tensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), outputDim0Data.data(),
                   outputDim0Data.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes(
                          {&xShape, &sortedIndicesShape, &preLocationIndicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedShape = CreateShape({4});
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedShape));
}

TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_shape_test_2d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_shape;

    gert::StorageShape xShape = {{8, 16}, {8, 16}};
    gert::StorageShape sortedIndicesShape = {{8}, {8}};
    gert::StorageShape preLocationIndicesShape = {{8}, {8}};
    gert::StorageShape segmentIdsShape = {{8}, {8}};
    gert::Shape yShape;

    vector<int32_t> outputDim0Data{4};
    size_t totalSize = 0;
    auto outputDim0TensorHolder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, totalSize);
    auto outputDim0Tensor = reinterpret_cast<gert::Tensor*>(outputDim0TensorHolder.get());
    outputDim0Tensor->MutableStorageShape().AppendDim(1);
    outputDim0Tensor->MutableOriginShape().AppendDim(1);
    outputDim0Tensor->SetOriginFormat(ge::FORMAT_ND);
    outputDim0Tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(outputDim0Tensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), outputDim0Data.data(),
                   outputDim0Data.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes(
                          {&xShape, &sortedIndicesShape, &preLocationIndicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedShape = CreateShape({4, 16});
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedShape));
}

TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_shape_test_3d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_shape;

    gert::StorageShape xShape = {{8, 16, 32}, {8, 16, 32}};
    gert::StorageShape sortedIndicesShape = {{8}, {8}};
    gert::StorageShape preLocationIndicesShape = {{8}, {8}};
    gert::StorageShape segmentIdsShape = {{8}, {8}};
    gert::Shape yShape;

    vector<int32_t> outputDim0Data{4};
    size_t totalSize = 0;
    auto outputDim0TensorHolder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, totalSize);
    auto outputDim0Tensor = reinterpret_cast<gert::Tensor*>(outputDim0TensorHolder.get());
    outputDim0Tensor->MutableStorageShape().AppendDim(1);
    outputDim0Tensor->MutableOriginShape().AppendDim(1);
    outputDim0Tensor->SetOriginFormat(ge::FORMAT_ND);
    outputDim0Tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(outputDim0Tensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), outputDim0Data.data(),
                   outputDim0Data.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes(
                          {&xShape, &sortedIndicesShape, &preLocationIndicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedShape = CreateShape({4, 16, 32});
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedShape));
}

TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_dtype_test_float)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_datatype;

    if (dataTypeFunc != nullptr) {
        ge::DataType inputRef1 = ge::DT_FLOAT;
        ge::DataType inputRef2 = ge::DT_INT32;
        ge::DataType inputRef3 = ge::DT_INT32;
        ge::DataType inputRef4 = ge::DT_INT32;
        ge::DataType inputRef5 = ge::DT_INT32;
        ge::DataType outputRef = ge::DT_FLOAT;
        auto contextHolder = gert::InferDataTypeContextFaker()
                                 .NodeIoNum(5, 1)
                                 .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .InputDataTypes({&inputRef1, &inputRef2, &inputRef3, &inputRef4, &inputRef5})
                                 .OutputDataTypes({&outputRef})
                                 .Build();
        auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetOutputDataType(0), outputRef);
    }
}

TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_dtype_test_float16)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_datatype;

    if (dataTypeFunc != nullptr) {
        ge::DataType inputRef1 = ge::DT_FLOAT16;
        ge::DataType inputRef2 = ge::DT_INT32;
        ge::DataType inputRef3 = ge::DT_INT32;
        ge::DataType inputRef4 = ge::DT_INT32;
        ge::DataType inputRef5 = ge::DT_INT32;
        ge::DataType outputRef = ge::DT_FLOAT16;
        auto contextHolder = gert::InferDataTypeContextFaker()
                                 .NodeIoNum(5, 1)
                                 .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .InputDataTypes({&inputRef1, &inputRef2, &inputRef3, &inputRef4, &inputRef5})
                                 .OutputDataTypes({&outputRef})
                                 .Build();
        auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetOutputDataType(0), outputRef);
    }
}

TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_dtype_test_bf16)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_datatype;

    if (dataTypeFunc != nullptr) {
        ge::DataType inputRef1 = ge::DT_BF16;
        ge::DataType inputRef2 = ge::DT_INT32;
        ge::DataType inputRef3 = ge::DT_INT32;
        ge::DataType inputRef4 = ge::DT_INT32;
        ge::DataType inputRef5 = ge::DT_INT32;
        ge::DataType outputRef = ge::DT_BF16;
        auto contextHolder = gert::InferDataTypeContextFaker()
                                 .NodeIoNum(5, 1)
                                 .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .InputDataTypes({&inputRef1, &inputRef2, &inputRef3, &inputRef4, &inputRef5})
                                 .OutputDataTypes({&outputRef})
                                 .Build();
        auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetOutputDataType(0), outputRef);
    }
}

// Failure test: wrong number of inputs (4 instead of 5) triggers CheckInputAndOutputNum failure
TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_shape_test_wrong_input_num)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_shape;

    gert::StorageShape xShape = {{8, 16}, {8, 16}};
    gert::StorageShape sortedIndicesShape = {{8}, {8}};
    gert::StorageShape preLocationIndicesShape = {{8}, {8}};
    gert::StorageShape segmentIdsShape = {{8}, {8}};
    gert::Shape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes({&xShape, &sortedIndicesShape, &preLocationIndicesShape, &segmentIdsShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// Failure test: wrong number of outputs (2 instead of 1) triggers CheckInputAndOutputNum failure
TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_shape_test_wrong_output_num)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_shape;

    gert::StorageShape xShape = {{8, 16}, {8, 16}};
    gert::StorageShape sortedIndicesShape = {{8}, {8}};
    gert::StorageShape preLocationIndicesShape = {{8}, {8}};
    gert::StorageShape segmentIdsShape = {{8}, {8}};
    gert::StorageShape outputDim0Shape = {{1}, {1}};
    gert::Shape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes(
                          {&xShape, &sortedIndicesShape, &preLocationIndicesShape, &segmentIdsShape, &outputDim0Shape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// Failure test: x rank 0 (scalar) triggers WithRankAtLeast failure in SortedSparseSegmentMeanGradCheck
TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_shape_test_x_rank0)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_shape;

    gert::StorageShape xShape = {{}, {}}; // Scalar (0-dim)
    gert::StorageShape sortedIndicesShape = {{8}, {8}};
    gert::StorageShape preLocationIndicesShape = {{8}, {8}};
    gert::StorageShape segmentIdsShape = {{8}, {8}};
    gert::Shape yShape;

    vector<int32_t> outputDim0Data{4};
    size_t totalSize = 0;
    auto outputDim0TensorHolder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, totalSize);
    auto outputDim0Tensor = reinterpret_cast<gert::Tensor*>(outputDim0TensorHolder.get());
    outputDim0Tensor->MutableStorageShape().AppendDim(1);
    outputDim0Tensor->MutableOriginShape().AppendDim(1);
    outputDim0Tensor->SetOriginFormat(ge::FORMAT_ND);
    outputDim0Tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(outputDim0Tensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), outputDim0Data.data(),
                   outputDim0Data.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes(
                          {&xShape, &sortedIndicesShape, &preLocationIndicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// Failure test: indices not 1D triggers WithRank failure in SortedSparseSegmentMeanGradCheck
TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_shape_test_indices_not_1d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_shape;

    gert::StorageShape xShape = {{8, 16}, {8, 16}};
    gert::StorageShape sortedIndicesShape = {{8, 4}, {8, 4}}; // 2D instead of 1D
    gert::StorageShape preLocationIndicesShape = {{8}, {8}};
    gert::StorageShape segmentIdsShape = {{8}, {8}};
    gert::Shape yShape;

    vector<int32_t> outputDim0Data{4};
    size_t totalSize = 0;
    auto outputDim0TensorHolder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, totalSize);
    auto outputDim0Tensor = reinterpret_cast<gert::Tensor*>(outputDim0TensorHolder.get());
    outputDim0Tensor->MutableStorageShape().AppendDim(1);
    outputDim0Tensor->MutableOriginShape().AppendDim(1);
    outputDim0Tensor->SetOriginFormat(ge::FORMAT_ND);
    outputDim0Tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(outputDim0Tensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), outputDim0Data.data(),
                   outputDim0Data.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes(
                          {&xShape, &sortedIndicesShape, &preLocationIndicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// Failure test: location not 1D triggers WithRank failure in SortedSparseSegmentMeanGradCheck
TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_shape_test_location_not_1d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_shape;

    gert::StorageShape xShape = {{8, 16}, {8, 16}};
    gert::StorageShape sortedIndicesShape = {{8}, {8}};
    gert::StorageShape preLocationIndicesShape = {{8, 4}, {8, 4}}; // 2D instead of 1D
    gert::StorageShape segmentIdsShape = {{8}, {8}};
    gert::Shape yShape;

    vector<int32_t> outputDim0Data{4};
    size_t totalSize = 0;
    auto outputDim0TensorHolder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, totalSize);
    auto outputDim0Tensor = reinterpret_cast<gert::Tensor*>(outputDim0TensorHolder.get());
    outputDim0Tensor->MutableStorageShape().AppendDim(1);
    outputDim0Tensor->MutableOriginShape().AppendDim(1);
    outputDim0Tensor->SetOriginFormat(ge::FORMAT_ND);
    outputDim0Tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(outputDim0Tensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), outputDim0Data.data(),
                   outputDim0Data.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes(
                          {&xShape, &sortedIndicesShape, &preLocationIndicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// Success test: x unknown rank (-2) → output should be unknown rank
TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_shape_test_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_shape;

    // Create unknown rank shape: dim_num=1, dim[0]=-2 (matches IsUnknownRank check)
    gert::StorageShape xUnknownRankShape;
    xUnknownRankShape.MutableOriginShape().SetDimNum(0);
    xUnknownRankShape.MutableOriginShape().AppendDim(-2);
    xUnknownRankShape.MutableStorageShape().SetDimNum(0);
    xUnknownRankShape.MutableStorageShape().AppendDim(-2);

    gert::StorageShape sortedIndicesShape = {{8}, {8}};
    gert::StorageShape preLocationIndicesShape = {{8}, {8}};
    gert::StorageShape segmentIdsShape = {{8}, {8}};
    gert::StorageShape outputDim0Shape = {{1}, {1}};
    gert::Shape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes({&xUnknownRankShape, &sortedIndicesShape, &preLocationIndicesShape, &segmentIdsShape,
                                    &outputDim0Shape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    // SetUnknownRank sets dim_num=1, dim[0]=-2
    ASSERT_EQ(output->GetDimNum(), 1);
    ASSERT_EQ(output->GetDim(0), -2);
}

// Success test: x unknown shape (-1 dims) → output dims should all be -1
TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_shape_test_unknown_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_shape;

    gert::StorageShape xShape = {{-1, 16}, {-1, 16}}; // Unknown shape (dim[0] = -1)
    gert::StorageShape sortedIndicesShape = {{8}, {8}};
    gert::StorageShape preLocationIndicesShape = {{8}, {8}};
    gert::StorageShape segmentIdsShape = {{8}, {8}};
    gert::StorageShape outputDim0Shape = {{1}, {1}};
    gert::Shape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes(
                          {&xShape, &sortedIndicesShape, &preLocationIndicesShape, &segmentIdsShape, &outputDim0Shape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(output->GetDimNum(), 2);
    ASSERT_EQ(output->GetDim(0), -1);
    ASSERT_EQ(output->GetDim(1), -1);
}

// Success test: cannot get output_dim0 (can_get_output_dim0 = false) → output dims all -1
TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_shape_test_cannot_get_output_dim0)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_shape;

    // Use regular known x_shape, but no output_dim0 tensor data → can_get_output_dim0 = false
    gert::StorageShape xShape = {{8, 16}, {8, 16}};
    gert::StorageShape sortedIndicesShape = {{8}, {8}};
    gert::StorageShape preLocationIndicesShape = {{8}, {8}};
    gert::StorageShape segmentIdsShape = {{8}, {8}};
    gert::StorageShape outputDim0Shape = {{1}, {1}}; // Plain StorageShape, no tensor data
    gert::Shape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes(
                          {&xShape, &sortedIndicesShape, &preLocationIndicesShape, &segmentIdsShape, &outputDim0Shape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(output->GetDimNum(), 2);
    ASSERT_EQ(output->GetDim(0), -1);
    ASSERT_EQ(output->GetDim(1), -1);
}

// Success test: indices with unknown rank → covers SetAllUnknownDim in WithRank
TEST_F(SortedSparseSegmentMeanGradProtoTest, infer_shape_test_indices_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SortedSparseSegmentMeanGrad")->infer_shape;

    gert::StorageShape xShape = {{8, 16}, {8, 16}};
    // indices unknown rank: dim_num=1, dim[0]=-2 → triggers SetAllUnknownDim in WithRank
    gert::StorageShape sortedIndicesUnknownRankShape;
    sortedIndicesUnknownRankShape.MutableOriginShape().SetDimNum(0);
    sortedIndicesUnknownRankShape.MutableOriginShape().AppendDim(-2);
    sortedIndicesUnknownRankShape.MutableStorageShape().SetDimNum(0);
    sortedIndicesUnknownRankShape.MutableStorageShape().AppendDim(-2);
    gert::StorageShape preLocationIndicesShape = {{8}, {8}};
    gert::StorageShape segmentIdsShape = {{8}, {8}};
    gert::Shape yShape;

    vector<int32_t> outputDim0Data{4};
    size_t totalSize = 0;
    auto outputDim0TensorHolder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, totalSize);
    auto outputDim0Tensor = reinterpret_cast<gert::Tensor*>(outputDim0TensorHolder.get());
    outputDim0Tensor->MutableStorageShape().AppendDim(1);
    outputDim0Tensor->MutableOriginShape().AppendDim(1);
    outputDim0Tensor->SetOriginFormat(ge::FORMAT_ND);
    outputDim0Tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(outputDim0Tensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), outputDim0Data.data(),
                   outputDim0Data.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes({&xShape, &sortedIndicesUnknownRankShape, &preLocationIndicesShape, &segmentIdsShape,
                                    outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();

    // WithRank for unknown rank indices returns SUCCESS via SetAllUnknownDim
    // The check passes, and SubShape/Concatenate proceed to compute output
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}