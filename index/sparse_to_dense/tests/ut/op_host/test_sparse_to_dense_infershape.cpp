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
#include <iostream>
#include <memory>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"
#include "util/shape_util.h"
#include "../../../op_graph/sparse_to_dense_proto.h"

namespace {
constexpr size_t kInputNum = 4;
constexpr size_t kOutputNum = 1;
constexpr size_t kOutputIndex = 0;

class SparseToDenseInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SparseToDenseInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SparseToDenseInfershapeTest TearDown" << std::endl; }
};

template <typename T>
gert::Tensor* CreateConstTensor(const std::vector<T>& data, ge::DataType dataType,
                                std::unique_ptr<uint8_t[]>& tensorHolder)
{
    size_t totalSize = 0;
    tensorHolder = gert::Tensor::CreateFollowing(static_cast<int64_t>(data.size()), dataType, totalSize);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensorHolder.get());
    tensor->MutableStorageShape().AppendDim(static_cast<int64_t>(data.size()));
    tensor->MutableOriginShape().AppendDim(static_cast<int64_t>(data.size()));
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(tensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), data.data(), data.size() * sizeof(T));
    return tensor;
}
} // namespace

TEST_F(SparseToDenseInfershapeTest, InferShapeWithConstOutputShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseToDense")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape indicesShape = {2, 2};
    gert::Shape valuesShape = {2};
    gert::Shape defaultValueShape = {};
    gert::Shape yShape = {};
    std::unique_ptr<uint8_t[]> outputShapeTensorHolder;
    auto outputShapeTensor = CreateConstTensor<int64_t>({2, 3}, ge::DT_INT64, outputShapeTensorHolder);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&indicesShape, outputShapeTensor, &valuesShape, &defaultValueShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto outputShape = context->GetOutputShape(kOutputIndex);
    ASSERT_NE(outputShape, nullptr);
    ASSERT_EQ(outputShape->GetDimNum(), 2);
    EXPECT_EQ(outputShape->GetDim(0), 2);
    EXPECT_EQ(outputShape->GetDim(1), 3);
}

TEST_F(SparseToDenseInfershapeTest, InferShapeWithNonConstOutputShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseToDense")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape indicesShape = {2, 2};
    gert::Shape outputShapeShape = {2};
    gert::Shape valuesShape = {2};
    gert::Shape defaultValueShape = {};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&indicesShape, &outputShapeShape, &valuesShape, &defaultValueShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto outputShape = context->GetOutputShape(kOutputIndex);
    ASSERT_NE(outputShape, nullptr);
    ASSERT_EQ(outputShape->GetDimNum(), 2);
    EXPECT_EQ(outputShape->GetDim(0), -1);
    EXPECT_EQ(outputShape->GetDim(1), -1);
}

TEST_F(SparseToDenseInfershapeTest, InferShapeWithUnknownOutputRank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseToDense")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape indicesShape = {2, 2};
    gert::Shape outputShapeShape = {ge::UNKNOWN_DIM};
    gert::Shape valuesShape = {2};
    gert::Shape defaultValueShape = {};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&indicesShape, &outputShapeShape, &valuesShape, &defaultValueShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto outputShape = context->GetOutputShape(kOutputIndex);
    ASSERT_NE(outputShape, nullptr);
    EXPECT_TRUE(Ops::Base::IsUnknownRank(*outputShape));
}

TEST_F(SparseToDenseInfershapeTest, InferShapeFailedWhenOutputShapeNot1d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseToDense")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape indicesShape = {2, 2};
    gert::Shape outputShapeShape = {1, 2};
    gert::Shape valuesShape = {2};
    gert::Shape defaultValueShape = {};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&indicesShape, &outputShapeShape, &valuesShape, &defaultValueShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
