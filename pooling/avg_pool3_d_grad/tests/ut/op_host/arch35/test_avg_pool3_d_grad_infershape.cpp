/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <memory>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

namespace avgpool3dgrad_infershape_ut {
template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}
} // namespace avgpool3dgrad_infershape_ut

class AvgPool3DGradInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AvgPool3DGradInferShape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "AvgPool3DGradInferShape TearDown" << std::endl; }
};

// Build a const tensor holding the orig_input_shape values. holder keeps the memory alive.
template <typename T>
static gert::Tensor* MakeConstShapeTensor(const std::vector<T>& shape, ge::DataType dtype,
                                          std::unique_ptr<uint8_t[]>& holder)
{
    size_t total_size = 0;
    holder = gert::Tensor::CreateFollowing(shape.size(), dtype, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(holder.get());
    tensor->MutableStorageShape().AppendDim(shape.size());
    tensor->MutableOriginShape().AppendDim(shape.size());
    tensor->SetOriginFormat(ge::FORMAT_NDHWC);
    tensor->SetStorageFormat(ge::FORMAT_NDHWC);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), shape.data(),
                   shape.size() * sizeof(T));
    return tensor;
}

static std::string RunInferShape(gert::Tensor* constTensor, gert::StorageShape& inputShape,
                                 gert::StorageShape& outputShape, ge::DataType gradDtype)
{
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool3DGrad")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({constTensor, &inputShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto ctx = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(ctx), ge::GRAPH_SUCCESS);
    auto output = ctx->GetOutputShape(0);
    return avgpool3dgrad_infershape_ut::Shape2String(*output);
}

// ============================ Success cases ============================

TEST_F(AvgPool3DGradInferShape, int32_orig_shape)
{
    std::vector<int32_t> shape_data = {2, 16, 8, 8, 8};
    std::unique_ptr<uint8_t[]> holder;
    auto constTensor = MakeConstShapeTensor<int32_t>(shape_data, ge::DT_INT32, holder);
    gert::StorageShape inputShape = {{2, 16, 4, 4, 4}, {2, 16, 4, 4, 4}};
    gert::StorageShape outputShape = {{}, {}};
    auto outStr = RunInferShape(constTensor, inputShape, outputShape, ge::DT_FLOAT);
    ASSERT_EQ(outStr, "[2, 16, 8, 8, 8]");
}

TEST_F(AvgPool3DGradInferShape, int64_orig_shape)
{
    std::vector<int64_t> shape_data = {1, 32, 16, 16, 16};
    std::unique_ptr<uint8_t[]> holder;
    auto constTensor = MakeConstShapeTensor<int64_t>(shape_data, ge::DT_INT64, holder);
    gert::StorageShape inputShape = {{1, 32, 8, 8, 8}, {1, 32, 8, 8, 8}};
    gert::StorageShape outputShape = {{}, {}};
    auto outStr = RunInferShape(constTensor, inputShape, outputShape, ge::DT_FLOAT16);
    ASSERT_EQ(outStr, "[1, 32, 16, 16, 16]");
}

// ============================ Failure cases ============================

TEST_F(AvgPool3DGradInferShape, invalid_dim_num)
{
    std::vector<int32_t> shape_data = {4, 4, 4}; // len 3, expect 5
    std::unique_ptr<uint8_t[]> holder;
    auto constTensor = MakeConstShapeTensor<int32_t>(shape_data, ge::DT_INT32, holder);
    gert::StorageShape inputShape = {{1, 1, 4, 4, 4}, {1, 1, 4, 4, 4}};
    gert::StorageShape outputShape = {{}, {}};
    auto infer = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool3DGrad")->infer_shape;
    auto faker = gert::InferShapeContextFaker()
                     .NodeIoNum(2, 1)
                     .IrInstanceNum({1, 1, 1})
                     .InputShapes({constTensor, &inputShape})
                     .OutputShapes({&outputShape})
                     .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
                                 {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
                                 {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
                                 {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                     .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                     .Build();
    ASSERT_EQ(infer(faker.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradInferShape, unsupported_dtype)
{
    std::vector<int32_t> shape_data = {1, 1, 4, 4, 4};
    std::unique_ptr<uint8_t[]> holder;
    auto constTensor = MakeConstShapeTensor<int32_t>(shape_data, ge::DT_FLOAT, holder); // wrong dtype
    gert::StorageShape inputShape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape outputShape = {{}, {}};
    auto infer = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool3DGrad")->infer_shape;
    auto faker = gert::InferShapeContextFaker()
                     .NodeIoNum(2, 1)
                     .IrInstanceNum({1, 1, 1})
                     .InputShapes({constTensor, &inputShape})
                     .OutputShapes({&outputShape})
                     .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
                                 {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
                                 {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
                                 {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                     .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                     .Build();
    ASSERT_EQ(infer(faker.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
