/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
// #include "matrix_calculation_ops.h"
#include "register/op_impl_registry.h"
// #include "error_util.h"
#include "../../../op_graph/avg_pool3_d_grad_proto.h"

namespace avgpool3dgrad_infershpe_ut {
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
} // namespace avgpool3dgrad_infershpe_ut
class AvgPool3DGradRuntimeProtoTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AvgPool3DGradRuntimeProtoTest Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AvgPool3DGradRuntimeProtoTest Proto Test TearDown" << std::endl;
    }
};

TEST_F(AvgPool3DGradRuntimeProtoTest, basic)
{
    vector<int64_t> ksize = {5, 7, 1, 24, 224};
    vector<int64_t> strides({1, 3, 3, 1, 1});
    vector<int64_t> pads({2, 2, 2, 3, 0, 0});
    string data_format("NDHWC");

    vector<int64_t> orig_input = {32, 16, 8, 8, 24};
    gert::StorageShape orig_input_shape = {{32, 16, 8, 8, 24}, {32, 16, 8, 8, 24}};
    gert::StorageShape grads_shape = {{32, 6, 3, 8, 224}, {32, 6, 3, 8, 224}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder =
        gert::Tensor::CreateFollowing(orig_input_shape.GetStorageShape().GetDimNum(), ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(orig_input_shape.MutableStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(orig_input_shape.MutableOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NDHWC);
    tensor->SetStorageFormat(ge::FORMAT_NDHWC);
    (void)memcpy_s(
        tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), orig_input.data(),
        orig_input.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &grads_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)}})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .Build();

    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool3DGrad")->infer_shape;
    ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(avgpool3dgrad_infershpe_ut::Shape2String(*output), "[32, 16, 8, 8, 24]");
}
