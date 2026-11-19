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
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/max_pool_grad_with_argmax_proto.h"

namespace {
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

class MaxPoolGradWithArgmaxInfer : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaxPoolGradWithArgmaxInferTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "max_pool_grad_with_argmax__infer_test TearDown" << std::endl; }
};

TEST_F(MaxPoolGradWithArgmaxInfer, max_pool_grad_with_argmax_infershape_test_00)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGradWithArgmax")->infer_shape;

    gert::StorageShape xShape = {{4, 512, 16, 16}, {}};
    gert::StorageShape gradShape = {{}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"include_batch_in_index", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
                      .InputShapes({&xShape, &gradShape, &indicesShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[4, 512, 16, 16]");
}

TEST_F(MaxPoolGradWithArgmaxInfer, max_pool_grad_with_argmax_infershape_test_02)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGradWithArgmax")->infer_shape;

    gert::StorageShape xShape = {{4, 512, 16, 16}, {}};
    gert::StorageShape gradShape = {{4, 512, 16, 16}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{4, 512, -1, 16}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"include_batch_in_index", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
                      .InputShapes({&xShape, &gradShape, &indicesShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[-1, -1, -1, -1]");
}

TEST_F(MaxPoolGradWithArgmaxInfer, max_pool_grad_with_argmax_infershape_test_03)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGradWithArgmax")->infer_shape;

    gert::StorageShape xShape = {{4, 512, 16, 16}, {}};
    gert::StorageShape gradShape = {{4, 512, 8, 8}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{4, 512, 8, 8}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"include_batch_in_index", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
                      .InputShapes({&xShape, &gradShape, &indicesShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[4, 512, 16, 16]");
}

void RunInferShapeCase(const ge::Format xFormat, const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides,
                       const std::string& padding, const std::string& dataFormat, const ge::graphStatus expectStatus)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGradWithArgmax")->infer_shape;

    gert::StorageShape xShape = {{4, 512, 16, 16}, {}};
    gert::StorageShape gradShape = {{4, 512, 8, 8}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{4, 512, 8, 8}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, xFormat, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, xFormat, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_INT32, xFormat, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(padding)},
                                  {"include_batch_in_index", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(dataFormat)}})
                      .InputShapes({&xShape, &gradShape, &indicesShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), expectStatus);
}

TEST_F(MaxPoolGradWithArgmaxInfer, max_pool_grad_with_argmax_infershape_null_context)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGradWithArgmax")->infer_shape;
    ASSERT_EQ(inferShapeFunc(nullptr), ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmaxInfer, max_pool_grad_with_argmax_infershape_invalid_x_format)
{
    RunInferShapeCase(ge::Format::FORMAT_NCDHW, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME", "NCHW", ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmaxInfer, max_pool_grad_with_argmax_infershape_ksize_list_size_invalid)
{
    RunInferShapeCase(ge::Format::FORMAT_NCHW, {1, 2}, {1, 2, 2, 1}, "SAME", "NCHW", ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmaxInfer, max_pool_grad_with_argmax_infershape_strides_list_size_invalid)
{
    RunInferShapeCase(ge::Format::FORMAT_NCHW, {1, 2, 2, 1}, {1, 2, 3}, "SAME", "NCHW", ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmaxInfer, max_pool_grad_with_argmax_infershape_batch_dim_invalid)
{
    RunInferShapeCase(ge::Format::FORMAT_NCHW, {2, 2, 2, 1}, {1, 2, 2, 1}, "SAME", "NCHW", ge::GRAPH_FAILED);
    RunInferShapeCase(ge::Format::FORMAT_NCHW, {1, 2, 2, 1}, {2, 2, 2, 1}, "SAME", "NCHW", ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmaxInfer, max_pool_grad_with_argmax_infershape_padding_invalid)
{
    RunInferShapeCase(ge::Format::FORMAT_NCHW, {1, 2, 2, 1}, {1, 2, 2, 1}, "CAUSAL", "NCHW", ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmaxInfer, max_pool_grad_with_argmax_infershape_nhwc_channel_dim_invalid)
{
    RunInferShapeCase(ge::Format::FORMAT_NHWC, {1, 2, 2, 2}, {1, 2, 2, 1}, "SAME", "NHWC", ge::GRAPH_FAILED);
    RunInferShapeCase(ge::Format::FORMAT_NHWC, {1, 2, 2, 1}, {1, 2, 2, 2}, "SAME", "NHWC", ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmaxInfer, max_pool_grad_with_argmax_infershape_nchw_success)
{
    RunInferShapeCase(ge::Format::FORMAT_NCHW, {1, 1, 2, 1}, {1, 1, 2, 1}, "VALID", "NCHW", ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolGradWithArgmaxInfer, max_pool_grad_with_argmax_infershape_nchw_channel_dim_invalid)
{
    RunInferShapeCase(ge::Format::FORMAT_NCHW, {1, 2, 2, 1}, {1, 1, 2, 1}, "SAME", "NCHW", ge::GRAPH_FAILED);
    RunInferShapeCase(ge::Format::FORMAT_NCHW, {1, 1, 2, 1}, {1, 2, 2, 1}, "SAME", "NCHW", ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmaxInfer, max_pool_grad_with_argmax_infershape_nd_format_success)
{
    RunInferShapeCase(ge::Format::FORMAT_ND, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME", "NHWC", ge::GRAPH_SUCCESS);
}

} // namespace
