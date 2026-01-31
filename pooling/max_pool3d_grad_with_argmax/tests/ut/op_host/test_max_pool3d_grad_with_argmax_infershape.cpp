/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "../../../op_graph/max_pool3d_grad_with_argmax_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "error_util.h"
#include "log/log.h"

using namespace ge;
using namespace op;

class MaxPool3DGradWithArgmaxInferShapeTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPool3DGradWithArgmax InferShape Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaxPool3DGradWithArgmax InferShape Test TearDown" << std::endl;
    }
};

TEST_F(MaxPool3DGradWithArgmaxInferShapeTest, max_pool3d_grad_with_argmax_infershape_test1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGradWithArgmax")->infer_shape;

    gert::StorageShape xShape = {{3698, 2, 2, 3, 2}, {3698, 2, 2, 3, 2}};
    gert::StorageShape gradShape = {{3698, 2, 2, 1, 2}, {3698, 2, 2, 1, 2}};
    gert::StorageShape yShape = {{3698, 2, 2, 3, 2}, {3698, 2, 2, 3, 2}};
    gert::StorageShape indicesShape = {{3698, 2, 2, 1, 2}, {3698, 2, 2, 1, 2}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 7, 1})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({})},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 0})},
                                  {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2474, 1, 1})},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")}})
                      .InputShapes({&xShape, &gradShape, &indicesShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[3698, 2, 2, 3, 2]");
}

TEST_F(MaxPool3DGradWithArgmaxInferShapeTest, max_pool3d_grad_with_argmax_infershape_test2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGradWithArgmax")->infer_shape;

    gert::StorageShape xShape = {{8, 5, 6, 8, 8}, {8, 5, 6, 8, 8}};
    gert::StorageShape gradShape = {{8, 1, 1, 2, 8}, {8, 1, 1, 2, 8}};
    gert::StorageShape yShape = {{8, 5, 6, 8, 8}, {8, 5, 6, 8, 8}};
    gert::StorageShape indicesShape = {{8, 1, 1, 2, 8}, {8, 1, 1, 2, 8}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_INT64, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({6, 7, 7})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({7404, 6, 6})},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                  {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &gradShape, &indicesShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[8, 5, 6, 8, 8]");
}