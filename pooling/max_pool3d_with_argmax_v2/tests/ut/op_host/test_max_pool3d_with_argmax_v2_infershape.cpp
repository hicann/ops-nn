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
#include "../../../op_graph/max_pool3d_with_argmax_v2_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "error_util.h"
#include "log/log.h"

using namespace ge;
using namespace op;

class MaxPool3DWithArgmaxV2InferShapeTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPool3DWithArgmaxV2 InferShape Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaxPool3DWithArgmaxV2 InferShape Test TearDown" << std::endl;
    }
};

TEST_F(MaxPool3DWithArgmaxV2InferShapeTest, max_pool3d_with_argmax_v2_infershape_test1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DWithArgmaxV2")->infer_shape;

    gert::StorageShape xShape = {{4, 512, 16, 16, 16}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
                                  {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &indicesShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* indices = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Shape2String(*output), "[4, 512, 8, 8, 8]");
    ASSERT_EQ(Shape2String(*indices), "[4, 512, 8, 8, 8]");
}

TEST_F(MaxPool3DWithArgmaxV2InferShapeTest, max_pool3d_with_argmax_v2_infershape_test2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DWithArgmaxV2")->infer_shape;

    gert::StorageShape xShape = {{5, 256, 144, 144, 589}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3, 3})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3, 3})},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
                                  {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({9, 5, 5})},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &indicesShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* indices = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Shape2String(*output), "[5, 125, 46, 46, 589]");
    ASSERT_EQ(Shape2String(*indices), "[5, 125, 46, 46, 589]");
}