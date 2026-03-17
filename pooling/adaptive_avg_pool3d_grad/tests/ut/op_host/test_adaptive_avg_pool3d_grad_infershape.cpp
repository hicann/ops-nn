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
 * \file test_adaptive_avg_pool3d_grad_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "../../../op_graph/adaptive_avg_pool3d_grad_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "error_util.h"
#include "log/log.h"

using namespace ge;
using namespace op;

class AdaptiveAvgPool3DGradInferShapeTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveAvgPool3DGrad InferShape Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveAvgPool3DGrad InferShape Test TearDown" << std::endl;
    }
};

TEST_F(AdaptiveAvgPool3DGradInferShapeTest, adaptive_avg_pool3d_grad_infershape_test01)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveAvgPool3dGrad")->infer_shape;
    gert::StorageShape yGradShape = {{3, 30, 37, 6, 29}, {3, 30, 37, 6, 29}};
    gert::StorageShape xShape = {{3, 30, 5, 1, 24}, {3, 30, 5, 1, 24}};
    gert::StorageShape xGradShape = {{3, 30, 5, 1, 24}, {3, 30, 5, 1, 24}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")}
                      })
                      .InputShapes({&yGradShape, &xShape})
                      .OutputShapes({&xGradShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[3, 30, 5, 1, 24]");
}

TEST_F(AdaptiveAvgPool3DGradInferShapeTest, adaptive_avg_pool3d_grad_infershape_test02)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveAvgPool3dGrad")->infer_shape;
    gert::StorageShape yGradShape = {{1, 1, 1, 1, 32768}, {1, 1, 1, 1, 32768}};
    gert::StorageShape xShape = {{1, 1, 1, 1, 16777216}, {1, 1, 1, 1, 16777216}};
    gert::StorageShape xGradShape = {{1, 1, 1, 1, 16777216}, {1, 1, 1, 1, 16777216}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")}
                      })
                      .InputShapes({&yGradShape, &xShape})
                      .OutputShapes({&xGradShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 1, 1, 1, 16777216]");
}