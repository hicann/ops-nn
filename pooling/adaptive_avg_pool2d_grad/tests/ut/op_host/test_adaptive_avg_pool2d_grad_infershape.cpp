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
 * \file test_adaptive_avg_pool2d_grad_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "error_util.h"
#include "log/log.h"

using namespace ge;

class AdaptiveAvgPool2dGradInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdaptiveAvgPool2dGrad InferShape Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AdaptiveAvgPool2dGrad InferShape Test TearDown" << std::endl; }
};

TEST_F(AdaptiveAvgPool2dGradInferShapeTest, adaptive_avg_pool2d_grad_infershape_nchw)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveAvgPool2dGrad")->infer_shape;
    gert::StorageShape yGradShape = {{2, 2, 4, 4}, {2, 2, 4, 4}};
    gert::StorageShape xShape = {{2, 2, 8, 8}, {2, 2, 8, 8}};
    gert::StorageShape xGradShape = {{2, 2, 8, 8}, {2, 2, 8, 8}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .InputShapes({&yGradShape, &xShape})
                      .OutputShapes({&xGradShape})
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 8, 8})}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[2, 2, 8, 8]");
}

TEST_F(AdaptiveAvgPool2dGradInferShapeTest, adaptive_avg_pool2d_grad_infershape_small_kernel_nchw_fp32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveAvgPool2dGrad")->infer_shape;
    gert::StorageShape yGradShape = {{1, 128, 8, 8}, {1, 128, 8, 8}};
    gert::StorageShape xShape = {{1, 128, 16, 16}, {1, 128, 16, 16}};
    gert::StorageShape xGradShape = {{1, 128, 16, 16}, {1, 128, 16, 16}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .InputShapes({&yGradShape, &xShape})
                      .OutputShapes({&xGradShape})
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 128, 16, 16})}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 128, 16, 16]");
}

TEST_F(AdaptiveAvgPool2dGradInferShapeTest, adaptive_avg_pool2d_grad_infershape_small_kernel_chw_fp16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveAvgPool2dGrad")->infer_shape;

    gert::StorageShape yGradShape = {{128, 16, 16}, {128, 16, 16}};
    gert::StorageShape xShape = {{128, 32, 32}, {128, 32, 32}};
    gert::StorageShape xGradShape = {{128, 32, 32}, {128, 32, 32}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCL, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCL, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCL, ge::Format::FORMAT_RESERVED)
                      .InputShapes({&yGradShape, &xShape})
                      .OutputShapes({&xGradShape})
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({128, 32, 32})}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[128, 32, 32]");
}

TEST_F(AdaptiveAvgPool2dGradInferShapeTest, adaptive_avg_pool2d_grad_infershape_simt_kernel_nchw_fp32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveAvgPool2dGrad")->infer_shape;

    gert::StorageShape yGradShape = {{1, 4, 8, 8}, {1, 4, 8, 8}};
    gert::StorageShape xShape = {{1, 4, 16, 16}, {1, 4, 16, 16}};
    gert::StorageShape xGradShape = {{1, 4, 16, 16}, {1, 4, 16, 16}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .InputShapes({&yGradShape, &xShape})
                      .OutputShapes({&xGradShape})
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 4, 16, 16})}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 4, 16, 16]");
}

TEST_F(AdaptiveAvgPool2dGradInferShapeTest, adaptive_avg_pool2d_grad_infershape_big_kernel_nchw_fp32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveAvgPool2dGrad")->infer_shape;

    gert::StorageShape yGradShape = {{1, 128, 16, 1}, {1, 128, 16, 1}};
    gert::StorageShape xShape = {{1, 128, 16, 512}, {1, 128, 16, 512}};
    gert::StorageShape xGradShape = {{1, 128, 16, 512}, {1, 128, 16, 512}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .InputShapes({&yGradShape, &xShape})
                      .OutputShapes({&xGradShape})
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 128, 16, 512})}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 128, 16, 512]");
}

TEST_F(AdaptiveAvgPool2dGradInferShapeTest, adaptive_avg_pool2d_grad_infershape_big_kernel_chw_fp16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveAvgPool2dGrad")->infer_shape;
    gert::StorageShape yGradShape = {{128, 16, 1}, {128, 16, 1}};
    gert::StorageShape xShape = {{128, 16, 512}, {128, 16, 512}};
    gert::StorageShape xGradShape = {{128, 16, 512}, {128, 16, 512}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCL, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCL, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCL, ge::Format::FORMAT_RESERVED)
                      .InputShapes({&yGradShape, &xShape})
                      .OutputShapes({&xGradShape})
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({128, 16, 512})}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[128, 16, 512]");
}
