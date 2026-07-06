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
#include <gtest/gtest.h>
#include <vector>
#include "array_ops.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
using namespace ge;

class ThresholdGradV2DInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ThresholdGradV2DInferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ThresholdGradV2DInferShapeTest TearDown" << std::endl; }
};

TEST_F(ThresholdGradV2DInferShapeTest, test_same_shape_fp32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ThresholdGradV2D")->infer_shape;
    gert::StorageShape gradOutput_shape = {{16, 16}, {16, 16}};
    gert::StorageShape self_shape = {{16, 16}, {16, 16}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradOutput_shape, &self_shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {16, 16};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(ThresholdGradV2DInferShapeTest, test_broadcast_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ThresholdGradV2D")->infer_shape;
    gert::StorageShape gradOutput_shape = {{4, 1}, {4, 1}};
    gert::StorageShape self_shape = {{1, 8}, {1, 8}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradOutput_shape, &self_shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {4, 8};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(ThresholdGradV2DInferShapeTest, test_fp16_same_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ThresholdGradV2D")->infer_shape;
    gert::StorageShape gradOutput_shape = {{32, 64, 128}, {32, 64, 128}};
    gert::StorageShape self_shape = {{32, 64, 128}, {32, 64, 128}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradOutput_shape, &self_shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ThresholdGradV2DInferShapeTest, test_bf16_same_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ThresholdGradV2D")->infer_shape;
    gert::StorageShape gradOutput_shape = {{2, 64}, {2, 64}};
    gert::StorageShape self_shape = {{2, 64}, {2, 64}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradOutput_shape, &self_shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ThresholdGradV2DInferShapeTest, test_int32_same_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ThresholdGradV2D")->infer_shape;
    gert::StorageShape gradOutput_shape = {{4, 4}, {4, 4}};
    gert::StorageShape self_shape = {{4, 4}, {4, 4}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradOutput_shape, &self_shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ThresholdGradV2DInferShapeTest, test_int8_same_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ThresholdGradV2D")->infer_shape;
    gert::StorageShape gradOutput_shape = {{8, 8}, {8, 8}};
    gert::StorageShape self_shape = {{8, 8}, {8, 8}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradOutput_shape, &self_shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ThresholdGradV2DInferShapeTest, test_uint8_same_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ThresholdGradV2D")->infer_shape;
    gert::StorageShape gradOutput_shape = {{16}, {16}};
    gert::StorageShape self_shape = {{16}, {16}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradOutput_shape, &self_shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ThresholdGradV2DInferShapeTest, test_8d_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ThresholdGradV2D")->infer_shape;
    gert::StorageShape gradOutput_shape = {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}};
    gert::StorageShape self_shape = {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradOutput_shape, &self_shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ThresholdGradV2DInferShapeTest, test_empty_tensor)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ThresholdGradV2D")->infer_shape;
    gert::StorageShape gradOutput_shape = {{0, 0}, {0, 0}};
    gert::StorageShape self_shape = {{0, 0}, {0, 0}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradOutput_shape, &self_shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ThresholdGradV2DInferShapeTest, test_scalar_input)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ThresholdGradV2D")->infer_shape;
    gert::StorageShape gradOutput_shape = {{}, {}};
    gert::StorageShape self_shape = {{}, {}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradOutput_shape, &self_shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ThresholdGradV2DInferShapeTest, test_large_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ThresholdGradV2D")->infer_shape;
    gert::StorageShape gradOutput_shape = {{1024, 1024}, {1024, 1024}};
    gert::StorageShape self_shape = {{1024, 1024}, {1024, 1024}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradOutput_shape, &self_shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ThresholdGradV2DInferShapeTest, test_broadcast_different_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ThresholdGradV2D")->infer_shape;
    gert::StorageShape gradOutput_shape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape self_shape = {{4}, {4}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradOutput_shape, &self_shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {2, 3, 4};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}
