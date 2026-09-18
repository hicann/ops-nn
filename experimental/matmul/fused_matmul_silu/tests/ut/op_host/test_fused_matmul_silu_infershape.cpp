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

#include "infer_shape_context_faker.h"
#include "op_impl_registry.h"

#include "../../../op_graph/fused_matmul_silu_proto.h"

class FusedMatmulSiluInferShapeTest : public testing::Test {};

TEST_F(FusedMatmulSiluInferShapeTest, infer_shape_success)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedMatmulSilu")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {2, 256};
    gert::Shape weightShape = {4096, 256};
    gert::Shape biasShape = {4096};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape, &weightShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto outputShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(outputShape, nullptr);
    EXPECT_EQ(outputShape->GetDimNum(), 2);
    EXPECT_EQ(outputShape->GetDim(0), 2);
    EXPECT_EQ(outputShape->GetDim(1), 4096);
}

TEST_F(FusedMatmulSiluInferShapeTest, infer_shape_dynamic_m)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedMatmulSilu")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {-1, 256};
    gert::Shape weightShape = {4096, 256};
    gert::Shape biasShape = {4096};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape, &weightShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto outputShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(outputShape, nullptr);
    EXPECT_EQ(outputShape->GetDimNum(), 2);
    EXPECT_EQ(outputShape->GetDim(0), -1);
    EXPECT_EQ(outputShape->GetDim(1), 4096);
}

TEST_F(FusedMatmulSiluInferShapeTest, infer_shape_invalid_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedMatmulSilu")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {2, 4, 256};
    gert::Shape weightShape = {4096, 256};
    gert::Shape biasShape = {4096};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape, &weightShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(FusedMatmulSiluInferShapeTest, infer_shape_invalid_k)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedMatmulSilu")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {2, 128};
    gert::Shape weightShape = {4096, 256};
    gert::Shape biasShape = {4096};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape, &weightShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(FusedMatmulSiluInferShapeTest, infer_shape_unaligned_k)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedMatmulSilu")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {2, 255};
    gert::Shape weightShape = {4096, 255};
    gert::Shape biasShape = {4096};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape, &weightShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(FusedMatmulSiluInferShapeTest, infer_shape_unsupported_k)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedMatmulSilu")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {2, 8192};
    gert::Shape weightShape = {4096, 8192};
    gert::Shape biasShape = {4096};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape, &weightShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(FusedMatmulSiluInferShapeTest, infer_shape_invalid_bias)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedMatmulSilu")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {2, 256};
    gert::Shape weightShape = {4096, 256};
    gert::Shape biasShape = {2048};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape, &weightShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
