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
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

static ge::graphStatus InferApplyCamePart1Shape(gert::StorageShape& grad)
{
    gert::StorageShape eps = {{1}, {-1}};
    gert::StorageShape sumR = {{}, {}};
    gert::StorageShape sumC = {{}, {}};
    gert::StorageShape sumRC = {{}, {}};
    auto infer = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart1")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 3)
                      .IrInstanceNum({2, 3})
                      .InputShapes({&grad, &eps})
                      .OutputShapes({&sumR, &sumC, &sumRC})
                      .Build();
    return infer(holder.GetContext<gert::InferShapeContext>());
}

TEST(ApplyCamePart1InferShape, TwoDimensionalGrad)
{
    gert::StorageShape grad = {{4, 8}, {-1, -1}};
    gert::StorageShape eps = {{1}, {-1}};
    gert::StorageShape sumR = {{}, {}};
    gert::StorageShape sumC = {{}, {}};
    gert::StorageShape sumRC = {{}, {}};
    auto infer = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart1")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 3)
                      .IrInstanceNum({2, 3})
                      .InputShapes({&grad, &eps})
                      .OutputShapes({&sumR, &sumC, &sumRC})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(infer(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 4);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(0), 8);
    EXPECT_EQ(context->GetOutputShape(2)->GetDimNum(), 0);
}

TEST(ApplyCamePart1InferShape, TwoDimensionalBfloat16Grad)
{
    gert::StorageShape grad = {{65, 67}, {-1, -1}};
    gert::StorageShape eps = {{1}, {-1}};
    gert::StorageShape sumR = {{}, {}};
    gert::StorageShape sumC = {{}, {}};
    gert::StorageShape sumRC = {{}, {}};
    auto infer = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart1")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 3)
                      .IrInstanceNum({2, 3})
                      .InputShapes({&grad, &eps})
                      .OutputShapes({&sumR, &sumC, &sumRC})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(infer(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 65);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(0), 67);
    EXPECT_EQ(context->GetOutputShape(2)->GetDimNum(), 0);
}

TEST(ApplyCamePart1InferShape, SupportsBatchGrad)
{
    gert::StorageShape grad = {{2, 4, 8}, {-1, -1, -1}};
    gert::StorageShape eps = {{1}, {-1}};
    gert::StorageShape sumR = {{}, {}};
    gert::StorageShape sumC = {{}, {}};
    gert::StorageShape sumRC = {{}, {}};
    auto infer = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart1")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 3)
                      .IrInstanceNum({2, 3})
                      .InputShapes({&grad, &eps})
                      .OutputShapes({&sumR, &sumC, &sumRC})
                      .Build();
    ASSERT_EQ(infer(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1)->GetDim(1), 8);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2)->GetDim(0), 2);
}

TEST(ApplyCamePart1InferShape, SupportsHigherRankBatchGrad)
{
    gert::StorageShape grad = {{2, 3, 4, 5}, {-1, -1, -1, -1}};
    gert::StorageShape eps = {{1}, {-1}};
    gert::StorageShape sumR = {{}, {}};
    gert::StorageShape sumC = {{}, {}};
    gert::StorageShape sumRC = {{}, {}};
    auto infer = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart1")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 3)
                      .IrInstanceNum({2, 3})
                      .InputShapes({&grad, &eps})
                      .OutputShapes({&sumR, &sumC, &sumRC})
                      .Build();
    ASSERT_EQ(infer(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)->GetDim(1), 3);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)->GetDim(2), 4);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1)->GetDim(1), 3);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1)->GetDim(2), 5);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2)->GetDim(1), 3);
}

TEST(ApplyCamePart1InferShape, SupportsScalarEps)
{
    gert::StorageShape grad = {{4, 8}, {-1, -1}};
    gert::StorageShape eps = {{}, {}};
    gert::StorageShape sumR = {{}, {}};
    gert::StorageShape sumC = {{}, {}};
    gert::StorageShape sumRC = {{}, {}};
    auto infer = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart1")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 3)
                      .IrInstanceNum({2, 3})
                      .InputShapes({&grad, &eps})
                      .OutputShapes({&sumR, &sumC, &sumRC})
                      .Build();
    ASSERT_EQ(infer(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)->GetDim(0), 4);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1)->GetDim(0), 8);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2)->GetDimNum(), 0);
}

TEST(ApplyCamePart1InferShape, RejectsOneDimensionalGrad)
{
    gert::StorageShape grad = {{0}, {0}};
    EXPECT_EQ(InferApplyCamePart1Shape(grad), ge::GRAPH_FAILED);
}

TEST(ApplyCamePart1InferShape, RejectsNonScalarEps)
{
    gert::StorageShape grad = {{4, 8}, {-1, -1}};
    gert::StorageShape eps = {{2}, {-1}};
    gert::StorageShape sumR = {{}, {}};
    gert::StorageShape sumC = {{}, {}};
    gert::StorageShape sumRC = {{}, {}};
    auto infer = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart1")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 3)
                      .IrInstanceNum({2, 3})
                      .InputShapes({&grad, &eps})
                      .OutputShapes({&sumR, &sumC, &sumRC})
                      .Build();
    EXPECT_EQ(infer(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(ApplyCamePart1InferShape, RejectsZeroRowDimension)
{
    gert::StorageShape grad = {{0, 4}, {0, 4}};
    EXPECT_EQ(InferApplyCamePart1Shape(grad), ge::GRAPH_FAILED);
}

TEST(ApplyCamePart1InferShape, RejectsZeroColumnDimension)
{
    gert::StorageShape grad = {{4, 0}, {4, 0}};
    EXPECT_EQ(InferApplyCamePart1Shape(grad), ge::GRAPH_FAILED);
}

TEST(ApplyCamePart1InferShape, RejectsBothZeroDimensions)
{
    gert::StorageShape grad = {{0, 0}, {0, 0}};
    EXPECT_EQ(InferApplyCamePart1Shape(grad), ge::GRAPH_FAILED);
}

TEST(ApplyCamePart1InferShape, RejectsZeroBatchDimension)
{
    gert::StorageShape grad = {{0, 4, 8}, {0, 4, 8}};
    EXPECT_EQ(InferApplyCamePart1Shape(grad), ge::GRAPH_FAILED);
}

TEST(ApplyCamePart1InferShape, UnknownRankPropagatesToAllOutputs)
{
    gert::StorageShape grad = {{-2}, {-2}};
    gert::StorageShape eps = {{1}, {-1}};
    gert::StorageShape sumR = {{}, {}};
    gert::StorageShape sumC = {{}, {}};
    gert::StorageShape sumRC = {{}, {}};
    auto infer = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart1")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 3)
                      .IrInstanceNum({2, 3})
                      .InputShapes({&grad, &eps})
                      .OutputShapes({&sumR, &sumC, &sumRC})
                      .Build();
    ASSERT_EQ(infer(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    for (size_t index = 0; index < 3; ++index) {
        const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(index);
        ASSERT_NE(output, nullptr);
        EXPECT_EQ(output->GetDimNum(), 1);
        EXPECT_EQ(output->GetDim(0), -2);
    }
}
