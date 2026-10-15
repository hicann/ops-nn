/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <utility>
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

TEST(ApplyCamePart1InferShape, ValidatesEpsShape)
{
    const std::array<std::pair<gert::StorageShape, ge::graphStatus>, 9> cases = {
        std::make_pair(gert::StorageShape({{}, {}}), ge::GRAPH_SUCCESS),
        std::make_pair(gert::StorageShape({{1}, {1}}), ge::GRAPH_SUCCESS),
        std::make_pair(gert::StorageShape({{1, 1}, {1, 1}}), ge::GRAPH_SUCCESS),
        std::make_pair(gert::StorageShape({{2}, {2}}), ge::GRAPH_FAILED),
        std::make_pair(gert::StorageShape({{2147483648LL}, {2147483648LL}}), ge::GRAPH_FAILED),
        std::make_pair(gert::StorageShape({{4294967296LL}, {4294967296LL}}), ge::GRAPH_FAILED),
        std::make_pair(gert::StorageShape({{-1}, {-1}}), ge::GRAPH_SUCCESS),
        std::make_pair(gert::StorageShape({{-1, 2}, {-1, 2}}), ge::GRAPH_FAILED),
        std::make_pair(gert::StorageShape({{-2}, {-2}}), ge::GRAPH_SUCCESS)};
    for (const auto& [epsShape, expected] : cases) {
        gert::StorageShape grad = {{4, 8}, {-1, -1}};
        gert::StorageShape eps = epsShape;
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
        ASSERT_EQ(infer(holder.GetContext<gert::InferShapeContext>()), expected);
        if (expected != ge::GRAPH_SUCCESS) {
            continue;
        }
        EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)->GetDim(0), 4);
        EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1)->GetDim(0), 8);
        EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2)->GetDimNum(), 0);
    }
}

TEST(ApplyCamePart1InferShape, UnknownGradRankDoesNotHideInvalidEps)
{
    gert::StorageShape grad = {{-2}, {-2}};
    gert::StorageShape eps = {{2147483648LL}, {2147483648LL}};
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

TEST(ApplyCamePart1InferShape, RejectsZeroNonLeadingBatchDimension)
{
    gert::StorageShape grad = {{2, 0, 4, 8}, {2, 0, 4, 8}};
    EXPECT_EQ(InferApplyCamePart1Shape(grad), ge::GRAPH_FAILED);
}

TEST(ApplyCamePart1InferShape, RejectsMultipleZeroDimensions)
{
    gert::StorageShape grad = {{2, 0, 0, 8}, {2, 0, 0, 8}};
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

TEST(ApplyCamePart1InferShape, InfersFloatOutputsForAllGradTypes)
{
    for (const auto inputType : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        std::array<ge::DataType, 2> inputs = {inputType, ge::DT_FLOAT};
        std::array<ge::DataType, 3> outputs = {ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_UNDEFINED};
        auto holder = gert::InferDataTypeContextFaker()
                          .IrInputNum(2)
                          .NodeIoNum(2, 3)
                          .IrInstanceNum({2, 3})
                          .InputDataTypes({&inputs[0], &inputs[1]})
                          .OutputDataTypes({&outputs[0], &outputs[1], &outputs[2]})
                          .Build();
        auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart1");
        ASSERT_NE(impl, nullptr);
        ASSERT_NE(impl->infer_datatype, nullptr);
        auto* context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
        EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
        EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
    }
}
