/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_modulate_grad_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"

class ModulateGradProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ModulateGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ModulateGrad Proto Test TearDown" << std::endl;
    }
};

TEST_F(ModulateGradProto, modulate_grad_infershape_with_scale_shift)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ModulateGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradOutputShape = {{4, 16, 128}, {4, 16, 128}};
    gert::StorageShape inputShape = {{4, 16, 128}, {4, 16, 128}};
    gert::StorageShape scaleShape = {{4, 128}, {4, 128}};
    gert::StorageShape shiftShape = {{4, 128}, {4, 128}};
    gert::StorageShape gradInputShape;
    gert::StorageShape gradScaleShape;
    gert::StorageShape gradShiftShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 3)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({&gradOutputShape, &inputShape, &scaleShape, &shiftShape})
                      .OutputShapes({&gradInputShape, &gradScaleShape, &gradShiftShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[4, 16, 128]");
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(1)), "[4, 128]");
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(2)), "[4, 128]");
}

TEST_F(ModulateGradProto, modulate_grad_infershape_without_scale_shift)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ModulateGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradOutputShape = {{4, 16, 128}, {4, 16, 128}};
    gert::StorageShape inputShape = {{4, 16, 128}, {4, 16, 128}};
    gert::StorageShape gradInputShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1, 0, 0}, {1, 0, 0})
                      .InputShapes({&gradOutputShape, &inputShape})
                      .OutputShapes({&gradInputShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[4, 16, 128]");
}

// Real GE keeps optional outputs at their fixed IR index (no compaction); an absent optional is a
// present-but-empty placeholder. So all 4 inputs / 3 outputs exist, the absent optional input is an
// empty shape, and grad_scale/grad_shift stay at IR index 1/2 respectively.
TEST_F(ModulateGradProto, modulate_grad_infershape_scale_only)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ModulateGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradOutputShape = {{4, 16, 128}, {4, 16, 128}};
    gert::StorageShape inputShape = {{4, 16, 128}, {4, 16, 128}};
    gert::StorageShape scaleShape = {{4, 128}, {4, 128}};
    gert::StorageShape shiftShape = {{}, {}};  // shift absent -> empty placeholder at fixed IR index 3
    gert::StorageShape gradInputShape;
    gert::StorageShape gradScaleShape;
    gert::StorageShape gradShiftShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 3)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({&gradOutputShape, &inputShape, &scaleShape, &shiftShape})
                      .OutputShapes({&gradInputShape, &gradScaleShape, &gradShiftShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[4, 16, 128]");
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(1)), "[4, 128]");  // grad_scale at fixed IR index 1
}

TEST_F(ModulateGradProto, modulate_grad_infershape_shift_only)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ModulateGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradOutputShape = {{4, 16, 128}, {4, 16, 128}};
    gert::StorageShape inputShape = {{4, 16, 128}, {4, 16, 128}};
    gert::StorageShape scaleShape = {{}, {}};  // scale absent -> empty placeholder at fixed IR index 2
    gert::StorageShape shiftShape = {{4, 128}, {4, 128}};
    gert::StorageShape gradInputShape;
    gert::StorageShape gradScaleShape;
    gert::StorageShape gradShiftShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 3)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({&gradOutputShape, &inputShape, &scaleShape, &shiftShape})
                      .OutputShapes({&gradInputShape, &gradScaleShape, &gradShiftShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[4, 16, 128]");
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(2)), "[4, 128]");  // grad_shift at fixed IR index 2
}

TEST_F(ModulateGradProto, modulate_grad_inferdtype_with_scale_shift)
{
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ModulateGrad")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType inputRef = ge::DT_FLOAT16;
    ge::DataType outputRef = ge::DT_FLOAT16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(4)
                              .NodeIoNum(4, 3)
                              .IrInstanceNum({1, 1, 1, 1}, {1, 1, 1})
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&inputRef, &inputRef, &inputRef, &inputRef})
                              .OutputDataTypes({&outputRef, &outputRef, &outputRef})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), outputRef);
    EXPECT_EQ(context->GetOutputDataType(1), outputRef);
    EXPECT_EQ(context->GetOutputDataType(2), outputRef);
}

TEST_F(ModulateGradProto, modulate_grad_inferdtype_shift_only)
{
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ModulateGrad")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType inputRef = ge::DT_FLOAT16;
    ge::DataType outputRef = ge::DT_FLOAT16;
    ge::DataType undefRef = ge::DT_UNDEFINED;
    // Real GE (no compaction): scale absent -> input 2 is an empty placeholder with DT_UNDEFINED;
    // grad_scale (output 1) stays undefined, grad_shift dtype stays at fixed IR index 2.
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(4)
                              .NodeIoNum(4, 3)
                              .IrInstanceNum({1, 1, 1, 1}, {1, 1, 1})
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&inputRef, &inputRef, &undefRef, &inputRef})
                              .OutputDataTypes({&outputRef, &undefRef, &outputRef})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), outputRef);   // grad_input
    EXPECT_EQ(context->GetOutputDataType(2), outputRef);   // grad_shift dtype at fixed IR index 2
}
