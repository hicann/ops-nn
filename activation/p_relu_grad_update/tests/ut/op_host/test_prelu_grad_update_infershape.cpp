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
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "log/log.h"
using namespace ge;

// PReluGradUpdate: inputs(grads 0, features 1, weights 2) -> outputs(dx 0, update 1)
// dx/update shape = BroadcastShape(grads, features); dx/update dtype = grads dtype
class PReluGradUpdateInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PReluGradUpdateInferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "PReluGradUpdateInferShapeTest TearDown" << std::endl; }
};

TEST_F(PReluGradUpdateInferShapeTest, test_same_shape_broadcast)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("PReluGradUpdate")->infer_shape;
    gert::StorageShape gradsShape = {{16, 16}, {16, 16}};
    gert::StorageShape featuresShape = {{16, 16}, {16, 16}};
    gert::StorageShape weightsShape = {{16}, {16}};
    gert::StorageShape dxShape = {{}, {}};
    gert::StorageShape updateShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&gradsShape, &featuresShape, &weightsShape})
                      .OutputShapes({&dxShape, &updateShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto dxDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedDxShape = {16, 16};
    ASSERT_EQ(Ops::Base::ToString(*dxDesc), Ops::Base::ToString(expectedDxShape));
    auto updateDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*updateDesc), Ops::Base::ToString(expectedDxShape));
}

TEST_F(PReluGradUpdateInferShapeTest, test_broadcast_grads_features)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("PReluGradUpdate")->infer_shape;
    gert::StorageShape gradsShape = {{4, 1}, {4, 1}};
    gert::StorageShape featuresShape = {{1, 8}, {1, 8}};
    gert::StorageShape weightsShape = {{8}, {8}};
    gert::StorageShape dxShape = {{}, {}};
    gert::StorageShape updateShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&gradsShape, &featuresShape, &weightsShape})
                      .OutputShapes({&dxShape, &updateShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto dxDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedShape = {4, 8};
    ASSERT_EQ(Ops::Base::ToString(*dxDesc), Ops::Base::ToString(expectedShape));
    auto updateDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*updateDesc), Ops::Base::ToString(expectedShape));
}

TEST_F(PReluGradUpdateInferShapeTest, test_3d_broadcast)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("PReluGradUpdate")->infer_shape;
    gert::StorageShape gradsShape = {{2, 32, 64}, {2, 32, 64}};
    gert::StorageShape featuresShape = {{2, 32, 64}, {2, 32, 64}};
    gert::StorageShape weightsShape = {{64}, {64}};
    gert::StorageShape dxShape = {{}, {}};
    gert::StorageShape updateShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&gradsShape, &featuresShape, &weightsShape})
                      .OutputShapes({&dxShape, &updateShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto dxDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedShape = {2, 32, 64};
    ASSERT_EQ(Ops::Base::ToString(*dxDesc), Ops::Base::ToString(expectedShape));
    auto updateDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*updateDesc), Ops::Base::ToString(expectedShape));
}

class PReluGradUpdateInferDataTypeTest : public testing::TestWithParam<ge::DataType> {};

TEST_P(PReluGradUpdateInferDataTypeTest, test_output_dtype_follows_grads)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("PReluGradUpdate")->infer_datatype;
    ge::DataType inputDtype = GetParam();
    ge::DataType inDtype0 = inputDtype;
    ge::DataType inDtype1 = inputDtype;
    ge::DataType inDtype2 = inputDtype;
    ge::DataType outDtype0 = ge::DT_UNDEFINED;
    ge::DataType outDtype1 = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .SetOpType("PReluGradUpdate")
                      .IrInputNum(3)
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&inDtype0, &inDtype1, &inDtype2})
                      .OutputDataTypes({&outDtype0, &outDtype1})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
    // dx/update dtype = grads(input 0) dtype, 对齐 canndev
    EXPECT_EQ(context->GetOutputDataType(0), inputDtype);
    EXPECT_EQ(context->GetOutputDataType(1), inputDtype);
}

INSTANTIATE_TEST_SUITE_P(PReluGradUpdateInferDataTypeSuite, PReluGradUpdateInferDataTypeTest,
                         testing::Values(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16));
