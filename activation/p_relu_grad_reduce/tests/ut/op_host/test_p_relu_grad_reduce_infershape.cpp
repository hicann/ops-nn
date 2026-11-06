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

// PReluGradReduce: inputs(grads 0, features 1, weights 2, updates 3) -> output(da 0)
// da shape = weights shape; da dtype = grads dtype (canndev InferDtype4SameInput)
class PReluGradReduceInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PReluGradReduceInferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "PReluGradReduceInferShapeTest TearDown" << std::endl; }
};

TEST_F(PReluGradReduceInferShapeTest, test_da_shape_follows_weights)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("PReluGradReduce")->infer_shape;
    gert::StorageShape gradsShape = {{32, 64}, {32, 64}};
    gert::StorageShape featuresShape = {{32, 64}, {32, 64}};
    gert::StorageShape weightsShape = {{64}, {64}};
    gert::StorageShape updatesShape = {{32, 64}, {32, 64}};
    gert::StorageShape daShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&gradsShape, &featuresShape, &weightsShape, &updatesShape})
                      .OutputShapes({&daShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedShape = {64};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedShape));
}

TEST_F(PReluGradReduceInferShapeTest, test_da_shape_weights_2d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("PReluGradReduce")->infer_shape;
    gert::StorageShape gradsShape = {{16, 128}, {16, 128}};
    gert::StorageShape featuresShape = {{16, 128}, {16, 128}};
    gert::StorageShape weightsShape = {{3, 128}, {3, 128}};
    gert::StorageShape updatesShape = {{16, 128}, {16, 128}};
    gert::StorageShape daShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&gradsShape, &featuresShape, &weightsShape, &updatesShape})
                      .OutputShapes({&daShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedShape = {3, 128};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedShape));
}

class PReluGradReduceInferDataTypeTest : public testing::TestWithParam<ge::DataType> {};

TEST_P(PReluGradReduceInferDataTypeTest, test_da_dtype_follows_grads)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("PReluGradReduce")->infer_datatype;
    ge::DataType inputDtype = GetParam();
    ge::DataType inDtype0 = inputDtype;
    ge::DataType inDtype1 = inputDtype;
    ge::DataType inDtype2 = inputDtype;
    ge::DataType inDtype3 = inputDtype;
    ge::DataType outDtype0 = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .SetOpType("PReluGradReduce")
                      .IrInputNum(4)
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&inDtype0, &inDtype1, &inDtype2, &inDtype3})
                      .OutputDataTypes({&outDtype0})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
    // da dtype = grads(input 0) dtype, 对齐 canndev InferDtype4SameInput
    EXPECT_EQ(context->GetOutputDataType(0), inputDtype);
}

INSTANTIATE_TEST_SUITE_P(PReluGradReduceInferDataTypeSuite, PReluGradReduceInferDataTypeTest,
                         testing::Values(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16));
