/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

class ApplyAdagradTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyAdagrad SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyAdagrad TearDown" << std::endl; }
};

static void RunInferShapeCase(gert::StorageShape& var, gert::StorageShape& lr, ge::DataType dtype,
                              const gert::Shape& expectedShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdagrad")->infer_shape;
    gert::StorageShape varShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({4, 1})
                      .InputShapes({&var, &var, &lr, &var})
                      .OutputShapes({&varShape})
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"update_slots", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expectedShape));
}

static void RunInferDataTypeCase(ge::DataType dtype)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdagrad")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    ge::DataType inputDtype = dtype;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&inputDtype, &inputDtype, &inputDtype, &inputDtype})
                      .OutputDataTypes({&outputDtype})
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), dtype);
}

TEST_F(ApplyAdagradTest, apply_adagrad_infer_shape_fp16_1d)
{
    gert::StorageShape var = {{64}, {-1}};
    gert::StorageShape lr = {{1}, {-1}};
    gert::Shape expectedOutputShape = {64};
    RunInferShapeCase(var, lr, ge::DT_FLOAT16, expectedOutputShape);
}

TEST_F(ApplyAdagradTest, apply_adagrad_infer_shape_bf16_2d)
{
    gert::StorageShape var = {{8, 16}, {-1, -1}};
    gert::StorageShape lr = {{1}, {-1}};
    gert::Shape expectedOutputShape = {8, 16};
    RunInferShapeCase(var, lr, ge::DT_BF16, expectedOutputShape);
}

TEST_F(ApplyAdagradTest, apply_adagrad_infer_shape_scalar)
{
    gert::StorageShape var = {{}, {}};
    gert::StorageShape lr = {{}, {}};
    gert::Shape expectedOutputShape = {};
    RunInferShapeCase(var, lr, ge::DT_FLOAT, expectedOutputShape);
}

TEST_F(ApplyAdagradTest, apply_adagrad_infer_shape_unknown_rank)
{
    gert::StorageShape var = {{-2}, {-2}};
    gert::StorageShape lr = {{1}, {-1}};
    gert::Shape expectedOutputShape = {-2};
    RunInferShapeCase(var, lr, ge::DT_FLOAT, expectedOutputShape);
}

TEST_F(ApplyAdagradTest, apply_adagrad_infer_datatype_follow_var)
{
    RunInferDataTypeCase(ge::DT_FLOAT);
    RunInferDataTypeCase(ge::DT_FLOAT16);
    RunInferDataTypeCase(ge::DT_BF16);
}
