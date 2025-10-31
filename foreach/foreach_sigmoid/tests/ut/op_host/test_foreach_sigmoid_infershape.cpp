/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @file test_foreach_sigmoid_infershape.cpp
 *
 * @brief
 *
 * @Version 2.0
 *
 */

#include "gtest/gtest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/kernel_context.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/foreach_sigmoid_proto.h"

class ForeachSigmoidTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ForeachSigmoid SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ForeachSigmoid TearDown" << std::endl;
    }
};

TEST_F(ForeachSigmoidTest, infer_shape_known_success)
{
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachSigmoid")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    // x
    gert::StorageShape x_shape_0 = {{1}, {}};
    gert::StorageShape x_shape_1 = {{2}, {}};
    gert::StorageShape x_shape_2 = {{3}, {}};

    gert::StorageShape y_shape_0 = {{}, {}};
    gert::StorageShape y_shape_1 = {{}, {}};
    gert::StorageShape y_shape_2 = {{}, {}};

    std::vector<void*> input_shape_ref(3);
    input_shape_ref[0] = &x_shape_0;
    input_shape_ref[1] = &x_shape_1;
    input_shape_ref[2] = &x_shape_2;

    std::vector<void*> output_shape_ref(3);
    output_shape_ref[0] = &y_shape_0;
    output_shape_ref[1] = &y_shape_1;
    output_shape_ref[2] = &y_shape_2;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({3, 3})
                      .InputShapes(input_shape_ref)
                      .OutputShapes(output_shape_ref)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto output_shape_0 = context->GetOutputShape(0);
    EXPECT_EQ(Ops::Base::ToString(*output_shape_0), "[1]");

    auto output_shape_1 = context->GetOutputShape(1);
    EXPECT_EQ(Ops::Base::ToString(*output_shape_1), "[2]");

    auto output_shape_2 = context->GetOutputShape(2);
    EXPECT_EQ(Ops::Base::ToString(*output_shape_2), "[3]");
}

TEST_F(ForeachSigmoidTest, infer_dtype_test_1)
{
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachSigmoid")->infer_datatype;
    ASSERT_NE(infer_datatype_func, nullptr);

    // x
    ge::DataType x_dtype_0 = ge::DT_FLOAT16;
    ge::DataType x_dtype_1 = ge::DT_FLOAT16;
    ge::DataType x_dtype_2 = ge::DT_FLOAT16;

    std::vector<void*> input_dtype_ref(3);
    input_dtype_ref[0] = &x_dtype_0;
    input_dtype_ref[1] = &x_dtype_1;
    input_dtype_ref[2] = &x_dtype_2;

    std::vector<void*> output_dtype_ref(3);

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({3, 3})
                      .InputDataTypes(input_dtype_ref)
                      .OutputDataTypes(output_dtype_ref)
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);

    ge::DataType expected_datatype = ge::DT_FLOAT16;
    auto output_dtype_0 = context->GetOutputDataType(0);
    EXPECT_EQ(output_dtype_0, expected_datatype);

    auto output_dtype_1 = context->GetOutputDataType(1);
    EXPECT_EQ(output_dtype_1, expected_datatype);

    auto output_dtype_2 = context->GetOutputDataType(2);
    EXPECT_EQ(output_dtype_2, expected_datatype);
}