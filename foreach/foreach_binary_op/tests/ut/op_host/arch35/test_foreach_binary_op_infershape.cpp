/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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
#include "../../../../op_graph/foreach_binary_op_proto.h"

// ForeachBinaryOp infershape/infer_datatype are registered via the shared
// foreach/foreach_utils/op_host/foreach_infershape.cpp (InferShape4ForeachCommon /
// InferDataType4ForeachCommon): y[i] shape/dtype == x1[i] shape/dtype.
class ForeachBinaryOpInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ForeachBinaryOpInferShape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ForeachBinaryOpInferShape TearDown" << std::endl; }
};

TEST_F(ForeachBinaryOpInferShapeTest, infer_shape_known_success)
{
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachBinaryOp")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);
    // x1 (3 tensors)
    gert::StorageShape x1_shape_0 = {{1}, {}};
    gert::StorageShape x1_shape_1 = {{2}, {}};
    gert::StorageShape x1_shape_2 = {{3}, {}};
    // x2 (3 tensors)
    gert::StorageShape x2_shape_0 = {{1}, {}};
    gert::StorageShape x2_shape_1 = {{2}, {}};
    gert::StorageShape x2_shape_2 = {{3}, {}};

    gert::StorageShape y_shape_0 = {{}, {}};
    gert::StorageShape y_shape_1 = {{}, {}};
    gert::StorageShape y_shape_2 = {{}, {}};

    std::vector<void*> input_shape_ref(6);
    input_shape_ref[0] = &x1_shape_0;
    input_shape_ref[1] = &x1_shape_1;
    input_shape_ref[2] = &x1_shape_2;
    input_shape_ref[3] = &x2_shape_0;
    input_shape_ref[4] = &x2_shape_1;
    input_shape_ref[5] = &x2_shape_2;

    std::vector<void*> output_shape_ref(3);
    output_shape_ref[0] = &y_shape_0;
    output_shape_ref[1] = &y_shape_1;
    output_shape_ref[2] = &y_shape_2;

    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({3, 3}, {3})
                      .InputShapes(input_shape_ref)
                      .OutputShapes(output_shape_ref)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[1]");
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShape(1)), "[2]");
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShape(2)), "[3]");
}

TEST_F(ForeachBinaryOpInferShapeTest, infer_dtype_float16)
{
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachBinaryOp")->infer_datatype;
    ASSERT_NE(infer_datatype_func, nullptr);
    ge::DataType x1_dtype_0 = ge::DT_FLOAT16;
    ge::DataType x1_dtype_1 = ge::DT_FLOAT16;
    ge::DataType x2_dtype_0 = ge::DT_FLOAT16;
    ge::DataType x2_dtype_1 = ge::DT_FLOAT16;

    std::vector<void*> input_dtype_ref(4);
    input_dtype_ref[0] = &x1_dtype_0;
    input_dtype_ref[1] = &x1_dtype_1;
    input_dtype_ref[2] = &x2_dtype_0;
    input_dtype_ref[3] = &x2_dtype_1;

    std::vector<void*> output_dtype_ref(2);

    auto holder = gert::InferDataTypeContextFaker()
                      .IrInstanceNum({2, 2}, {2})
                      .InputDataTypes(input_dtype_ref)
                      .OutputDataTypes(output_dtype_ref)
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT16);
}

TEST_F(ForeachBinaryOpInferShapeTest, infer_dtype_bf16)
{
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachBinaryOp")->infer_datatype;
    ASSERT_NE(infer_datatype_func, nullptr);
    ge::DataType x1_dtype_0 = ge::DT_BF16;
    ge::DataType x2_dtype_0 = ge::DT_BF16;

    std::vector<void*> input_dtype_ref(2);
    input_dtype_ref[0] = &x1_dtype_0;
    input_dtype_ref[1] = &x2_dtype_0;

    std::vector<void*> output_dtype_ref(1);

    auto holder = gert::InferDataTypeContextFaker()
                      .IrInstanceNum({1, 1}, {1})
                      .InputDataTypes(input_dtype_ref)
                      .OutputDataTypes(output_dtype_ref)
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}
