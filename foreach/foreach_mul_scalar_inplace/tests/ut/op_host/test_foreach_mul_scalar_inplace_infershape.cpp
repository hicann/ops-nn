/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "../../../op_graph/foreach_mul_scalar_inplace_proto.h"

class ForeachMulScalarInplaceTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(ForeachMulScalarInplaceTest, infer_shape_success)
{
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachMulScalarInplace")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);
    gert::StorageShape x_shape_0 = {{8}, {}};
    gert::StorageShape x_shape_1 = {{9}, {}};
    gert::StorageShape x_shape_2 = {{10}, {}};
    gert::StorageShape x_shape_3 = {{11}, {}};
    gert::StorageShape y_shape_0 = {{}, {}};
    gert::StorageShape y_shape_1 = {{}, {}};
    gert::StorageShape y_shape_2 = {{}, {}};
    std::vector<void*> input_shape_ref(4);
    input_shape_ref[0] = &x_shape_0;
    input_shape_ref[1] = &x_shape_1;
    input_shape_ref[2] = &x_shape_2;
    input_shape_ref[3] = &x_shape_3;
    std::vector<void*> output_shape_ref(3);
    output_shape_ref[0] = &y_shape_0;
    output_shape_ref[1] = &y_shape_1;
    output_shape_ref[2] = &y_shape_2;
    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({3, 1}, {3})
                      .InputShapes(input_shape_ref)
                      .OutputShapes(output_shape_ref)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
}

TEST_F(ForeachMulScalarInplaceTest, infer_datatype_success)
{
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachMulScalarInplace")->infer_datatype;
    ASSERT_NE(infer_datatype_func, nullptr);
    ge::DataType x_dt_0 = ge::DT_FLOAT16;
    ge::DataType x_dt_1 = ge::DT_FLOAT16;
    ge::DataType x_dt_2 = ge::DT_FLOAT16;
    std::vector<void*> input_dt_ref(3);
    input_dt_ref[0] = &x_dt_0;
    input_dt_ref[1] = &x_dt_1;
    input_dt_ref[2] = &x_dt_2;
    std::vector<void*> output_dt_ref(3);
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInstanceNum({3}, {3})
                      .InputDataTypes(input_dt_ref)
                      .OutputDataTypes(output_dt_ref)
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
}
