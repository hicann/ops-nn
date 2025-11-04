/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "runtime/infer_shape_range_context.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "../../../../dynamic_quant_update_scatter_v2/op_graph/dynamic_quant_update_scatter_v2_proto.h"
#include "runtime/infer_shape_range_context.h"

class DynamicQuantUpdateScatterV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DynamicQuantUpdateScatterV2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DynamicQuantUpdateScatterV2 TearDown" << std::endl;
    }
};

TEST_F(DynamicQuantUpdateScatterV2Test, DynamicQuantUpdateScatterV2_infershape_case_0)
{
    ge::op::DynamicQuantUpdateScatterV2 op;
    op.UpdateInputDesc("x", create_desc({192, 1, 128}, ge::DT_FLOAT16));
    op.UpdateInputDesc("indices", create_desc({192}, ge::DT_INT32));
    op.UpdateInputDesc("var", create_desc({192, 1024, 1, 128}, ge::DT_INT4));
    op.UpdateInputDesc("var_scale", create_desc({192, 1024}, ge::DT_FLOAT));
    op.UpdateInputDesc("var_offset", create_desc({192, 1024}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    auto output_var = op.GetOutputDesc(0);
    auto output_var_scale = op.GetOutputDesc(1);
    auto output_var_offset = op.GetOutputDesc(2);
    std::vector<int64_t> expected_var_shape = {192, 1024, 128};
    std::vector<int64_t> expected_var_scale_shape = {1, 192, 1024};
    std::vector<int64_t> expected_var_offset_shape = {1, 192, 1024};
    EXPECT_EQ(output_var.GetShape().GetDims(), expected_var_shape);
    EXPECT_EQ(output_var_scale.GetShape().GetDims(), expected_var_scale_shape);
    EXPECT_EQ(output_var_offset.GetShape().GetDims(), expected_var_offset_shape);
}

TEST_F(DynamicQuantUpdateScatterV2Test, DynamicQuantUpdateScatterV2_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantUpdateScatterV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantUpdateScatterV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x = ge::DT_FLOAT16;
        ge::DataType input_indices = ge::DT_INT32;
        ge::DataType input_var = ge::DT_INT4;
        ge::DataType input_var_scale = ge::DT_FLOAT;
        ge::DataType input_var_offset = ge::DT_FLOAT;
        ge::DataType output_var = ge::DT_INT4;
        ge::DataType output_var_scale = ge::DT_FLOAT;
        ge::DataType output_var_offset = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(5)
                .NodeIoNum(5, 3)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_x, &input_indices, &input_var, &input_var_scale, &input_var_offset})
                .OutputDataTypes({&output_var, &output_var_scale, &output_var_offset})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_x);
        EXPECT_EQ(context->GetInputDataType(1), input_indices);
        EXPECT_EQ(context->GetInputDataType(2), input_var);
        EXPECT_EQ(context->GetInputDataType(3), input_var_scale);
        EXPECT_EQ(context->GetInputDataType(4), input_var_offset);

        EXPECT_EQ(context->GetOutputDataType(0), output_var);
        EXPECT_EQ(context->GetOutputDataType(1), output_var_scale);
        EXPECT_EQ(context->GetOutputDataType(2), output_var_offset);
    }
}
