/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_inplace_index_fill_infershape.cpp
 * \brief
 */

#include "ut_op_util.h"
#include "infershape_test_util.h"
#include <iostream>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/inplace_index_fill_proto.h"
#include "ut_op_common.h"

class InplaceIndexFill : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InplaceIndexFill SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "InplaceIndexFill TearDown" << std::endl; }
};

TEST_F(InplaceIndexFill, InplaceIndexFill_infershape_case_1)
{
    ge::op::InplaceIndexFill op;
    op.UpdateInputDesc("x", create_desc({2, 3}, ge::DT_INT32));
    op.UpdateInputDesc("indices", create_desc({2, 3}, ge::DT_INT64));
    op.UpdateInputDesc("value", create_desc({2, 3}, ge::DT_INT32));
    op.SetAttr("dim", 1);
    Runtime2TestParam param{{"dim"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputX = op.GetOutputDesc(0);
    std::vector<int64_t> expectedXShape = {2, 3};
    EXPECT_EQ(outputX.GetShape().GetDims(), expectedXShape);
}

TEST_F(InplaceIndexFill, InplaceIndexFill_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceIndexFill"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceIndexFill")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_INT32;
        ge::DataType input_indices_ref = ge::DT_INT64;
        ge::DataType input_value_ref = ge::DT_INT32;
        ge::DataType output_x_ref = ge::DT_INT32;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 1)
                                  .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                                  .InputDataTypes({&input_x_ref, &input_indices_ref, &input_value_ref})
                                  .OutputDataTypes({&output_x_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_x_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_indices_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_value_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_x_ref);
    }
}
