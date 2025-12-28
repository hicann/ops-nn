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
 * \file test_index_fill_d_infershape.cpp
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
#include "../../../op_graph/index_fill_d_proto.h"
#include "ut_op_common.h"

class IndexFillD : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "IndexFillD SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "IndexFillD TearDown" << std::endl;
    }
};

TEST_F(IndexFillD, IndexFillD_infershape_case_1)
{
    ge::op::IndexFillD op;
    op.UpdateInputDesc("x", create_desc({1000, 1000}, ge::DT_INT64));
    op.UpdateInputDesc("assist1", create_desc({1000, 1000}, ge::DT_INT64));
    op.UpdateInputDesc("assist2", create_desc({1000, 1000}, ge::DT_INT64));
    op.SetAttr("dim", 0);
    Runtime2TestParam param{{"dim"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    std::vector<int64_t> expectedYShape = {1000, 1000};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

TEST_F(IndexFillD, IndexFillD_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("IndexFillD"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexFillD")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_INT64;
        ge::DataType input_assist1_ref = ge::DT_INT64;
        ge::DataType input_assist2_ref = ge::DT_INT64;
        ge::DataType output_y_ref = ge::DT_INT64;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 1)
                                  .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                                  .InputDataTypes({&input_x_ref, &input_assist1_ref, &input_assist2_ref})
                                  .OutputDataTypes({&output_y_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_x_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_assist1_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_assist2_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
    }
}