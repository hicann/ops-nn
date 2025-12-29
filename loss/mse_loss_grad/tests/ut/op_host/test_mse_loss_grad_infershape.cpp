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
 * \file test_mse_loss_grad_proto.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "../../../op_graph/mse_loss_grad_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

class MseLossGradTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MseLossGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MseLossGrad Proto Test TearDown" << std::endl;
    }
};

TEST_F(MseLossGradTest, mse_loss_grad_infer_shape_test1) {
  ge::op::MseLossGrad op;

  ge::DataType dtype = ge::DT_FLOAT;
  ge::Format format = ge::FORMAT_ND;
  
  auto input_tensor = create_desc_with_ori({182,4}, dtype, format, {182,4}, format);
  
  op.UpdateInputDesc("predict", input_tensor);
  op.UpdateInputDesc("label", input_tensor);
  op.UpdateInputDesc("dout", input_tensor);

  op.SetAttr("reduction", "mean");
  Runtime2TestParam param{{"reduction"}};
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");

  EXPECT_EQ(output_desc.GetShape().GetDimNum(), 2);
}

TEST_F(MseLossGradTest, mse_loss_grad_infer_data_type)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGrad")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_x = ge::DT_FLOAT;
    ge::DataType y_datatype = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                              .InputDataTypes({&input_x, &input_x, &input_x})
                              .OutputDataTypes({&y_datatype})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_x);
    EXPECT_EQ(context->GetOutputDataType(0), y_datatype);
}
