/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_instance_norm_grad_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "../../../op_graph/instance_norm_grad_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "platform/platform_info.h"

class InstanceNormGradTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InstanceNormGrad Proto Test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "InstanceNormGrad Proto Test TearDown" << std::endl; }
};

TEST_F(InstanceNormGradTest, instance_norm_grad_infer_shape_test1)
{
    ge::op::InstanceNormGrad op;

    ge::DataType dtype = ge::DT_FLOAT;
    ge::Format format = ge::FORMAT_ND;

    auto x_tensor = create_desc_with_ori({2, 1, 2, 3, 8}, dtype, format, {2, 1, 2, 3, 8}, format);
    auto var_tensor = create_desc_with_ori({2, 1, 1, 1, 8}, dtype, format, {2, 1, 1, 1, 8}, format);
    auto gamma_tensor = create_desc_with_ori({8}, dtype, format, {8}, format);

    op.UpdateInputDesc("dy", x_tensor);
    op.UpdateInputDesc("x", x_tensor);
    op.UpdateInputDesc("variance", var_tensor);
    op.UpdateInputDesc("mean", var_tensor);
    op.UpdateInputDesc("gamma", gamma_tensor);

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto pdx_desc = op.GetOutputDescByName("pd_x");
    auto pdgamma_desc = op.GetOutputDescByName("pd_gamma");
    auto pdbeta_desc = op.GetOutputDescByName("pd_beta");
    EXPECT_EQ(pdx_desc.GetShape().GetDimNum(), 5);
    EXPECT_EQ(pdgamma_desc.GetShape().GetDimNum(), 1);
    EXPECT_EQ(pdbeta_desc.GetShape().GetDimNum(), 1);
}

TEST_F(InstanceNormGradTest, instance_norm_grad_infer_data_type)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("InstanceNormGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("InstanceNormGrad")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_dtype = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(5)
                              .NodeIoNum(5, 3)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_dtype, &input_dtype, &input_dtype, &input_dtype, &input_dtype})
                              .OutputDataTypes({&input_dtype, &input_dtype, &input_dtype})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetOutputDataType(0), input_dtype);
    EXPECT_EQ(context->GetOutputDataType(1), input_dtype);
    EXPECT_EQ(context->GetOutputDataType(2), input_dtype);
}
