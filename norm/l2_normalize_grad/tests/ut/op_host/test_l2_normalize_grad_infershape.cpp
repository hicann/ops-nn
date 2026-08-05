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
 * \file test_l2_normalize_grad_infershape.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "../../../op_graph/l2_normalize_grad_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

class L2NormalizeGradTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "L2NormalizeGrad Proto Test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "L2NormalizeGrad Proto Test TearDown" << std::endl; }
};

TEST_F(L2NormalizeGradTest, l2_normalize_grad_infer_shape_test1)
{
    ge::op::L2NormalizeGrad op;

    ge::DataType dtype = ge::DT_FLOAT;
    ge::Format format = ge::FORMAT_ND;

    auto input_tensor = create_desc_with_ori({32, 512}, dtype, format, {32, 512}, format);
    op.UpdateInputDesc("x", input_tensor);
    op.UpdateInputDesc("y", input_tensor);
    op.UpdateInputDesc("dy", input_tensor);

    op.SetAttr("dim", std::vector<int64_t>{1});
    op.SetAttr("eps", 1e-4f);
    Runtime2TestParam param{{"dim", "eps"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDescByName("dx");
    EXPECT_EQ(output_desc.GetShape().GetDimNum(), 2);
    EXPECT_EQ(output_desc.GetShape().GetDim(0), 32);
    EXPECT_EQ(output_desc.GetShape().GetDim(1), 512);
}

TEST_F(L2NormalizeGradTest, l2_normalize_grad_infer_data_type)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("L2NormalizeGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("L2NormalizeGrad")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_x = ge::DT_FLOAT;
    ge::DataType dx_datatype = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
                                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-4f)}})
                              .InputDataTypes({&input_x, &input_x, &input_x})
                              .OutputDataTypes({&dx_datatype})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_x);
    EXPECT_EQ(context->GetOutputDataType(0), dx_datatype);
}

// -2 UNKNOWN_RANK：红线 R4 点名的必验项。dx 是 x 的整体拷贝，unknown-rank 应原样透传。
TEST_F(L2NormalizeGradTest, l2_normalize_grad_infer_shape_unknown_rank)
{
    ge::op::L2NormalizeGrad op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("y", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("dy", create_desc({-2}, ge::DT_FLOAT));
    op.SetAttr("dim", std::vector<int64_t>{1});
    op.SetAttr("eps", 1e-4f);
    Runtime2TestParam param{{"dim", "eps"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    std::vector<int64_t> expected{-2};
    EXPECT_EQ(op.GetOutputDescByName("dx").GetShape().GetDims(), expected);
}
