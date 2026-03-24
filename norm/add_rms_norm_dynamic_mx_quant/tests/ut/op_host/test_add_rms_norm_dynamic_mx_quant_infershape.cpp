/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "../../../op_graph/add_rms_norm_dynamic_mx_quant_proto.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

class AddRmsNormDynamicMxQuantTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddRmsNormDynamicMxQuantInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddRmsNormDynamicMxQuantInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(AddRmsNormDynamicMxQuantTest, AddRmsNormDynamicMxQuant_infer_shape_case1)
{
    ge::op::AddRmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x1", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({64}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_x_desc = op.GetOutputDesc(1);
    auto output_mxscale_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_mxscale_shape = {8, 1, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(AddRmsNormDynamicMxQuantTest, AddRmsNormDynamicMxQuant_infer_shape_case2)
{
    ge::op::AddRmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x1", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({64}, ge::DT_FLOAT16));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("scale_alg", 0);
    op.SetAttr("round_mode", "rint");
    op.SetAttr("dst_type", 40);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "scale_alg", "round_mode", "dst_type", "output_rstd"},{},{}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_x_desc = op.GetOutputDesc(1);
    auto output_mxscale_desc = op.GetOutputDesc(2);
    auto output_rstd_desc = op.GetOutputDesc(3);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_mxscale_shape = {8, 1, 2};
    std::vector<int64_t> expected_rstd_shape = {8, 1};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(AddRmsNormDynamicMxQuantTest, AddRmsNormDynamicMxQuant_infer_dtype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicMxQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType y_ref = ge::DT_FLOAT8_E5M2;
        ge::DataType mx_scale_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType rstd_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(4)
                .NodeIoNum(4, 4)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)}})
                .NodeAttrs({{"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")}})
                .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}})
                .NodeAttrs({{"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref})
                .OutputDataTypes({&y_ref, &input_ref, &mx_scale_ref, &rstd_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), input_ref);
        EXPECT_EQ(context->GetOutputDataType(2), mx_scale_ref);
        EXPECT_EQ(context->GetOutputDataType(3), rstd_ref);
    }
}
