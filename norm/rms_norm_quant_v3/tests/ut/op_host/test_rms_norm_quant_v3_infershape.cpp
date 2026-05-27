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
#include "../../../op_graph/rms_norm_quant_v3_proto.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

class RmsNormQuantV3Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RmsNormQuantV3InferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RmsNormQuantV3InferShapeTest TearDown" << std::endl;
    }
};

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_shape_case1)
{
    ge::op::RmsNormQuantV3 op;
    op.UpdateInputDesc("x", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({64}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    op.SetAttr("output_rstd", false);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_rstd_shape = {};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_shape_case2)
{
    ge::op::RmsNormQuantV3 op;
    op.UpdateInputDesc("x", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({64}, ge::DT_FLOAT));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("div_mode", true);
    op.SetAttr("dst_type", 2);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "div_mode", "dst_type", "output_rstd"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_rstd_shape = {8, 1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_dtype_output_rstd_true)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormQuantV3"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormQuantV3")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType y_ref = ge::DT_INT8;
        ge::DataType rstd_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(3)
                .NodeIoNum(3, 3)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)}})
                .NodeAttrs({{"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                .NodeAttrs({{"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                .InputDataTypes({&input_ref, &input_ref, &input_ref})
                .OutputDataTypes({&y_ref, &y_ref, &rstd_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), y_ref);
        EXPECT_EQ(context->GetOutputDataType(2), rstd_ref);
    }
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_dtype_output_rstd_false)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormQuantV3"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormQuantV3")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType y_ref = ge::DT_INT8;
        ge::DataType rstd_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(3)
                .NodeIoNum(3, 3)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)}})
                .NodeAttrs({{"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                .NodeAttrs({{"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                .InputDataTypes({&input_ref, &input_ref, &input_ref})
                .OutputDataTypes({&y_ref, &y_ref, &rstd_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), y_ref);
        // When output_rstd is false, rstd dtype should NOT be set
        EXPECT_NE(context->GetOutputDataType(2), rstd_ref);
    }
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_shape_m_not0_n_0_true)
{
    ge::op::RmsNormQuantV3 op;
    op.UpdateInputDesc("x", create_desc({176, 0}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({0}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({0}, ge::DT_FLOAT));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("div_mode", true);
    op.SetAttr("dst_type", 2);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "div_mode", "dst_type", "output_rstd"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {176, 0};
    std::vector<int64_t> expected_rstd_shape = {176, 1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_shape_m_0_n_not0_true)
{
    ge::op::RmsNormQuantV3 op;
    op.UpdateInputDesc("x", create_desc({0, 100}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({100}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({100}, ge::DT_FLOAT));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("div_mode", true);
    op.SetAttr("dst_type", 2);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "div_mode", "dst_type", "output_rstd"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {0, 100};
    std::vector<int64_t> expected_rstd_shape = {0, 1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_m_0_n_0_true)
{
    ge::op::RmsNormQuantV3 op;
    op.UpdateInputDesc("x", create_desc({0, 0}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({0}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({0}, ge::DT_FLOAT));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("div_mode", true);
    op.SetAttr("dst_type", 2);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "div_mode", "dst_type", "output_rstd"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {0, 0};
    std::vector<int64_t> expected_rstd_shape = {0, 1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}
