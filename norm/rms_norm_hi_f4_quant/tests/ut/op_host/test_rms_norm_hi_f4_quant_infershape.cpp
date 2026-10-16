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
#include "../../../op_graph/rms_norm_hi_f4_quant_proto.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../../common/inc/op_host/op_const_def_nn.h"

using namespace std;

class RmsNormHiF4QuantTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RmsNormHiF4QuantInferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "RmsNormHiF4QuantInferShapeTest TearDown" << std::endl; }
};

TEST_F(RmsNormHiF4QuantTest, RmsNormHiF4Quant_infer_shape_fp16_fp16)
{
    ge::op::RmsNormHiF4Quant op;
    op.UpdateInputDesc("x", create_desc({8, 64}, Ops::NN::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, Ops::NN::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({64}, Ops::NN::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_scale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_scale_shape = {8, 1, 1};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_scale_desc.GetShape().GetDims(), expected_scale_shape);
}

TEST_F(RmsNormHiF4QuantTest, RmsNormHiF4Quant_infer_shape_fp16_fp32_rstd)
{
    ge::op::RmsNormHiF4Quant op;
    op.UpdateInputDesc("x", create_desc({8, 128}, Ops::NN::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({128}, Ops::NN::DT_FLOAT));
    op.UpdateInputDesc("beta", create_desc({128}, Ops::NN::DT_FLOAT));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "output_rstd"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_scale_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {8, 128};
    std::vector<int64_t> expected_scale_shape = {8, 2, 1};
    std::vector<int64_t> expected_rstd_shape = {8, 1};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_scale_desc.GetShape().GetDims(), expected_scale_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNormHiF4QuantTest, RmsNormHiF4Quant_infer_shape_bf16_bf16)
{
    ge::op::RmsNormHiF4Quant op;
    op.UpdateInputDesc("x", create_desc({4, 128}, Ops::NN::DT_BF16));
    op.UpdateInputDesc("gamma", create_desc({128}, Ops::NN::DT_BF16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_scale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_scale_shape = {4, 2, 1};
    EXPECT_EQ(output_scale_desc.GetShape().GetDims(), expected_scale_shape);
}

TEST_F(RmsNormHiF4QuantTest, RmsNormHiF4Quant_infer_dtype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormHiF4Quant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormHiF4Quant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = Ops::NN::DT_FLOAT16;
        ge::DataType gamma_ref = Ops::NN::DT_FLOAT;
        ge::DataType y_ref = Ops::NN::DT_HIFLOAT4;
        ge::DataType scale_ref = Ops::NN::DT_HIFLOAT4_SCALE;
        ge::DataType rstd_ref = Ops::NN::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 3)
                                  .NodeInputTd(0, Ops::NN::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, Ops::NN::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, Ops::NN::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, Ops::NN::DT_HIFLOAT4, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, Ops::NN::DT_HIFLOAT4_SCALE, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, Ops::NN::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)}})
                                  .NodeAttrs({{"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                                  .InputDataTypes({&input_ref, &gamma_ref, &gamma_ref})
                                  .OutputDataTypes({&y_ref, &scale_ref, &rstd_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), scale_ref);
        EXPECT_EQ(context->GetOutputDataType(2), rstd_ref);
    }
}
