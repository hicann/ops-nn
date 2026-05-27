/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/anti_mx_quant_proto.h"

namespace {
class AntiMxQuant : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AntiMxQuant InferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AntiMxQuant InferShape TearDown" << std::endl;
    }
};

// x shape [2048, 2360], axis=-1, blockSize=32
// mxscale shape: axis_dim = ceil(2360/32) = 74, then (74+1)/2 = 37 (rounded up to pair)
// Wait, per the formula: ceil(x[axis]/blocksize) = ceil(2360/32) = 74, then (74+2-1)/2 = 75/2 = 37 (integer division)
// So mxscale shape = [2048, 37, 2]
TEST_F(AntiMxQuant, AntiMxQuant_infershape_case_0)
{
    ge::op::AntiMxQuant op;
    op.UpdateInputDesc("x", create_desc({2048, 2360}, ge::DT_FLOAT8_E5M2));
    op.UpdateInputDesc("mxscale", create_desc({2048, 37, 2}, ge::DT_FLOAT8_E8M0));
    op.SetAttr("axis", -1);
    op.SetAttr("dst_type", 1);  // FP16
    Runtime2TestParam param{{"axis", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    std::vector<int64_t> expectedYShape = {2048, 2360};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

// x shape [2048, 2370], axis=-1
// ceil(2370/32) = 75, (75+1)/2 = 38
// mxscale shape = [2048, 38, 2]
TEST_F(AntiMxQuant, AntiMxQuant_infershape_case_1)
{
    ge::op::AntiMxQuant op;
    op.UpdateInputDesc("x", create_desc({2048, 2370}, ge::DT_FLOAT8_E4M3FN));
    op.UpdateInputDesc("mxscale", create_desc({2048, 38, 2}, ge::DT_FLOAT8_E8M0));
    op.SetAttr("axis", -1);
    op.SetAttr("dst_type", 27);  // BF16
    Runtime2TestParam param{{"axis", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    std::vector<int64_t> expectedYShape = {2048, 2370};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

// Example from requirement: x=[1440, 2, 84], axis=-1
// ceil(84/32) = 3, (3+1)/2 = 2
// mxscale shape = [1440, 2, 2, 2]
TEST_F(AntiMxQuant, AntiMxQuant_infershape_case_2)
{
    ge::op::AntiMxQuant op;
    op.UpdateInputDesc("x", create_desc({1440, 2, 84}, ge::DT_FLOAT4_E2M1));
    op.UpdateInputDesc("mxscale", create_desc({1440, 2, 2, 2}, ge::DT_FLOAT8_E8M0));
    op.SetAttr("axis", -1);
    op.SetAttr("dst_type", 0);  // FP32
    Runtime2TestParam param{{"axis", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    std::vector<int64_t> expectedYShape = {1440, 2, 84};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

// Unknown dim case
TEST_F(AntiMxQuant, AntiMxQuant_infershape_case_unknow_dim)
{
    ge::op::AntiMxQuant op;
    op.UpdateInputDesc("x", create_desc({2048, -1}, ge::DT_FLOAT8_E5M2));
    op.UpdateInputDesc("mxscale", create_desc({2048, -1, 2}, ge::DT_FLOAT8_E8M0));
    op.SetAttr("axis", -1);
    op.SetAttr("dst_type", 1);
    Runtime2TestParam param{{"axis", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    std::vector<int64_t> expectedYShape = {2048, -1};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

// Error: rank exceeds 7
TEST_F(AntiMxQuant, AntiMxQuant_infershape_error_rank)
{
    ge::op::AntiMxQuant op;
    op.UpdateInputDesc("x", create_desc({1, 1, 1, 1, 1, 1, 1, 1}, ge::DT_FLOAT8_E5M2));
    op.UpdateInputDesc("mxscale", create_desc({1, 1, 1, 1, 1, 1, 1, 1, 2}, ge::DT_FLOAT8_E8M0));
    op.SetAttr("axis", -1);
    op.SetAttr("dst_type", 1);
    Runtime2TestParam param{{"axis", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

// Unknown rank case
TEST_F(AntiMxQuant, AntiMxQuant_infershape_case_unknow_rank)
{
    ge::op::AntiMxQuant op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT8_E5M2));
    op.UpdateInputDesc("mxscale", create_desc({-2}, ge::DT_FLOAT8_E8M0));
    op.SetAttr("axis", -1);
    op.SetAttr("dst_type", 1);
    Runtime2TestParam param{{"axis", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    std::vector<int64_t> expectedYShape = {-2};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

TEST_F(AntiMxQuant, AntiMxQuant_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AntiMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AntiMxQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT8_E5M2;
        ge::DataType input_mxscale_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType output_y_ref = ge::DT_FLOAT16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 1)
                                  .NodeInputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                       {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                                  .InputDataTypes({&input_x_ref, &input_mxscale_ref})
                                  .OutputDataTypes({&output_y_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_x_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_mxscale_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
    }
}

TEST_F(AntiMxQuant, AntiMxQuant_InferDtype_case_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AntiMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AntiMxQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT8_E5M2;
        ge::DataType input_mxscale_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType output_y_ref = ge::DT_FLOAT16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 1)
                                  .NodeInputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                       {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}})  // dst_type mismatch
                                  .InputDataTypes({&input_x_ref, &input_mxscale_ref})
                                  .OutputDataTypes({&output_y_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

} // namespace
