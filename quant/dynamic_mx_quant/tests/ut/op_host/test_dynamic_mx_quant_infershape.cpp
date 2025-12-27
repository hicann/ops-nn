/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../../../op_graph/dynamic_mx_quant_proto.h"

namespace {
class DynamicMxQuant : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DynamicMxQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DynamicMxQuantTiling TearDown" << std::endl;
    }
};

TEST_F(DynamicMxQuant, DynamicMxQuant_infershape_case_0)
{
    ge::op::DynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({2048, 2360}, ge::DT_FLOAT16));
    op.SetAttr("axis", -1);
    op.SetAttr("round_mode", "rint");
    op.SetAttr("dst_type", 40);
    op.SetAttr("blocksize", 32);
    Runtime2TestParam param{{"axis", "round_mode", "dst_type", "blocksize"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {2048, 2360};
    std::vector<int64_t> expectedScaleShape = {2048, 37, 2};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(DynamicMxQuant, DynamicMxQuant_infershape_case_1)
{
    ge::op::DynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({2048, 2370}, ge::DT_FLOAT16));
    op.SetAttr("axis", -1);
    op.SetAttr("round_mode", "rint");
    op.SetAttr("dst_type", 40);
    op.SetAttr("blocksize", 32);
    Runtime2TestParam param{{"axis", "round_mode", "dst_type", "blocksize"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {2048, 2370};
    std::vector<int64_t> expectedScaleShape = {2048, 38, 2};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(DynamicMxQuant, DynamicMxQuant_infershape_case_unknow_dim)
{
    ge::op::DynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({2048, -1}, ge::DT_FLOAT16));
    op.SetAttr("axis", -1);
    op.SetAttr("round_mode", "rint");
    op.SetAttr("dst_type", 40);
    op.SetAttr("blocksize", 32);
    Runtime2TestParam param{{"axis", "round_mode", "dst_type", "blocksize"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {2048, -1};
    std::vector<int64_t> expectedScaleShape = {2048, -1, 2};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(DynamicMxQuant, DynamicMxQuant_infershape_error_rank)
{
    ge::op::DynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({1, 1, 1, 1, 1, 1, 1, 1}, ge::DT_FLOAT16));
    op.SetAttr("axis", -1);
    op.SetAttr("round_mode", "rint");
    op.SetAttr("dst_type", 40);
    op.SetAttr("blocksize", 32);
    Runtime2TestParam param{{"axis", "round_mode", "dst_type", "blocksize"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(DynamicMxQuant, DynamicMxQuant_infershape_case_unknow_rank)
{
    ge::op::DynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT16));
    op.SetAttr("axis", -1);
    op.SetAttr("round_mode", "rint");
    op.SetAttr("dst_type", 40);
    op.SetAttr("blocksize", 32);
    Runtime2TestParam param{{"axis", "round_mode", "dst_type", "blocksize"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {-2};
    std::vector<int64_t> expectedScaleShape = {-2};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(DynamicMxQuant, DynamicMxQuant_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicMxQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType output_y_ref = ge::DT_FLOAT8_E5M2;
        ge::DataType output_scale_ref = ge::DT_FLOAT8_E8M0;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(1, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                       {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                       {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)},
                                       {"blocksize", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                                       {"global_pooling", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                                  .InputDataTypes({&input_x_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_x_ref);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}

TEST_F(DynamicMxQuant, DynamicMxQuant_InferDtype_case_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicMxQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType output_y_ref = ge::DT_FLOAT8_E5M2;
        ge::DataType output_scale_ref = ge::DT_FLOAT8_E8M0;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(1, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                       {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                       {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                       {"blocksize", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                                       {"global_pooling", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                                  .InputDataTypes({&input_x_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

} // namespace