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
#include "gtest/gtest.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/flat_quant_proto.h"

using namespace ge;
using namespace op;

class FlatQuantProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlatQuantProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlatQuantProto TearDown" << std::endl;
    }
};

TEST_F(FlatQuantProto, inferDtype_case_1)
{
    // mock Ascend910B
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_p1_ref = ge::DT_FLOAT16;
        ge::DataType input_p2_ref = ge::DT_FLOAT16;
        ge::DataType output_out_ref = ge::DT_INT4;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(3, 2)
                                  .NodeAttrs(
                                      {{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                       {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(29L)}})
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_x_ref, &input_p1_ref, &input_p2_ref})
                                  .OutputDataTypes({&output_out_ref, &output_scale_ref})
                                  .Build();
 
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_out_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}

TEST_F(FlatQuantProto, inferDtype_950_case_1)
{
    // mock Ascend950
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950PR_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_p1_ref = ge::DT_FLOAT16;
        ge::DataType input_p2_ref = ge::DT_FLOAT16;
        ge::DataType output_out_ref = ge::DT_FLOAT4_E2M1;
        ge::DataType output_scale_ref = ge::DT_FLOAT8_E8M0;
        // MxFP4
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(3, 2)
                                  .NodeAttrs(
                                      {{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                       {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(40L)}})
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_x_ref, &input_p1_ref, &input_p2_ref})
                                  .OutputDataTypes({&output_out_ref, &output_scale_ref})
                                  .Build();
          
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_out_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}

TEST_F(FlatQuantProto, inferDtype_950_case_2)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950PR_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_p1_ref = ge::DT_FLOAT16;
        ge::DataType input_p2_ref = ge::DT_FLOAT16;
        ge::DataType output_out_ref = ge::DT_INT4;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        // FP4
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(3, 2)
                                  .NodeAttrs(
                                      {{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                       {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(29L)}})
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_x_ref, &input_p1_ref, &input_p2_ref})
                                  .OutputDataTypes({&output_out_ref, &output_scale_ref})
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_out_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}

TEST_F(FlatQuantProto, infershape_case_1)
{
    // mock Ascend910B
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant")->infer_shape;

    int64_t dimK = 16;
    int64_t dimM = 64;
    int64_t dimN = 64;
    gert::StorageShape xShape = {{dimK, dimM, dimN}, {dimK, dimM, dimN}};
    gert::StorageShape p1Shape = {{dimM, dimM}, {dimM, dimM}};
    gert::StorageShape p2Shape = {{dimN, dimN}, {dimN, dimN}};
    gert::StorageShape outShape = {{dimK, dimM, dimN}, {dimK, dimM, dimN}};
    gert::StorageShape scaleShape = {{dimK}, {dimK}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .NodeAttrs(
                          {{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                           {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(29L)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes({&xShape, &p1Shape, &outShape})
                      .OutputShapes({&outShape, &scaleShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* quantScale = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*output), "[16, 64, 64]");
    ASSERT_EQ(Ops::Base::ToString(*quantScale), "[16]");
}

TEST_F(FlatQuantProto, infershape_950_case_1)
{
    // mock Ascend950
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950PR_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant")->infer_shape;

    int64_t dimK = 16;
    int64_t dimM = 64;
    int64_t dimN = 64;
    int64_t dimMNCeil = 64;
    int64_t dimN_EQUAL_2 = 2;
    gert::StorageShape xShape = {{dimK, dimM, dimN}, {dimK, dimM, dimN}};
    gert::StorageShape p1Shape = {{dimM, dimM}, {dimM, dimM}};
    gert::StorageShape p2Shape = {{dimN, dimN}, {dimN, dimN}};
    gert::StorageShape outShape = {{dimK, dimM * dimN}, {dimK, dimM * dimN}};
    gert::StorageShape scaleShape = {{dimK, dimMNCeil, dimN_EQUAL_2}, {dimK, dimMNCeil, dimN_EQUAL_2}};

    // MxFP4
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .NodeAttrs(
                          {{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                           {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(40L)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes({&xShape, &p1Shape, &outShape})
                      .OutputShapes({&outShape, &scaleShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* quantScale = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*output), "[16, 2048]");
    ASSERT_EQ(Ops::Base::ToString(*quantScale), "[16, 64, 2]");
}

TEST_F(FlatQuantProto, infershape_950_case_2)
{
    // mock Ascend950
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950PR_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant")->infer_shape;

    int64_t dimK = 16;
    int64_t dimM = 64;
    int64_t dimN = 64;
    gert::StorageShape xShape = {{dimK, dimM, dimN}, {dimK, dimM, dimN}};
    gert::StorageShape p1Shape = {{dimM, dimM}, {dimM, dimM}};
    gert::StorageShape p2Shape = {{dimN, dimN}, {dimN, dimN}};
    gert::StorageShape outShape = {{dimK, dimM, dimN}, {dimK, dimM, dimN}};
    gert::StorageShape scaleShape = {{dimK}, {dimK}};

    // FP4
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .NodeAttrs(
                          {{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                           {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(29L)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes({&xShape, &p1Shape, &outShape})
                      .OutputShapes({&outShape, &scaleShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* quantScale = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*output), "[16, 64, 64]");
    ASSERT_EQ(Ops::Base::ToString(*quantScale), "[16]");
}