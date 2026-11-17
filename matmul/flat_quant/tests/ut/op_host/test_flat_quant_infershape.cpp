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
#include "op_common/op_host/util/op_const_def.h"
#include "../../../../../common/inc/op_host/op_const_def_nn.h"
#include "../../../op_graph/flat_quant_proto.h"

using namespace ge;
using namespace op;

class FlatQuantProto : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FlatQuantProto SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FlatQuantProto TearDown" << std::endl; }
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
                                  .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
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

TEST_F(FlatQuantProto, inferDtype_dav3510_case_1)
{
    // mock DAV_3510
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.soc_info.arch_type = 3510;
    platformInfo.str_info.short_soc_version = "dav3510";
    optiCompilationInfo.soc_version = "dav3510";
    fe::PlatformInfoManager::Instance().platform_info_map_["dav3510"] = platformInfo;
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
                                  .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
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

TEST_F(FlatQuantProto, inferDtype_dav3510_case_2)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.soc_info.arch_type = 3510;
    platformInfo.str_info.short_soc_version = "dav3510";
    optiCompilationInfo.soc_version = "dav3510";
    fe::PlatformInfoManager::Instance().platform_info_map_["dav3510"] = platformInfo;
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
                                  .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
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
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
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

TEST_F(FlatQuantProto, infershape_dav3510_case_1)
{
    // mock DAV_3510
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.soc_info.arch_type = 3510;
    platformInfo.str_info.short_soc_version = "dav3510";
    optiCompilationInfo.soc_version = "dav3510";
    fe::PlatformInfoManager::Instance().platform_info_map_["dav3510"] = platformInfo;
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
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
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
    ASSERT_EQ(Ops::Base::ToString(*output), "[16, 4096]");
    ASSERT_EQ(Ops::Base::ToString(*quantScale), "[16, 64, 2]");
}

TEST_F(FlatQuantProto, infershape_dav3510_case_2)
{
    // mock DAV_3510
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.soc_info.arch_type = 3510;
    platformInfo.str_info.short_soc_version = "dav3510";
    optiCompilationInfo.soc_version = "dav3510";
    fe::PlatformInfoManager::Instance().platform_info_map_["dav3510"] = platformInfo;
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
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
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

// ==================== ascend960dt/pr (DAV_9201/DAV_9202) 平台 mock ====================
static void SetFlatQuantPlatform920Series(const char* socVersion, int32_t archType)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    // SocVersion→NpuArch迁移后infershape按soc_info.arch_type判断平台, mock需同步补该字段
    platformInfo.soc_info.arch_type = archType;
    platformInfo.str_info.short_soc_version = socVersion;
    optiCompilationInfo.soc_version = socVersion;
    fe::PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

TEST_F(FlatQuantProto, inferDtype_dav9201_case_1)
{
    // mock DAV_9201, MxFP4
    SetFlatQuantPlatform920Series("Ascend960DT", static_cast<int32_t>(Ops::Base::DAV_9201));
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_p1_ref = ge::DT_FLOAT16;
        ge::DataType input_p2_ref = ge::DT_FLOAT16;
        ge::DataType output_out_ref = ge::DT_FLOAT4_E2M1;
        ge::DataType output_scale_ref = ge::DT_FLOAT8_E8M0;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(3, 2)
                                  .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
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

TEST_F(FlatQuantProto, inferDtype_dav9201_case_2)
{
    // mock DAV_9201, HIF4 (仅ascend960系列支持)
    SetFlatQuantPlatform920Series("Ascend960DT", static_cast<int32_t>(Ops::Base::DAV_9201));
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_p1_ref = ge::DT_FLOAT16;
        ge::DataType input_p2_ref = ge::DT_FLOAT16;
        ge::DataType output_out_ref = Ops::NN::DT_HIFLOAT4;
        ge::DataType output_scale_ref = Ops::NN::DT_HIFLOAT4_SCALE;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(3, 2)
                                  .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                              {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(42L)}})
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, Ops::NN::DT_HIFLOAT4, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, Ops::NN::DT_HIFLOAT4_SCALE, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(FlatQuantProto, inferDtype_dav9202_case_1)
{
    // mock DAV_9202, INT4
    SetFlatQuantPlatform920Series("Ascend960PR", static_cast<int32_t>(Ops::Base::DAV_9202));
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
                                  .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
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

TEST_F(FlatQuantProto, infershape_dav9201_case_1)
{
    // mock DAV_9201, MxFP4: out=[K, M*N], scale=[K, ceil(M*N/64), 2]
    SetFlatQuantPlatform920Series("Ascend960DT", static_cast<int32_t>(Ops::Base::DAV_9201));
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

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
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
    ASSERT_EQ(Ops::Base::ToString(*output), "[16, 4096]");
    ASSERT_EQ(Ops::Base::ToString(*quantScale), "[16, 64, 2]");
}

TEST_F(FlatQuantProto, infershape_dav9201_case_2)
{
    // mock DAV_9201, HIF4: out=[K, M*N], scale=[K, ceil(M*N/64), 1]
    SetFlatQuantPlatform920Series("Ascend960DT", static_cast<int32_t>(Ops::Base::DAV_9201));
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FlatQuant")->infer_shape;

    int64_t dimK = 16;
    int64_t dimM = 64;
    int64_t dimN = 64;
    int64_t dimMNCeil = 64;
    gert::StorageShape xShape = {{dimK, dimM, dimN}, {dimK, dimM, dimN}};
    gert::StorageShape p1Shape = {{dimM, dimM}, {dimM, dimM}};
    gert::StorageShape p2Shape = {{dimN, dimN}, {dimN, dimN}};
    gert::StorageShape outShape = {{dimK, dimM * dimN}, {dimK, dimM * dimN}};
    gert::StorageShape scaleShape = {{dimK, dimMNCeil, 1}, {dimK, dimMNCeil, 1}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(42L)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, Ops::NN::DT_HIFLOAT4, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, Ops::NN::DT_HIFLOAT4_SCALE, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes({&xShape, &p1Shape, &outShape})
                      .OutputShapes({&outShape, &scaleShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* quantScale = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*output), "[16, 4096]");
    ASSERT_EQ(Ops::Base::ToString(*quantScale), "[16, 64, 1]");
}

TEST_F(FlatQuantProto, infershape_dav9202_case_1)
{
    // mock DAV_9202, INT4: out=[K, M, N], scale=[K]
    SetFlatQuantPlatform920Series("Ascend960PR", static_cast<int32_t>(Ops::Base::DAV_9202));
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
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
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

TEST_F(FlatQuantProto, infershape_dav9201_case_3)
{
    // mock DAV_9201, MxFP4且dst_type_max=5.0(不在0或[6,12]内) -> 失败
    SetFlatQuantPlatform920Series("Ascend960DT", static_cast<int32_t>(Ops::Base::DAV_9201));
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

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(40L)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(5.0f)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .InputShapes({&xShape, &p1Shape, &outShape})
                      .OutputShapes({&outShape, &scaleShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
