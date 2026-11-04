/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../../../common/inc/op_host/op_const_def_nn.h"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "platform/platform_info.h"
#include "op_common/op_host/util/op_const_def.h"

using namespace ge;
using namespace std;

constexpr int32_t DIMENSION_2 = 2;
constexpr int32_t CASE1_X_M = 16;
constexpr int32_t CASE1_X_N = 256;
constexpr int32_t CASE0_X_M = 4;
constexpr int32_t CASE0_X_N = 64;
constexpr int32_t DIMENSION_3 = 3;
constexpr int32_t AICORE_CNT_960DT = 64;
constexpr int32_t HIF4_X_M = 4;
constexpr int32_t HIF4_X_N_64 = 64;
constexpr int32_t HIF4_X_N_128 = 128;
constexpr int32_t HIF4_3D_M1 = 2;
constexpr int32_t CEILDIV_64_64 = 1;
constexpr int32_t CEILDIV_128_64 = 2;
constexpr int32_t HIF4_SCALE_LAST_DIM = 1;
constexpr int32_t MXFP_SCALE_LAST_DIM = 2;
constexpr int32_t SCALE_DIM_NUM_2D = 3;
constexpr int32_t SCALE_DIM_NUM_3D = 4;
constexpr int32_t DIM_IDX_2 = 2;
constexpr int32_t DIM_IDX_3 = 3;

class RotateQuantInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RotateQuantInferShape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RotateQuantInferShape TearDown" << std::endl; }
};

TEST_F(RotateQuantInferShape, RotateQuant_infershape_case_0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{4, 64}, {4, 64}};
    gert::StorageShape rot_shape = {{64, 64}, {64, 64}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape scale_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x_shape, &rot_shape})
                      .OutputShapes({&y_shape, &scale_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_INT8))},
                           {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto yShape = context->GetOutputShape(0);
    auto scaleShape = context->GetOutputShape(1);
    ASSERT_NE(yShape, nullptr);
    ASSERT_NE(scaleShape, nullptr);

    EXPECT_EQ(yShape->GetDimNum(), DIMENSION_2);
    EXPECT_EQ(yShape->GetDim(0), CASE0_X_M);
    EXPECT_EQ(yShape->GetDim(1), CASE0_X_N);

    EXPECT_EQ(scaleShape->GetDimNum(), 1);
    EXPECT_EQ(scaleShape->GetDim(0), CASE0_X_M);
}

TEST_F(RotateQuantInferShape, RotateQuant_infershape_case_1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{16, 256}, {16, 256}};
    gert::StorageShape rot_shape = {{256, 256}, {256, 256}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape scale_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x_shape, &rot_shape})
                      .OutputShapes({&y_shape, &scale_shape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_INT8))},
                           {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto yShape = context->GetOutputShape(0);
    auto scaleShape = context->GetOutputShape(1);
    ASSERT_NE(yShape, nullptr);
    ASSERT_NE(scaleShape, nullptr);

    EXPECT_EQ(yShape->GetDimNum(), DIMENSION_2);
    EXPECT_EQ(yShape->GetDim(0), CASE1_X_M);
    EXPECT_EQ(yShape->GetDim(1), CASE1_X_N);

    EXPECT_EQ(scaleShape->GetDimNum(), 1);
    EXPECT_EQ(scaleShape->GetDim(0), CASE1_X_M);
}

// InferDataType: INT8输出
TEST_F(RotateQuantInferShape, RotateQuant_InferDtype_case_0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_BF16;
        ge::DataType input_rot_ref = ge::DT_BF16;
        ge::DataType output_y_ref = ge::DT_INT8;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .IrInstanceNum({1, 1})
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(
                                                              static_cast<int64_t>(ge::DT_INT8))},
                                              {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                                  .InputDataTypes({&input_x_ref, &input_rot_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}

static void CheckGroupedInferShapePlatform(const string& socVersion, ge::graphStatus expected, int64_t rotSlices = 0,
                                           int32_t archType = 0)
{
    SCOPED_TRACE(::testing::Message() << "soc=" << socVersion << ", rotSlices=" << rotSlices);
    fe::PlatformInfo platform;
    fe::OptionalInfo optional;
    platform.str_info.short_soc_version = socVersion;
    platform.soc_info.arch_type = archType;
    optional.soc_version = socVersion;
    fe::PlatformInfoManager::Instance().platform_info_map_[socVersion] = platform;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optional);
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant");
    ASSERT_NE(impl, nullptr);
    // Runtime inputs are compacted: the missing alpha occupies no instance.
    // The group list is the fourth IR input, but the third runtime input.
    gert::StorageShape x = {{29, 96}, {29, 96}};
    gert::StorageShape rot = {{7, 32, 32}, {7, 32, 32}};
    if (rotSlices != 0) {
        rot = gert::StorageShape({7, rotSlices, 32, 32}, {7, rotSlices, 32, 32});
    }
    gert::StorageShape groups = {{7}, {7}};
    gert::StorageShape y = {{}, {}};
    gert::StorageShape scale = {{}, {}};
    // IONum and IOInstanceNum are mutually exclusive; the first call takes precedence.
    // Describe optional input holes with IrInstanceNum only, without NodeIoNum.
    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1, 1, 0, 1}, {1, 1})
                      .InputShapes({&x, &rot, &groups})
                      .OutputShapes({&y, &scale})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT8_E4M3FN)},
                                  {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                  {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0f)},
                                  {"trans", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->GetOptionalInputShape(2), nullptr);
    ASSERT_NE(context->GetOptionalInputShape(3), nullptr);
    ASSERT_EQ(*context->GetOptionalInputShape(3), gert::Shape({7}));
    ASSERT_EQ(impl->infer_shape(context), expected);
    if (expected == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(*context->GetOutputShape(0), gert::Shape({29, 96}));
        EXPECT_EQ(*context->GetOutputShape(1), gert::Shape({29, 2, 2}));
    }
}

TEST_F(RotateQuantInferShape, GroupedRotationWithoutAlphaUsesOptionalInputIndex)
{
    CheckGroupedInferShapePlatform("Ascend950", ge::GRAPH_SUCCESS, 0, static_cast<int32_t>(Ops::Base::DAV_3510));
}

TEST_F(RotateQuantInferShape, GroupedRotationRequiresAscend950)
{
    CheckGroupedInferShapePlatform("Ascend350", ge::GRAPH_FAILED);
    CheckGroupedInferShapePlatform("Ascend910B", ge::GRAPH_FAILED);
    CheckGroupedInferShapePlatform("Ascend960DT", ge::GRAPH_FAILED);
    CheckGroupedInferShapePlatform("Ascend960PR", ge::GRAPH_FAILED);
}

TEST_F(RotateQuantInferShape, GroupedPerNRotation)
{
    CheckGroupedInferShapePlatform("Ascend950", ge::GRAPH_SUCCESS, 3, static_cast<int32_t>(Ops::Base::DAV_3510));
    CheckGroupedInferShapePlatform("Ascend950", ge::GRAPH_FAILED, 2, static_cast<int32_t>(Ops::Base::DAV_3510));
    CheckGroupedInferShapePlatform("Ascend350", ge::GRAPH_FAILED, 3);
}

static void CheckOptionalAlphaInferShape(bool hasAlpha, int64_t groupNum, int64_t alphaSize, ge::graphStatus expected)
{
    SCOPED_TRACE(::testing::Message() << "hasAlpha=" << hasAlpha << ", groupNum=" << groupNum
                                      << ", alphaSize=" << alphaSize);
    fe::PlatformInfo platform;
    fe::OptionalInfo optional;
    platform.str_info.short_soc_version = "Ascend950";
    platform.soc_info.arch_type = static_cast<int32_t>(Ops::Base::DAV_3510);
    optional.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platform;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optional);
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant");
    ASSERT_NE(impl, nullptr);

    const bool grouped = groupNum > 0;
    gert::StorageShape x = {{29, 96}, {29, 96}};
    gert::StorageShape rot = grouped ? gert::StorageShape{{groupNum, 32, 32}, {groupNum, 32, 32}} :
                                       gert::StorageShape{{32, 32}, {32, 32}};
    gert::StorageShape alpha = {{alphaSize}, {alphaSize}};
    gert::StorageShape groups = {{groupNum}, {groupNum}};
    gert::StorageShape y = {{}, {}};
    gert::StorageShape scale = {{}, {}};
    std::vector<gert::StorageShape*> inputs{&x, &rot};
    if (hasAlpha) {
        inputs.push_back(&alpha);
    }
    if (grouped) {
        inputs.push_back(&groups);
    }
    gert::InferShapeContextFaker faker;
    // Do not call NodeIoNum before IrInstanceNum: it would mask the optional input mapping.
    faker.IrInstanceNum({1, 1, hasAlpha ? 1U : 0U, grouped ? 1U : 0U}, {1, 1})
        .InputShapes(inputs)
        .OutputShapes({&y, &scale})
        .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT8_E4M3FN)},
                    {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                    {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                    {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                    {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0f)},
                    {"trans", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}});
    if (hasAlpha) {
        faker.NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (grouped) {
        faker.NodeInputTd(hasAlpha ? 3 : 2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->GetOptionalInputShape(2) != nullptr, hasAlpha);
    ASSERT_EQ(context->GetOptionalInputShape(3) != nullptr, grouped);
    if (hasAlpha) {
        ASSERT_EQ(*context->GetOptionalInputShape(2), gert::Shape({alphaSize}));
    }
    if (grouped) {
        ASSERT_EQ(*context->GetOptionalInputShape(3), gert::Shape({groupNum}));
    }
    ASSERT_EQ(impl->infer_shape(context), expected);
    if (expected == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(*context->GetOutputShape(0), gert::Shape({29, 96}));
        EXPECT_EQ(*context->GetOutputShape(1), gert::Shape({29, 2, 2}));
    }
}

TEST_F(RotateQuantInferShape, OptionalAlphaPreservesOrdinaryAndGroupedInputs)
{
    for (int64_t groupNum : {0, 1, 7, 1024}) {
        CheckOptionalAlphaInferShape(false, groupNum, 1, ge::GRAPH_SUCCESS);
        CheckOptionalAlphaInferShape(true, groupNum, 1, ge::GRAPH_SUCCESS);
        CheckOptionalAlphaInferShape(true, groupNum, 0, ge::GRAPH_FAILED);
        CheckOptionalAlphaInferShape(true, groupNum, 2, ge::GRAPH_FAILED);
    }
}

// ==================== ascend960dt (v121) platform mock ====================
static void SetPlatform960DT()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = AICORE_CNT_960DT;
    platformInfo.str_info.short_soc_version = "Ascend960DT";
    optiCompilationInfo.soc_version = "Ascend960DT";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend960DT"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

// hif4 infershape: 2D x, scale shape = (M, CeilDiv(N,64), 1)
TEST_F(RotateQuantInferShape, RotateQuant_infershape_hif4_2d_k64)
{
    SetPlatform960DT();
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{4, 64}, {4, 64}};
    gert::StorageShape rot_shape = {{64, 64}, {64, 64}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape scale_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x_shape, &rot_shape})
                      .OutputShapes({&y_shape, &scale_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, Ops::NN::DT_HIFLOAT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, Ops::NN::DT_HIFLOAT4_SCALE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"y_dtype",
                                   Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(Ops::NN::DT_HIFLOAT4))},
                                  {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto yShape = context->GetOutputShape(0);
    auto scaleShape = context->GetOutputShape(1);
    ASSERT_NE(yShape, nullptr);
    ASSERT_NE(scaleShape, nullptr);

    EXPECT_EQ(yShape->GetDimNum(), DIMENSION_2);
    EXPECT_EQ(yShape->GetDim(0), HIF4_X_M);
    EXPECT_EQ(yShape->GetDim(1), HIF4_X_N_64);

    EXPECT_EQ(scaleShape->GetDimNum(), SCALE_DIM_NUM_2D);
    EXPECT_EQ(scaleShape->GetDim(0), HIF4_X_M);
    EXPECT_EQ(scaleShape->GetDim(1), CEILDIV_64_64);               // CeilDiv(64,64)=1
    EXPECT_EQ(scaleShape->GetDim(DIM_IDX_2), HIF4_SCALE_LAST_DIM); // hif4 last dim = 1
}

// hif4 infershape: 2D x, K=128, scale shape = (M, CeilDiv(N,64), 1) = (4, 2, 1)
TEST_F(RotateQuantInferShape, RotateQuant_infershape_hif4_2d_k128)
{
    SetPlatform960DT();
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{4, 128}, {4, 128}};
    gert::StorageShape rot_shape = {{128, 128}, {128, 128}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape scale_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x_shape, &rot_shape})
                      .OutputShapes({&y_shape, &scale_shape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, Ops::NN::DT_HIFLOAT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, Ops::NN::DT_HIFLOAT4_SCALE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"y_dtype",
                                   Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(Ops::NN::DT_HIFLOAT4))},
                                  {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto yShape = context->GetOutputShape(0);
    auto scaleShape = context->GetOutputShape(1);
    ASSERT_NE(yShape, nullptr);
    ASSERT_NE(scaleShape, nullptr);

    EXPECT_EQ(yShape->GetDimNum(), DIMENSION_2);
    EXPECT_EQ(yShape->GetDim(0), HIF4_X_M);
    EXPECT_EQ(yShape->GetDim(1), HIF4_X_N_128);

    EXPECT_EQ(scaleShape->GetDimNum(), SCALE_DIM_NUM_2D);
    EXPECT_EQ(scaleShape->GetDim(0), HIF4_X_M);
    EXPECT_EQ(scaleShape->GetDim(1), CEILDIV_128_64);              // CeilDiv(128,64)=2
    EXPECT_EQ(scaleShape->GetDim(DIM_IDX_2), HIF4_SCALE_LAST_DIM); // hif4 last dim = 1
}

// mxfp4 infershape: 2D x, K=64, scale shape = (M, CeilDiv(N,64), 2)
TEST_F(RotateQuantInferShape, RotateQuant_infershape_mxfp4_2d_k64)
{
    SetPlatform960DT();
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{4, 64}, {4, 64}};
    gert::StorageShape rot_shape = {{64, 64}, {64, 64}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape scale_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x_shape, &rot_shape})
                      .OutputShapes({&y_shape, &scale_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"y_dtype",
                                   Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT4_E2M1))},
                                  {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto yShape = context->GetOutputShape(0);
    auto scaleShape = context->GetOutputShape(1);
    ASSERT_NE(yShape, nullptr);
    ASSERT_NE(scaleShape, nullptr);

    EXPECT_EQ(yShape->GetDimNum(), DIMENSION_2);
    EXPECT_EQ(yShape->GetDim(0), HIF4_X_M);
    EXPECT_EQ(yShape->GetDim(1), HIF4_X_N_64);

    EXPECT_EQ(scaleShape->GetDimNum(), SCALE_DIM_NUM_2D);
    EXPECT_EQ(scaleShape->GetDim(0), HIF4_X_M);
    EXPECT_EQ(scaleShape->GetDim(1), CEILDIV_64_64);               // CeilDiv(64,64)=1
    EXPECT_EQ(scaleShape->GetDim(DIM_IDX_2), MXFP_SCALE_LAST_DIM); // mxfp4 last dim = 2
}

// mxfp8 infershape: 2D x, K=128, scale shape = (M, CeilDiv(N,64), 2) = (4, 2, 2)
TEST_F(RotateQuantInferShape, RotateQuant_infershape_mxfp8_2d_k128)
{
    SetPlatform960DT();
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{4, 128}, {4, 128}};
    gert::StorageShape rot_shape = {{128, 128}, {128, 128}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape scale_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x_shape, &rot_shape})
                      .OutputShapes({&y_shape, &scale_shape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"y_dtype",
                                   Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN))},
                                  {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto yShape = context->GetOutputShape(0);
    auto scaleShape = context->GetOutputShape(1);
    ASSERT_NE(yShape, nullptr);
    ASSERT_NE(scaleShape, nullptr);

    EXPECT_EQ(yShape->GetDimNum(), DIMENSION_2);
    EXPECT_EQ(yShape->GetDim(0), HIF4_X_M);
    EXPECT_EQ(yShape->GetDim(1), HIF4_X_N_128);

    EXPECT_EQ(scaleShape->GetDimNum(), SCALE_DIM_NUM_2D);
    EXPECT_EQ(scaleShape->GetDim(0), HIF4_X_M);
    EXPECT_EQ(scaleShape->GetDim(1), CEILDIV_128_64);              // CeilDiv(128,64)=2
    EXPECT_EQ(scaleShape->GetDim(DIM_IDX_2), MXFP_SCALE_LAST_DIM); // mxfp8 last dim = 2
}

// hif4 infershape: 3D x (M1, M2, N), scale shape = (M1, M2, CeilDiv(N,64), 1)
TEST_F(RotateQuantInferShape, RotateQuant_infershape_hif4_3d_k64)
{
    SetPlatform960DT();
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{2, 4, 64}, {2, 4, 64}};
    gert::StorageShape rot_shape = {{64, 64}, {64, 64}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape scale_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x_shape, &rot_shape})
                      .OutputShapes({&y_shape, &scale_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, Ops::NN::DT_HIFLOAT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, Ops::NN::DT_HIFLOAT4_SCALE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"y_dtype",
                                   Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(Ops::NN::DT_HIFLOAT4))},
                                  {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto yShape = context->GetOutputShape(0);
    auto scaleShape = context->GetOutputShape(1);
    ASSERT_NE(yShape, nullptr);
    ASSERT_NE(scaleShape, nullptr);

    EXPECT_EQ(yShape->GetDimNum(), DIMENSION_3);
    EXPECT_EQ(yShape->GetDim(0), HIF4_3D_M1);
    EXPECT_EQ(yShape->GetDim(1), HIF4_X_M);
    EXPECT_EQ(yShape->GetDim(DIM_IDX_2), HIF4_X_N_64);

    EXPECT_EQ(scaleShape->GetDimNum(), SCALE_DIM_NUM_3D);
    EXPECT_EQ(scaleShape->GetDim(0), HIF4_3D_M1);
    EXPECT_EQ(scaleShape->GetDim(1), HIF4_X_M);
    EXPECT_EQ(scaleShape->GetDim(DIM_IDX_2), CEILDIV_64_64);       // CeilDiv(64,64)=1
    EXPECT_EQ(scaleShape->GetDim(DIM_IDX_3), HIF4_SCALE_LAST_DIM); // hif4 last dim = 1
}

// InferDataType: HIFLOAT4 output -> HIFLOAT4_SCALE
TEST_F(RotateQuantInferShape, RotateQuant_InferDtype_hif4)
{
    SetPlatform960DT();
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_x_ref = ge::DT_BF16;
    ge::DataType input_rot_ref = ge::DT_BF16;
    ge::DataType output_y_ref = Ops::NN::DT_HIFLOAT4;
    ge::DataType output_scale_ref = Ops::NN::DT_HIFLOAT4_SCALE;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(2)
                              .NodeIoNum(2, 2)
                              .IrInstanceNum({1, 1})
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, Ops::NN::DT_HIFLOAT4, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, Ops::NN::DT_HIFLOAT4_SCALE, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(
                                                          static_cast<int64_t>(Ops::NN::DT_HIFLOAT4))},
                                          {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                              .InputDataTypes({&input_x_ref, &input_rot_ref})
                              .OutputDataTypes({&output_y_ref, &output_scale_ref})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
    EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
}

// InferDataType: mxfp4 (FLOAT4_E2M1) output -> FLOAT8_E8M0 scale
TEST_F(RotateQuantInferShape, RotateQuant_InferDtype_mxfp4)
{
    SetPlatform960DT();
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_x_ref = ge::DT_FLOAT16;
    ge::DataType input_rot_ref = ge::DT_FLOAT16;
    ge::DataType output_y_ref = ge::DT_FLOAT4_E2M1;
    ge::DataType output_scale_ref = ge::DT_FLOAT8_E8M0;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(2)
                              .NodeIoNum(2, 2)
                              .IrInstanceNum({1, 1})
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(
                                                          static_cast<int64_t>(ge::DT_FLOAT4_E2M1))},
                                          {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                              .InputDataTypes({&input_x_ref, &input_rot_ref})
                              .OutputDataTypes({&output_y_ref, &output_scale_ref})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
    EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
}
