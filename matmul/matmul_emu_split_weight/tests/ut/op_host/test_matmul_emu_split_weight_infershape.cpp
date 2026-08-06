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
#include "ut_op_common.h"
#include "platform/platform_info.h"

using namespace ge;
using namespace std;

constexpr int32_t DIMENSION_2 = 2;
constexpr int32_t CASE0_X_M = 128;
constexpr int32_t CASE0_X_K = 256;
constexpr int32_t CASE0_W_N = 128;
constexpr int32_t CASE1_X_M = 16;
constexpr int32_t CASE1_X_K = 64;
constexpr int32_t CASE1_W_N = 64;

class MatmulEmuSplitWeightInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MatmulEmuSplitWeightInferShape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MatmulEmuSplitWeightInferShape TearDown" << std::endl; }
};

TEST_F(MatmulEmuSplitWeightInferShape, MatmulEmuSplitWeight_infershape_case_0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulEmuSplitWeight"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulEmuSplitWeight")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{CASE0_X_M, CASE0_X_K}, {CASE0_X_M, CASE0_X_K}};
    gert::StorageShape wHigh_shape = {{CASE0_X_K, CASE0_W_N}, {CASE0_X_K, CASE0_W_N}};
    gert::StorageShape wLow_shape = {{CASE0_X_K, CASE0_W_N}, {CASE0_X_K, CASE0_W_N}};
    gert::StorageShape y_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &wHigh_shape, &wLow_shape})
                      .OutputShapes({&y_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"w_low_scale", Ops::NN::AnyValue::CreateFrom<float>(0.00390625f)},
                                  {"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_w", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto yShape = context->GetOutputShape(0);
    ASSERT_NE(yShape, nullptr);
    EXPECT_EQ(yShape->GetDimNum(), DIMENSION_2);
    EXPECT_EQ(yShape->GetDim(0), CASE0_X_M);
    EXPECT_EQ(yShape->GetDim(1), CASE0_W_N);
}

TEST_F(MatmulEmuSplitWeightInferShape, MatmulEmuSplitWeight_infershape_case_1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulEmuSplitWeight"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulEmuSplitWeight")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{CASE1_X_M, CASE1_X_K}, {CASE1_X_M, CASE1_X_K}};
    gert::StorageShape wHigh_shape = {{CASE1_X_K, CASE1_W_N}, {CASE1_X_K, CASE1_W_N}};
    gert::StorageShape wLow_shape = {{CASE1_X_K, CASE1_W_N}, {CASE1_X_K, CASE1_W_N}};
    gert::StorageShape y_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &wHigh_shape, &wLow_shape})
                      .OutputShapes({&y_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"w_low_scale", Ops::NN::AnyValue::CreateFrom<float>(0.00390625f)},
                                  {"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_w", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto yShape = context->GetOutputShape(0);
    ASSERT_NE(yShape, nullptr);
    EXPECT_EQ(yShape->GetDimNum(), DIMENSION_2);
    EXPECT_EQ(yShape->GetDim(0), CASE1_X_M);
    EXPECT_EQ(yShape->GetDim(1), CASE1_W_N);
}

TEST_F(MatmulEmuSplitWeightInferShape, MatmulEmuSplitWeight_infershape_k_mismatch_fail)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulEmuSplitWeight"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulEmuSplitWeight")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{CASE0_X_M, CASE0_X_K}, {CASE0_X_M, CASE0_X_K}};
    gert::StorageShape wHigh_shape = {{CASE0_X_K + 1, CASE0_W_N}, {CASE0_X_K + 1, CASE0_W_N}};
    gert::StorageShape wLow_shape = {{CASE0_X_K + 1, CASE0_W_N}, {CASE0_X_K + 1, CASE0_W_N}};
    gert::StorageShape y_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &wHigh_shape, &wLow_shape})
                      .OutputShapes({&y_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"w_low_scale", Ops::NN::AnyValue::CreateFrom<float>(0.00390625f)},
                                  {"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_w", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_NE(infer_shape_func(context), ge::GRAPH_SUCCESS);
}

TEST_F(MatmulEmuSplitWeightInferShape, MatmulEmuSplitWeight_infershape_wlow_n_mismatch_fail)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulEmuSplitWeight"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulEmuSplitWeight")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{CASE0_X_M, CASE0_X_K}, {CASE0_X_M, CASE0_X_K}};
    gert::StorageShape wHigh_shape = {{CASE0_X_K, CASE0_W_N}, {CASE0_X_K, CASE0_W_N}};
    gert::StorageShape wLow_shape = {{CASE0_X_K, CASE0_W_N + 1}, {CASE0_X_K, CASE0_W_N + 1}};
    gert::StorageShape y_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &wHigh_shape, &wLow_shape})
                      .OutputShapes({&y_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"w_low_scale", Ops::NN::AnyValue::CreateFrom<float>(0.00390625f)},
                                  {"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_w", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_NE(infer_shape_func(context), ge::GRAPH_SUCCESS);
}

TEST_F(MatmulEmuSplitWeightInferShape, MatmulEmuSplitWeight_InferDtype_case_0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulEmuSplitWeight"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulEmuSplitWeight")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_x_ref = ge::DT_BF16;
    ge::DataType input_wHigh_ref = ge::DT_BF16;
    ge::DataType input_wLow_ref = ge::DT_BF16;
    ge::DataType output_y_ref = ge::DT_FLOAT;

    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({{"w_low_scale", Ops::NN::AnyValue::CreateFrom<float>(0.00390625f)},
                                          {"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                          {"transpose_w", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                          {"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                              .InputDataTypes({&input_x_ref, &input_wHigh_ref, &input_wLow_ref})
                              .OutputDataTypes({&output_y_ref})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
}
