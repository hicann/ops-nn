/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "log/log.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "platform/platform_info.h"

class LpNormV2Test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "LpNormV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LpNormV2 TearDown" << std::endl;
  }
};

TEST_F(LpNormV2Test, lp_norm_v2_infer_shape_fp16_p_1_keepdim_true) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LpNormV2")->infer_shape;
    gert::Shape xShape = {2, 64, 224, 224};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs(
                          {{"p", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"axes", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 2})},
                           {"keepdim", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {2, 1, 1, 224};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

TEST_F(LpNormV2Test, lp_norm_v2_infer_shape_fp32_p_2_keepdim_false) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LpNormV2")->infer_shape;
    gert::Shape xShape = {2, 64, 224, 224};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs(
                          {{"p", Ops::NN::AnyValue::CreateFrom<float>(2.0)},
                           {"axes", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 2})},
                           {"keepdim", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {2, 224};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

TEST_F(LpNormV2Test, lp_norm_v2_infer_shape_fp32_p_3_5_keepdim_false) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LpNormV2")->infer_shape;
    gert::Shape xShape = {2, 64, 224, 224};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs(
                          {{"p", Ops::NN::AnyValue::CreateFrom<float>(3.5)},
                           {"axes", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 2})},
                           {"keepdim", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {64, 224};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

TEST_F(LpNormV2Test, lp_norm_v2_infer_shape_fp32_p_0_keepdim_true) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LpNormV2")->infer_shape;
    gert::Shape xShape = {2, 64, 224, 224};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs(
                          {{"p", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                           {"axes", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 2})},
                           {"keepdim", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {1, 64, 1, 224};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

TEST_F(LpNormV2Test, lp_norm_v2_infer_shape_dynamic_fp32_p_2_keepdim_true) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LpNormV2")->infer_shape;
    gert::Shape xShape = {-1, -1, -1, 224};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs(
                          {{"p", Ops::NN::AnyValue::CreateFrom<float>(2.0)},
                           {"axes", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 2})},
                           {"keepdim", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {1, -1, 1, 224};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

TEST_F(LpNormV2Test, lp_norm_v2_infer_shape_dynamic_fp32_p_neg_2_keepdim_false) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LpNormV2")->infer_shape;
    gert::Shape xShape = {-1, -1, -1, 224};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs(
                          {{"p", Ops::NN::AnyValue::CreateFrom<float>(-2.0)},
                           {"axes", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({})},
                           {"keepdim", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}