/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_gelu_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/gelu_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class GeluProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gelu Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gelu Proto Test TearDown" << std::endl;
  }
};

TEST_F(GeluProtoTest, gelu_infershape_diff_test) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 64;
  platformInfo.str_info.short_soc_version = "Ascend950";
  optiCompilationInfo.soc_version = "Ascend950";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
  
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Gelu")->infer_shape;

  gert::Shape input_shape_0 = {4, 3, 4};
  gert::Shape output_shape_0 = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&input_shape_0})
                    .OutputShapes({&output_shape_0})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();
  
  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(GeluProtoTest, gelu_infershape_same_test) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 64;
  platformInfo.str_info.short_soc_version = "Ascend950";
  optiCompilationInfo.soc_version = "Ascend950";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
  
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Gelu")->infer_shape;

  gert::Shape input_shape_0 = {1, 3, 4};
  gert::Shape output_shape_0 = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&input_shape_0})
                    .OutputShapes({&output_shape_0})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();
  
  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}