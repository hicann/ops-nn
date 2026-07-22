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
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/inplace_apply_adagrad_v2_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class InplaceApplyAdagradV2InferShapeTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "InplaceApplyAdagradV2 InferShape Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "InplaceApplyAdagradV2 InferShape Test TearDown" << std::endl;
  }

  static void InitPlatform() {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
  }
};

TEST_F(InplaceApplyAdagradV2InferShapeTest, infershape_1d_fp32_test) {
  InitPlatform();
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradV2")->infer_shape;

  gert::Shape var_shape = {128};
  gert::Shape out_var_shape = {};
  gert::Shape out_accum_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 2)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&var_shape, &var_shape, &var_shape, &var_shape})
                    .OutputShapes({&out_var_shape, &out_accum_shape})
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(InplaceApplyAdagradV2InferShapeTest, infershape_scalar_test) {
  InitPlatform();
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradV2")->infer_shape;

  gert::Shape var_shape = {};
  gert::Shape out_var_shape = {};
  gert::Shape out_accum_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 2)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&var_shape, &var_shape, &var_shape, &var_shape})
                    .OutputShapes({&out_var_shape, &out_accum_shape})
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(InplaceApplyAdagradV2InferShapeTest, infershape_high_rank_test) {
  InitPlatform();
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradV2")->infer_shape;

  gert::Shape var_shape = {2, 3, 4, 5, 6};
  gert::Shape out_var_shape = {};
  gert::Shape out_accum_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 2)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&var_shape, &var_shape, &var_shape, &var_shape})
                    .OutputShapes({&out_var_shape, &out_accum_shape})
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}
