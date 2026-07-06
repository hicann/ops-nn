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
#include "../../../op_graph/apply_proximal_adagrad_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class ApplyProximalAdagradProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyProximalAdagrad Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyProximalAdagrad Proto Test TearDown" << std::endl; }
};

namespace {
void RegisterAscend950Platform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}
} // namespace

TEST_F(ApplyProximalAdagradProtoTest, infershape_3d_fp32_test)
{
    RegisterAscend950Platform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyProximalAdagrad")->infer_shape;

    gert::Shape var_shape = {4, 3, 4};
    gert::Shape accum_shape = {4, 3, 4};
    gert::Shape lr_shape = {1};
    gert::Shape l1_shape = {1};
    gert::Shape l2_shape = {1};
    gert::Shape grad_shape = {4, 3, 4};
    gert::Shape var_out = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(6, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&var_shape, &accum_shape, &lr_shape, &l1_shape, &l2_shape, &grad_shape})
                      .OutputShapes({&var_out})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ApplyProximalAdagradProtoTest, infershape_2d_fp16_test)
{
    RegisterAscend950Platform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyProximalAdagrad")->infer_shape;

    gert::Shape var_shape = {8, 16};
    gert::Shape accum_shape = {8, 16};
    gert::Shape lr_shape = {1};
    gert::Shape l1_shape = {1};
    gert::Shape l2_shape = {1};
    gert::Shape grad_shape = {8, 16};
    gert::Shape var_out = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(6, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&var_shape, &accum_shape, &lr_shape, &l1_shape, &l2_shape, &grad_shape})
                      .OutputShapes({&var_out})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ApplyProximalAdagradProtoTest, infershape_1d_bf16_test)
{
    RegisterAscend950Platform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyProximalAdagrad")->infer_shape;

    gert::Shape var_shape = {128};
    gert::Shape accum_shape = {128};
    gert::Shape lr_shape = {1};
    gert::Shape l1_shape = {1};
    gert::Shape l2_shape = {1};
    gert::Shape grad_shape = {128};
    gert::Shape var_out = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(6, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&var_shape, &accum_shape, &lr_shape, &l1_shape, &l2_shape, &grad_shape})
                      .OutputShapes({&var_out})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}
