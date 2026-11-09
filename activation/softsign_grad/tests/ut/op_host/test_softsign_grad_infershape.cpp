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
#include "../../../op_graph/softsign_grad_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class SoftsignGradProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SoftsignGrad Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SoftsignGrad Proto Test TearDown" << std::endl; }
};

TEST_F(SoftsignGradProtoTest, softsign_grad_infershape_3d_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftsignGrad")->infer_shape;

    gert::Shape input_shape_0 = {4, 3, 4};
    gert::Shape input_shape_1 = {4, 3, 4};
    gert::Shape output_shape_0 = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_0, &input_shape_1})
                      .OutputShapes({&output_shape_0})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(SoftsignGradProtoTest, softsign_grad_infershape_2d_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftsignGrad")->infer_shape;

    gert::Shape input_shape_0 = {8, 16};
    gert::Shape input_shape_1 = {8, 16};
    gert::Shape output_shape_0 = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_0, &input_shape_1})
                      .OutputShapes({&output_shape_0})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(SoftsignGradProtoTest, softsign_grad_infershape_1d_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftsignGrad")->infer_shape;

    gert::Shape input_shape_0 = {128};
    gert::Shape input_shape_1 = {128};
    gert::Shape output_shape_0 = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_0, &input_shape_1})
                      .OutputShapes({&output_shape_0})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// ===== 负例 UT（infer-exception-005 补齐）=====

class SoftsignGradNegativeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static void TearDownTestCase() {}
};

// rank 9 越界
TEST_F(SoftsignGradNegativeTest, infershape_rank9_rejected)
{
    auto func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftsignGrad")->infer_shape;
    gert::Shape shape0 = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    gert::Shape shape1 = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    gert::Shape out = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&shape0, &shape1})
                      .OutputShapes({&out})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// shape 具体值冲突
TEST_F(SoftsignGradNegativeTest, infershape_dim_conflict_rejected)
{
    auto func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftsignGrad")->infer_shape;
    gert::Shape shape0 = {2, 3};
    gert::Shape shape1 = {2, 4};
    gert::Shape out = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&shape0, &shape1})
                      .OutputShapes({&out})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// rank 不一致（2D vs 3D）
TEST_F(SoftsignGradNegativeTest, infershape_rank_mismatch_rejected)
{
    auto func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftsignGrad")->infer_shape;
    gert::Shape shape0 = {2, 3};
    gert::Shape shape1 = {2, 3, 4};
    gert::Shape out = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&shape0, &shape1})
                      .OutputShapes({&out})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 动态 shape 相容（-1 与具体值合并）→ 保护性正例
TEST_F(SoftsignGradNegativeTest, infershape_dyn_compat_pass)
{
    auto func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftsignGrad")->infer_shape;
    gert::Shape shape0 = {-1, 5};
    gert::Shape shape1 = {3, 5};
    gert::Shape out = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&shape0, &shape1})
                      .OutputShapes({&out})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// UnknownRank (-2) 传播 → 保护性正例
TEST_F(SoftsignGradNegativeTest, infershape_unknown_rank_propagate)
{
    auto func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftsignGrad")->infer_shape;
    gert::Shape shape0 = {-2};
    gert::Shape shape1 = {3, 5};
    gert::Shape out = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&shape0, &shape1})
                      .OutputShapes({&out})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// InferDataType 负例 UT 暂缺：InferDataType 经 op_graph/graph_infer.cpp 的 IMPL_OP 注册，
// op_host UT 链接链不含 op_graph 目标，infer_datatype 函数指针在 UT 环境中为 nullptr。
// InferDataType 校验由设备端 L2 套件（a5-exceptions 19 例全过）覆盖。
// 如需主机级 UT，需创建 tests/ut/op_graph/ 或将注册移回 op_host（GEIR.01/1 修复决定在 op_graph）。
