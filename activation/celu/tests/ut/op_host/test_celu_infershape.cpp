/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_celu_infershape.cpp
 * \brief
 */

#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "base/context_builder/op_infer_shape_context_builder.h"
#include "graph/operator_factory.h"
#include "../../../op_graph/celu_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class CeluProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "celu Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "celu Proto Test TearDown" << std::endl; }
};

TEST_F(CeluProtoTest, celu_infershape_float16_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Celu")->infer_shape;

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

TEST_F(CeluProtoTest, celu_infershape_float_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Celu")->infer_shape;

    gert::Shape input_shape_0 = {1, 3, 4};
    gert::Shape output_shape_0 = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape_0})
                      .OutputShapes({&output_shape_0})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(CeluProtoTest, celu_infershape_scalar_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Celu")->infer_shape;

    gert::Shape input_shape_0 = {};
    gert::Shape output_shape_0 = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape_0})
                      .OutputShapes({&output_shape_0})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(CeluProtoTest, celu_infershape_empty_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Celu")->infer_shape;

    gert::Shape input_shape_0 = {0};
    gert::Shape output_shape_0 = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape_0})
                      .OutputShapes({&output_shape_0})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// 必选输入 shape 不可达（IONum(1,1) 但未提供 InputTensors，输入链空位），
// 推断须早期拒绝并上报正式首报码 EZ0004。
TEST_F(CeluProtoTest, celu_infershape_input_shape_unreachable_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Celu")->infer_shape;

    gert::OpInferShapeContextBuilder builder;
    builder.OpType("Celu").OpName("celu_infershape_input_unreachable");
    builder.IONum(1, 1);
    builder.OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->GetInputShape(0), nullptr);

    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

// InferShape 框架输出对象不可达（IOInstanceNum 输出实例 0，输出链空位），
// 推断须拒绝并上报正式首报码 EZ1014。
TEST_F(CeluProtoTest, celu_infershape_output_shape_unreachable_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Celu")->infer_shape;

    gert::Tensor inputTensor;
    gert::OpInferShapeContextBuilder builder;
    builder.OpType("Celu").OpName("celu_infershape_output_unreachable");
    builder.IOInstanceNum(std::vector<uint32_t>{1}, std::vector<uint32_t>{0});
    builder.InputTensors({&inputTensor});
    builder.OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->GetOutputShape(0), nullptr);

    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

// proto 注册面可经 OperatorFactory 构造（覆盖 celu_proto.h 的 REG_OP 注册链：
// INPUT/OUTPUT/ATTR 定义与 creator 首次实例化）。
TEST_F(CeluProtoTest, celu_proto_factory_create_test)
{
    ASSERT_TRUE(ge::OperatorFactory::IsExistOp("Celu"));
    ge::Operator celuOp = ge::OperatorFactory::CreateOperator("celu_ut_factory", "Celu");
    ASSERT_FALSE(celuOp.IsEmpty());
}
