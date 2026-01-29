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
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/adam_apply_one_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;

class AdamApplyOneProtoTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdamApplyOneProtoTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdamApplyOneProtoTest TearDown" << std::endl;
    }
};

TEST_F(AdamApplyOneProtoTest, adam_apply_one_case_0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOne")->infer_shape;

    gert::Shape inputShape1 = {-1};
    gert::Shape inputShape2 = {-1};

    gert::Shape output_shape_0 = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(10, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&inputShape1, &inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2,
                           &inputShape2, &inputShape2, &inputShape2, &inputShape2})
                      .OutputShapes({&output_shape_0, &output_shape_0, &output_shape_0})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc_0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape_0 = {-1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_0), Ops::Base::ToString(expected_output_shape_0));

    auto output_desc_1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_output_shape_1 = {-1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_1), Ops::Base::ToString(expected_output_shape_1));

    auto output_desc_2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    gert::Shape expected_output_shape_2 = {-1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_2), Ops::Base::ToString(expected_output_shape_2));
}

TEST_F(AdamApplyOneProtoTest, infer_axis_type_with_input_dim_num_equals_1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOne")->infer_shape;

    gert::Shape inputShape1 = {-1};
    gert::Shape inputShape2 = {-1};

    gert::Shape output_shape_0 = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(10, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&inputShape1, &inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2,
                           &inputShape2, &inputShape2, &inputShape2, &inputShape2})
                      .OutputShapes({&output_shape_0, &output_shape_0, &output_shape_0})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc_0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape_0 = {-1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_0), Ops::Base::ToString(expected_output_shape_0));

    auto output_desc_1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_output_shape_1 = {-1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_1), Ops::Base::ToString(expected_output_shape_1));

    auto output_desc_2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    gert::Shape expected_output_shape_2 = {-1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_2), Ops::Base::ToString(expected_output_shape_2));
}

TEST_F(AdamApplyOneProtoTest, infer_axis_type_with_input_dim_num_great_than_1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOne")->infer_shape;

    gert::Shape inputShape1 = {-1, -1};
    gert::Shape inputShape2 = {-1, -1};

    gert::Shape output_shape_0 = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(10, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&inputShape1, &inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2,
                           &inputShape2, &inputShape2, &inputShape2, &inputShape2})
                      .OutputShapes({&output_shape_0, &output_shape_0, &output_shape_0})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc_0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape_0 = {-1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_0), Ops::Base::ToString(expected_output_shape_0));

    auto output_desc_1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_output_shape_1 = {-1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_1), Ops::Base::ToString(expected_output_shape_1));

    auto output_desc_2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    gert::Shape expected_output_shape_2 = {-1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_2), Ops::Base::ToString(expected_output_shape_2));
}

TEST_F(AdamApplyOneProtoTest, infer_axis_type_case3)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOne")->infer_shape;

    gert::Shape inputShape1 = {-1, -1, -1, -1};
    gert::Shape inputShape2 = {-1, -1, -1, -1};
    gert::Shape inputShape3 = {-1, -1};

    gert::Shape output_shape_0 = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(10, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&inputShape1, &inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2,
                           &inputShape2, &inputShape2, &inputShape2, &inputShape3})
                      .OutputShapes({&output_shape_0, &output_shape_0, &output_shape_0})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc_0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape_0 = {-1, -1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_0), Ops::Base::ToString(expected_output_shape_0));

    auto output_desc_1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_output_shape_1 = {-1, -1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_1), Ops::Base::ToString(expected_output_shape_1));

    auto output_desc_2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    gert::Shape expected_output_shape_2 = {-1, -1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_2), Ops::Base::ToString(expected_output_shape_2));
}

TEST_F(AdamApplyOneProtoTest, infer_axis_type_case4)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOne")->infer_shape;

    gert::Shape inputShape1 = {-1, -1, -1, -1};
    gert::Shape inputShape2 = {-1, -1, -1, -1};
    gert::Shape inputShape3 = {1};

    gert::Shape output_shape_0 = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(10, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&inputShape1, &inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2,
                           &inputShape2, &inputShape2, &inputShape2, &inputShape3})
                      .OutputShapes({&output_shape_0, &output_shape_0, &output_shape_0})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc_0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape_0 = {-1, -1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_0), Ops::Base::ToString(expected_output_shape_0));

    auto output_desc_1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_output_shape_1 = {-1, -1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_1), Ops::Base::ToString(expected_output_shape_1));

    auto output_desc_2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    gert::Shape expected_output_shape_2 = {-1, -1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_2), Ops::Base::ToString(expected_output_shape_2));
}

TEST_F(AdamApplyOneProtoTest, infer_axis_type_case5)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOne")->infer_shape;

    gert::Shape inputShape1 = {-1, -1, -1, -1};
    gert::Shape inputShape2 = {-1, -1, -1, -1};
    gert::Shape inputShape3 = {};

    gert::Shape output_shape_0 = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(10, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&inputShape1, &inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2,
                           &inputShape2, &inputShape2, &inputShape2, &inputShape3})
                      .OutputShapes({&output_shape_0, &output_shape_0, &output_shape_0})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc_0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape_0 = {-1, -1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_0), Ops::Base::ToString(expected_output_shape_0));

    auto output_desc_1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_output_shape_1 = {-1, -1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_1), Ops::Base::ToString(expected_output_shape_1));

    auto output_desc_2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    gert::Shape expected_output_shape_2 = {-1, -1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_2), Ops::Base::ToString(expected_output_shape_2));
}