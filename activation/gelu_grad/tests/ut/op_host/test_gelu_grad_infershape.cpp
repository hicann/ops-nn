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
#include <gtest/gtest.h>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/gelu_grad_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class gelugrad : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "gelugrad SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "gelugrad TearDown" << std::endl; }
};

TEST_F(gelugrad, gelugrad_infershape_95_test)
{
    // set version
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    ge::op::GeluGrad op;

    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 16}, {1, 16}};

    auto tensor_desc = create_desc_shape_range({-1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {16, 16}, ge::FORMAT_ND,
                                               shape_range);

    std::vector<std::pair<int64_t, int64_t>> shape_range2 = {{1, 1}, {1, 16}};

    auto tensor_desc2 = create_desc_shape_range({-1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 16}, ge::FORMAT_ND,
                                                shape_range);
    op.UpdateInputDesc("dy", tensor_desc2);
    op.UpdateInputDesc("x", tensor_desc);
    op.UpdateInputDesc("y", tensor_desc2);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("z");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {-1, -1};
    EXPECT_EQ(output_desc.GetShape().GetDimNum(), 2);

    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);

    auto output_desc1 = op.GetOutputDesc(0);
    EXPECT_EQ(output_desc1.GetShape().GetDimNum(), 2);
}

TEST_F(gelugrad, gelugrad_infershape_95_test2)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GeluGrad")->infer_shape;

    gert::Shape input_shape_dy = {1, 16};
    gert::Shape input_shape_x = {16, 16};
    gert::Shape input_shape_y = {1, 16};
    gert::Shape output_shape_z = {16, 16};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape_dy, &input_shape_x, &input_shape_y})
                      .OutputShapes({&output_shape_z})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}
TEST_F(gelugrad, gelugrad_infershape_910_93)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_93";
    optiCompilationInfo.soc_version = "Ascend910_93";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GeluGrad")->infer_shape;

    gert::Shape input_shape_dy = {1, 16};
    gert::Shape input_shape_x = {16, 16};
    gert::Shape input_shape_y = {1, 16};
    gert::Shape output_shape_z = {16, 16};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape_dy, &input_shape_x, &input_shape_y})
                      .OutputShapes({&output_shape_z})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(gelugrad, gelugrad_infershape_95_float16_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GeluGrad")->infer_shape;

    gert::Shape input_shape_dy = {32, 64, 128};
    gert::Shape input_shape_x = {32, 64, 128};
    gert::Shape input_shape_y = {32, 64, 128};
    gert::Shape output_shape_z = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape_dy, &input_shape_x, &input_shape_y})
                      .OutputShapes({&output_shape_z})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(gelugrad, gelugrad_infershape_95_4d_nchw_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GeluGrad")->infer_shape;

    gert::Shape input_shape_dy = {2, 64, 112, 112};
    gert::Shape input_shape_x = {2, 64, 112, 112};
    gert::Shape input_shape_y = {2, 64, 112, 112};
    gert::Shape output_shape_z = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape_dy, &input_shape_x, &input_shape_y})
                      .OutputShapes({&output_shape_z})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(gelugrad, gelugrad_infershape_95_5d_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GeluGrad")->infer_shape;

    gert::Shape input_shape_dy = {2, 4, 8, 16, 32};
    gert::Shape input_shape_x = {2, 4, 8, 16, 32};
    gert::Shape input_shape_y = {2, 4, 8, 16, 32};
    gert::Shape output_shape_z = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape_dy, &input_shape_x, &input_shape_y})
                      .OutputShapes({&output_shape_z})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(gelugrad, gelugrad_infershape_95_empty_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GeluGrad")->infer_shape;

    gert::Shape input_shape_dy = {0, 0};
    gert::Shape input_shape_x = {0, 0};
    gert::Shape input_shape_y = {0, 0};
    gert::Shape output_shape_z = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape_dy, &input_shape_x, &input_shape_y})
                      .OutputShapes({&output_shape_z})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(gelugrad, gelugrad_infershape_95_scalar_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GeluGrad")->infer_shape;

    gert::Shape input_shape_dy = {};
    gert::Shape input_shape_x = {};
    gert::Shape input_shape_y = {};
    gert::Shape output_shape_z = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape_dy, &input_shape_x, &input_shape_y})
                      .OutputShapes({&output_shape_z})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(gelugrad, gelugrad_infershape_95_large_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GeluGrad")->infer_shape;

    gert::Shape input_shape_dy = {1024, 1024};
    gert::Shape input_shape_x = {1024, 1024};
    gert::Shape input_shape_y = {1024, 1024};
    gert::Shape output_shape_z = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape_dy, &input_shape_x, &input_shape_y})
                      .OutputShapes({&output_shape_z})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}