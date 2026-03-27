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
#include "array_ops.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/reverse_sequence_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class ReverseSequenceInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReverseSequenceInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReverseSequenceInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(ReverseSequenceInfershapeTest, reverse_sequence_infershape_float16_2d)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ReverseSequence")->infer_shape;

    gert::Shape input_x_shape = {3, 5};
    gert::Shape input_seq_lengths_shape = {3};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_x_shape, &input_seq_lengths_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"seq_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"batch_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(output_shape.GetDimNum(), 2);
    ASSERT_EQ(output_shape.GetDim(0), 3);
    ASSERT_EQ(output_shape.GetDim(1), 5);
}

TEST_F(ReverseSequenceInfershapeTest, reverse_sequence_infershape_float32_4d)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ReverseSequence")->infer_shape;

    gert::Shape input_x_shape = {2, 3, 4, 5};
    gert::Shape input_seq_lengths_shape = {2};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_x_shape, &input_seq_lengths_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"seq_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"batch_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(output_shape.GetDimNum(), 4);
    ASSERT_EQ(output_shape.GetDim(0), 2);
    ASSERT_EQ(output_shape.GetDim(1), 3);
    ASSERT_EQ(output_shape.GetDim(2), 4);
    ASSERT_EQ(output_shape.GetDim(3), 5);
}

TEST_F(ReverseSequenceInfershapeTest, reverse_sequence_infershape_int32_3d)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ReverseSequence")->infer_shape;

    gert::Shape input_x_shape = {4, 6, 8};
    gert::Shape input_seq_lengths_shape = {4};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_x_shape, &input_seq_lengths_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"seq_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                  {"batch_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(output_shape.GetDimNum(), 3);
    ASSERT_EQ(output_shape.GetDim(0), 4);
    ASSERT_EQ(output_shape.GetDim(1), 6);
    ASSERT_EQ(output_shape.GetDim(2), 8);
}

TEST_F(ReverseSequenceInfershapeTest, reverse_sequence_infershape_bfloat16_5d)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ReverseSequence")->infer_shape;

    gert::Shape input_x_shape = {2, 3, 4, 5, 6};
    gert::Shape input_seq_lengths_shape = {3};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_x_shape, &input_seq_lengths_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"seq_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"batch_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(output_shape.GetDimNum(), 5);
    ASSERT_EQ(output_shape.GetDim(0), 2);
    ASSERT_EQ(output_shape.GetDim(1), 3);
    ASSERT_EQ(output_shape.GetDim(2), 4);
    ASSERT_EQ(output_shape.GetDim(3), 5);
    ASSERT_EQ(output_shape.GetDim(4), 6);
}

TEST_F(ReverseSequenceInfershapeTest, reverse_sequence_infershape_int8_2d)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ReverseSequence")->infer_shape;

    gert::Shape input_x_shape = {10, 20};
    gert::Shape input_seq_lengths_shape = {10};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_x_shape, &input_seq_lengths_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"seq_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"batch_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(output_shape.GetDimNum(), 2);
    ASSERT_EQ(output_shape.GetDim(0), 10);
    ASSERT_EQ(output_shape.GetDim(1), 20);
}
