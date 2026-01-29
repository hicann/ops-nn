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

// ----------------NLLLoss--------------
class NLLLossProtoTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NLLLossProtoTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NLLLossProtoTest TearDown" << std::endl;
    }
};

TEST_F(NLLLossProtoTest, nll_loss_infershape_test1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLoss")->infer_shape;
    gert::Shape xShape = {3, 4};
    gert::Shape targetShape = {3};
    gert::Shape weightShape = {3};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &targetShape, &weightShape})
                      .OutputShapes({&output_shape, &output_shape})
                      .NodeAttrs(
                          {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("none")},
                           {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {3};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(NLLLossProtoTest, nll_loss_infershape_test2)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLoss")->infer_shape;
    gert::Shape xShape = {3, 4};
    gert::Shape targetShape = {3};
    gert::Shape weightShape = {3};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &targetShape, &weightShape})
                      .OutputShapes({&output_shape, &output_shape})
                      .NodeAttrs(
                          {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                           {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(NLLLossProtoTest, nll_loss_infershape_test3)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLoss")->infer_shape;
    gert::Shape xShape = {-1, -1};
    gert::Shape targetShape = {-1};
    gert::Shape weightShape = {-1};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &targetShape, &weightShape})
                      .OutputShapes({&output_shape, &output_shape})
                      .NodeAttrs(
                          {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("none")},
                           {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {-1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(NLLLossProtoTest, nll_loss_infershape_test4)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLoss")->infer_shape;
    gert::Shape xShape = {3, 4, 5, 6};
    gert::Shape targetShape = {3, 5, 6};
    gert::Shape weightShape = {4};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &targetShape, &weightShape})
                      .OutputShapes({&output_shape, &output_shape})
                      .NodeAttrs(
                          {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("none")},
                           {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {3, 5, 6};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(NLLLossProtoTest, nll_loss_infershape_test5)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLoss")->infer_shape;
    gert::Shape xShape = {-1, -1, -1, -1};
    gert::Shape targetShape = {-1, -1, -1};
    gert::Shape weightShape = {-1};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &targetShape, &weightShape})
                      .OutputShapes({&output_shape, &output_shape})
                      .NodeAttrs(
                          {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("none")},
                           {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {-1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(NLLLossProtoTest, nll_loss_infershape_test6)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLoss")->infer_shape;
    gert::Shape xShape = {-2};
    gert::Shape targetShape = {-2};
    gert::Shape weightShape = {-2};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &targetShape, &weightShape})
                      .OutputShapes({&output_shape, &output_shape})
                      .NodeAttrs(
                          {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("none")},
                           {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(NLLLossProtoTest, nll_loss_infershape_test7)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLoss")->infer_shape;
    gert::Shape xShape = {3, 4, 5};
    gert::Shape targetShape = {3, 4, 5};
    gert::Shape weightShape = {3, 4, 5};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &targetShape, &weightShape})
                      .OutputShapes({&output_shape, &output_shape})
                      .NodeAttrs(
                          {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                           {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(NLLLossProtoTest, success_dtype_infer_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLoss"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLoss")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType output_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_FLOAT;
        ge::DataType input_ref2 = ge::DT_INT32;
        auto holder = gert::InferDataTypeContextFaker()
                          .NodeIoNum(3, 2)
                          .IrInstanceNum({1, 1, 1})
                          .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs(
                              {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                               {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                          .InputDataTypes({&input_ref1, &input_ref2, &input_ref1})
                          .OutputDataTypes({&output_ref, &output_ref})
                          .Build();

        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(NLLLossProtoTest, fail_dtype_infer_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLoss"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLoss")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType output_ref = ge::DT_INT32;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType input_ref2 = ge::DT_INT32;
        auto holder = gert::InferDataTypeContextFaker()
                          .NodeIoNum(3, 2)
                          .IrInstanceNum({1, 1, 1})
                          .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs(
                              {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                               {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                          .InputDataTypes({&input_ref1, &input_ref2, &input_ref1})
                          .OutputDataTypes({&output_ref, &output_ref})
                          .Build();

        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}