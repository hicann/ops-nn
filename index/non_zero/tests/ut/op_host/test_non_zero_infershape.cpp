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

class NonZeroProtoTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NonZeroProtoTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NonZeroProtoTest TearDown" << std::endl;
    }
};

TEST_F(NonZeroProtoTest, nonzero_test_1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x_shape = {-2};

    gert::Shape output_shape = {-2};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(NonZeroProtoTest, nonzero_test_2)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x_shape = {2, 2};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {-1, 2};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(NonZeroProtoTest, nonzero_test_3)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x_shape = {2, -1};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {-1, 2};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(NonZeroProtoTest, nonzero_test_4)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x_shape = {2, 2};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {2, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(NonZeroProtoTest, nonzero_test_5_1d_tensor)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x_shape = {10};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {-1, 1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(NonZeroProtoTest, nonzero_test_6_multidim)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x_shape = {2, 3, 4};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {-1, 3};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(NonZeroProtoTest, nonzero_test_7_multidim_transpose)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x_shape = {2, 3, 4, 5};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {4, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(NonZeroProtoTest, nonzero_inferdatatype_test_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero"), nullptr);
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(1)
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
}

TEST_F(NonZeroProtoTest, nonzero_inferdatatype_test_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero"), nullptr);
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(1)
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(5)}})
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
}

TEST_F(NonZeroProtoTest, nonzero_inferdatatype_test_3)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero"), nullptr);
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(1)
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)}})
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
}

TEST_F(NonZeroProtoTest, nonzero_inferdatatype_nullptr_test)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero"), nullptr);
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    ASSERT_EQ(inferDataTypeFunc(nullptr), ge::GRAPH_FAILED);
}

TEST_F(NonZeroProtoTest, nonzero_infershape_range_test_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero"), nullptr);
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero")->infer_shape_range;
    ASSERT_NE(inferShapeRangeFunc, nullptr);

    gert::Shape min1{2, 3, 4};
    gert::Shape max1{2, 3, 4};
    gert::Shape outmin{0, 3};
    gert::Shape outmax{24, 3};
    gert::Shape null1{};
    gert::Shape null2{};

    gert::Range<gert::Shape> in_shape_range(&min1, &max1);
    gert::Range<gert::Shape> out_shape_range(&null1, &null2);

    auto context_holder = gert::InferShapeRangeContextFaker()
                              .IrInputNum(1)
                              .NodeIoNum(1, 1)
                              .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputShapeRanges({&in_shape_range})
                              .OutputShapeRanges({&out_shape_range})
                              .NodeAttrs({{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                              .Build();

    auto context = context_holder.GetContext<gert::InferShapeRangeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeRangeFunc(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(0), 0);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(1), 3);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(0), 24);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(1), 3);
}

TEST_F(NonZeroProtoTest, nonzero_infershape_range_test_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero"), nullptr);
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZero")->infer_shape_range;
    ASSERT_NE(inferShapeRangeFunc, nullptr);

    gert::Shape min1{2, 3, 4};
    gert::Shape max1{2, 3, 4};
    gert::Shape outmin{3, 0};
    gert::Shape outmax{3, 24};
    gert::Shape null1{};
    gert::Shape null2{};

    gert::Range<gert::Shape> in_shape_range(&min1, &max1);
    gert::Range<gert::Shape> out_shape_range(&null1, &null2);

    auto context_holder = gert::InferShapeRangeContextFaker()
                              .IrInputNum(1)
                              .NodeIoNum(1, 1)
                              .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputShapeRanges({&in_shape_range})
                              .OutputShapeRanges({&out_shape_range})
                              .NodeAttrs({{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                              .Build();

    auto context = context_holder.GetContext<gert::InferShapeRangeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeRangeFunc(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(0), 3);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(1), 0);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(0), 3);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(1), 24);
}
