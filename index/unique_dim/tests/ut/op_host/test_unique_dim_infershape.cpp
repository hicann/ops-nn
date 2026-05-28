/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

class UniqueDimInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UniqueDimInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UniqueDimInfershapeTest TearDown" << std::endl;
    }
};

// Basic test: 2D input, check output shapes
TEST_F(UniqueDimInfershapeTest, unique_dim_infershape_basic_2d)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueDim"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueDim")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape input_x_shape = {5, 3};
    gert::Shape output_y_shape = {};
    gert::Shape output_idx_shape = {};
    gert::Shape output_count_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInputNum({1})
                      .InputShapes({&input_x_shape})
                      .OutputShapes({&output_y_shape, &output_idx_shape, &output_count_shape})
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"sorted", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"return_inverse", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"return_counts", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto *yShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto *idxShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto *countShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);

    // y: same rank as input (2D), all dims set to -1 by SetUnknownShape
    gert::Shape expected_y = {-1, -1};
    ASSERT_EQ(Ops::Base::ToString(*yShape), Ops::Base::ToString(expected_y));

    // idx: [numInp] = [5]
    gert::Shape expected_idx = {5};
    ASSERT_EQ(Ops::Base::ToString(*idxShape), Ops::Base::ToString(expected_idx));

    // count: [-1] (unknown)
    gert::Shape expected_count = {-1};
    ASSERT_EQ(Ops::Base::ToString(*countShape), Ops::Base::ToString(expected_count));
}

// 1D input
TEST_F(UniqueDimInfershapeTest, unique_dim_infershape_1d)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueDim")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape input_x_shape = {10};
    gert::Shape output_y_shape = {};
    gert::Shape output_idx_shape = {};
    gert::Shape output_count_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInputNum({1})
                      .InputShapes({&input_x_shape})
                      .OutputShapes({&output_y_shape, &output_idx_shape, &output_count_shape})
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"sorted", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"return_inverse", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"return_counts", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto *yShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto *idxShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto *countShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);

    gert::Shape expected_y = {-1};
    ASSERT_EQ(Ops::Base::ToString(*yShape), Ops::Base::ToString(expected_y));

    gert::Shape expected_idx = {10};
    ASSERT_EQ(Ops::Base::ToString(*idxShape), Ops::Base::ToString(expected_idx));

    gert::Shape expected_count = {-1};
    ASSERT_EQ(Ops::Base::ToString(*countShape), Ops::Base::ToString(expected_count));
}

// 3D input
TEST_F(UniqueDimInfershapeTest, unique_dim_infershape_3d)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueDim")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape input_x_shape = {4, 6, 8};
    gert::Shape output_y_shape = {};
    gert::Shape output_idx_shape = {};
    gert::Shape output_count_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInputNum({1})
                      .InputShapes({&input_x_shape})
                      .OutputShapes({&output_y_shape, &output_idx_shape, &output_count_shape})
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"sorted", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"return_inverse", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"return_counts", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto *yShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto *idxShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto *countShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);

    gert::Shape expected_y = {-1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*yShape), Ops::Base::ToString(expected_y));

    gert::Shape expected_idx = {4};
    ASSERT_EQ(Ops::Base::ToString(*idxShape), Ops::Base::ToString(expected_idx));

    gert::Shape expected_count = {-1};
    ASSERT_EQ(Ops::Base::ToString(*countShape), Ops::Base::ToString(expected_count));
}

// Empty input
TEST_F(UniqueDimInfershapeTest, unique_dim_infershape_empty_input)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueDim")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape input_x_shape = {0, 5};
    gert::Shape output_y_shape = {};
    gert::Shape output_idx_shape = {};
    gert::Shape output_count_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInputNum({1})
                      .InputShapes({&input_x_shape})
                      .OutputShapes({&output_y_shape, &output_idx_shape, &output_count_shape})
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"sorted", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"return_inverse", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"return_counts", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto *yShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto *idxShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto *countShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);

    // empty input -> all outputs are empty
    gert::Shape expected_y = {0, 0};
    ASSERT_EQ(Ops::Base::ToString(*yShape), Ops::Base::ToString(expected_y));

    gert::Shape expected_idx = {0};
    ASSERT_EQ(Ops::Base::ToString(*idxShape), Ops::Base::ToString(expected_idx));

    gert::Shape expected_count = {0};
    ASSERT_EQ(Ops::Base::ToString(*countShape), Ops::Base::ToString(expected_count));
}

// dim != 0 should fail
TEST_F(UniqueDimInfershapeTest, unique_dim_infershape_invalid_dim)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueDim")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape input_x_shape = {4, 5, 6};
    gert::Shape output_y_shape = {};
    gert::Shape output_idx_shape = {};
    gert::Shape output_count_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInputNum({1})
                      .InputShapes({&input_x_shape})
                      .OutputShapes({&output_y_shape, &output_idx_shape, &output_count_shape})
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"sorted", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"return_inverse", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"return_counts", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// InferDataType test
TEST_F(UniqueDimInfershapeTest, unique_dim_infer_dtype_float)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueDim"), nullptr);
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueDim")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(1)
                      .NodeIoNum(1, 3)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT64);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_INT64);
}

// InferDataType test with float16
TEST_F(UniqueDimInfershapeTest, unique_dim_infer_dtype_float16)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueDim"), nullptr);
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueDim")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(1)
                      .NodeIoNum(1, 3)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT64);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_INT64);
}
