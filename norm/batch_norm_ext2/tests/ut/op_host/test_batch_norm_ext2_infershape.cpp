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
#include <vector>
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "array_ops.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_info.h"
#include "../../../op_graph/batch_norm_ext2_proto.h"

using namespace ge;

namespace {
static std::vector<int64_t> ShapeToVec(const gert::Shape& s)
{
    std::vector<int64_t> v;
    for (size_t i = 0; i < s.GetDimNum(); i++) {
        v.push_back(s.GetDim(i));
    }
    return v;
}
} // namespace

class BatchNormExt2InfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BatchNormExt2Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BatchNormExt2Test TearDown" << std::endl; }
};

TEST_F(BatchNormExt2InfershapeTest, batch_norm_ext2_infershape_test_0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNormExt2"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNormExt2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape input_x_shape = {2, 3, 4, 5};
    gert::Shape input_scale_shape = {5};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1, 1, 1, 1, 1})
                      .InputShapes({&input_x_shape, &input_scale_shape, &input_scale_shape, &input_scale_shape,
                                    &input_scale_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*output0), "[2, 3, 4, 5]");
    ASSERT_EQ(Ops::Base::ToString(*output1), "[5]");
}

// 动态 rank：x 为 -2(UNKNOWN_RANK) 时，5 个输出都必须是未知秩。
// def 声明了 DynamicRankSupportFlag(true)，infershape 需有 IsUnknownRank 分支早退，
// 否则 GetDim(i) 取到 -2 标记值会把输出推成非法形状（红线 R4）。
TEST_F(BatchNormExt2InfershapeTest, batch_norm_ext2_infershape_unknown_rank_test_0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNormExt2"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNormExt2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape input_x_shape = {-2};
    gert::Shape input_scale_shape = {-2};
    gert::Shape y_shape = {};
    gert::Shape mean_shape = {};
    gert::Shape variance_shape = {};
    gert::Shape rs1_shape = {};
    gert::Shape rs2_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1, 1, 1, 1, 1})
                      .InputShapes({&input_x_shape, &input_scale_shape, &input_scale_shape, &input_scale_shape,
                                    &input_scale_shape})
                      .OutputShapes({&y_shape, &mean_shape, &variance_shape, &rs1_shape, &rs2_shape})
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto ctx = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(ShapeToVec(*ctx->GetOutputShape(0)), (std::vector<int64_t>{-2}));
    EXPECT_EQ(ShapeToVec(*ctx->GetOutputShape(1)), (std::vector<int64_t>{-2}));
    EXPECT_EQ(ShapeToVec(*ctx->GetOutputShape(2)), (std::vector<int64_t>{-2}));
    EXPECT_EQ(ShapeToVec(*ctx->GetOutputShape(3)), (std::vector<int64_t>{-2}));
    EXPECT_EQ(ShapeToVec(*ctx->GetOutputShape(4)), (std::vector<int64_t>{-2}));
}

TEST_F(BatchNormExt2InfershapeTest, batch_norm_ext2_inferdtype_test_0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNormExt2"), nullptr);
    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNormExt2")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    ge::DataType dfp16 = ge::DT_FLOAT16;
    ge::DataType dfp32 = ge::DT_FLOAT;

    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(5)
                              .NodeIoNum(5, 5)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                              .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                              .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                              .InputDataTypes({&dfp16, &dfp32, &dfp32, &dfp32, &dfp32})
                              .OutputDataTypes({&dfp16, &dfp32, &dfp32, &dfp32, &dfp32})
                              .Build();

    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), dfp16);
    EXPECT_EQ(context->GetInputDataType(1), dfp32);
}
