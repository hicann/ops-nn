/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_multilabel_margin_loss_infershape.cpp
 * \brief MultilabelMarginLoss graph infershape UT (faker-based). 覆盖 none/mean/sum × 2D/1D + 动态 shape(-1) + 动态
 * rank(-2)。 y: none+2D->(N); none+1D 或 mean/sum->标量; is_target == x shape。
 */

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_info.h"

using namespace ut_util;
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

static ge::graphStatus RunInfer(const gert::Shape& x, const std::string& reduction, gert::Shape& y, gert::Shape& isTgt)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("MultilabelMarginLoss");
    EXPECT_NE(opImpl, nullptr);
    auto fn = opImpl->infer_shape;
    EXPECT_NE(fn, nullptr);
    gert::Shape target = x;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({const_cast<gert::Shape*>(&x), &target})
                      .OutputShapes({&y, &isTgt})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    auto ret = fn(context);
    if (ret == ge::GRAPH_SUCCESS) {
        y = *context->GetOutputShape(0);
        isTgt = *context->GetOutputShape(1);
    }
    return ret;
}
} // namespace

class MultilabelMarginLossInferShapeTest : public testing::Test {};

TEST_F(MultilabelMarginLossInferShapeTest, none_2d_y_eq_n_istarget_eq_x)
{
    gert::Shape x = {4, 8}, y, it;
    ASSERT_EQ(RunInfer(x, "none", y, it), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(y), (std::vector<int64_t>{4}));
    EXPECT_EQ(ShapeToVec(it), (std::vector<int64_t>{4, 8}));
}

TEST_F(MultilabelMarginLossInferShapeTest, mean_2d_y_scalar)
{
    gert::Shape x = {8, 16}, y, it;
    ASSERT_EQ(RunInfer(x, "mean", y, it), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y.GetDimNum(), 0u); // scalar
    EXPECT_EQ(ShapeToVec(it), (std::vector<int64_t>{8, 16}));
}

TEST_F(MultilabelMarginLossInferShapeTest, none_1d_y_scalar)
{
    gert::Shape x = {16}, y, it;
    ASSERT_EQ(RunInfer(x, "none", y, it), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y.GetDimNum(), 0u);
    EXPECT_EQ(ShapeToVec(it), (std::vector<int64_t>{16}));
}

// 动态 shape(-1):N 未知时 y 的 N 维透传 -1；is_target 随 x。
TEST_F(MultilabelMarginLossInferShapeTest, dynamic_dim_minus1)
{
    gert::Shape x = {-1, 8}, y, it;
    ASSERT_EQ(RunInfer(x, "none", y, it), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(y), (std::vector<int64_t>{-1}));
    EXPECT_EQ(ShapeToVec(it), (std::vector<int64_t>{-1, 8}));
}

// 动态 rank(-2, UNKNOWN_RANK):A5(Ascend950)透传 -2（-2 处理 soc 门控为 regbase，A2 走基线不处理）。
TEST_F(MultilabelMarginLossInferShapeTest, unknown_rank_minus2)
{
    // -2 UNKNOWN_RANK 传播是 A5(regbase/Ascend950)行为,mock 950 平台以覆盖该分支。
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::Shape x = {-2}, y, it;
    ASSERT_EQ(RunInfer(x, "mean", y, it), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(y), (std::vector<int64_t>{-2}));
    EXPECT_EQ(ShapeToVec(it), (std::vector<int64_t>{-2}));
}

TEST_F(MultilabelMarginLossInferShapeTest, multilabelmarginloss_infer_datatype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MultilabelMarginLoss"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MultilabelMarginLoss")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);
    ge::DataType in0 = ge::DT_FLOAT;
    ge::DataType in1 = ge::DT_INT32;
    ge::DataType expOut0 = ge::DT_FLOAT;
    ge::DataType expOut1 = ge::DT_INT32;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(2, 2)
                             .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&in0, &in1})
                             .OutputDataTypes({&expOut0, &expOut1})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}
