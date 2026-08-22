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
 * \file test_inplace_add_infershape.cpp
 * \brief InferShape UT —— y 的 shape 恒等于 x，indices/v 不参与推导。
 *
 *   InferDataType（y 的 dtype 恒随 x）已按交付件划分挪到
 *   op_graph/inplace_add_graph_infer.cpp。op_graph UT 模块只链 graph_plugin_obj，
 *   不含 tests/ut/common 的 infershape 公共对象，此处调不到 infer_datatype；
 *   与仓内其他把 InferDataType 放 op_graph 的算子保持一致，本文件不再覆盖该分支。
 */

#include <gtest/gtest.h>

#include "infershape_test_util.h"
#include "platform/platform_info.h"
#include "ut_op_common.h"

namespace {
void SetAscend950Platform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optionalInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

void ExpectShapeEq(const gert::Shape& actual, const gert::Shape& expected)
{
    ASSERT_EQ(actual.GetDimNum(), expected.GetDimNum());
    for (size_t i = 0; i < expected.GetDimNum(); ++i) {
        EXPECT_EQ(actual.GetDim(i), expected.GetDim(i));
    }
}

void CheckInferShapeOnlyUsesX(const gert::Shape& xInputShape, const gert::Shape& indicesInputShape,
                              const gert::Shape& vInputShape)
{
    SetAscend950Platform();
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceAdd");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);

    gert::Shape xShape = xInputShape;
    gert::Shape indicesShape = indicesInputShape;
    gert::Shape vShape = vInputShape;
    gert::Shape outputShape = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &indicesShape, &vShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(opImpl->infer_shape(context), ge::GRAPH_SUCCESS);
    auto inferredShape = context->GetOutputShape(0);
    ASSERT_NE(inferredShape, nullptr);
    ExpectShapeEq(*inferredShape, xInputShape);
}
} // namespace

class InplaceAddInferShapeTest : public testing::Test {};

TEST_F(InplaceAddInferShapeTest, staticShapeOnlyUsesX) { CheckInferShapeOnlyUsesX({4, 8}, {99, 1}, {7}); }

TEST_F(InplaceAddInferShapeTest, rankOne) { CheckInferShapeOnlyUsesX({4}, {2}, {2}); }

TEST_F(InplaceAddInferShapeTest, rankEight)
{
    CheckInferShapeOnlyUsesX({2, 2, 1, 1, 1, 1, 1, 3}, {1}, {1, 2, 1, 1, 1, 1, 1, 3});
}

TEST_F(InplaceAddInferShapeTest, dynamicShape) { CheckInferShapeOnlyUsesX({2, -1, 16}, {3}, {3, -1, 16}); }

TEST_F(InplaceAddInferShapeTest, unknownRank) { CheckInferShapeOnlyUsesX({-2}, {1}, {1}); }
