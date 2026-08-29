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
#include "infershape_test_util.h"
#include "platform/platform_info.h"
#include "ut_op_common.h"

using namespace ge;

namespace {
void SetPlatform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

void ExpectShapeEq(const gert::Shape& actual, const gert::Shape& expected)
{
    ASSERT_EQ(actual.GetDimNum(), expected.GetDimNum());
    for (size_t i = 0; i < expected.GetDimNum(); ++i) {
        EXPECT_EQ(actual.GetDim(i), expected.GetDim(i));
    }
}

void CheckInferShape(const gert::Shape& inputShape)
{
    SetPlatform();
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("BNInfer");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = inputShape;
    gert::Shape paramShape = {inputShape.GetDimNum() > 1 ? inputShape.GetDim(1) : -1};
    gert::Shape outputShape = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1})
                      .InputShapes({&xShape, &paramShape, &paramShape, &paramShape, &paramShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(outputDesc, nullptr);
    ExpectShapeEq(*outputDesc, inputShape);
}
} // namespace

class BNInferInferShapeTest : public testing::Test {};

TEST_F(BNInferInferShapeTest, staticShape) { CheckInferShape({2, 3, 4}); }

TEST_F(BNInferInferShapeTest, dynamicShape) { CheckInferShape({2, -1, 16}); }

TEST_F(BNInferInferShapeTest, nchwLikeShape) { CheckInferShape({2, 3, 4, 5}); }

TEST_F(BNInferInferShapeTest, ndhwcLikeShape) { CheckInferShape({2, 3, 4, 5, 6}); }

TEST_F(BNInferInferShapeTest, unknownRank) { CheckInferShape({-2}); }

void CheckInferDataType(ge::DataType xDataType)
{
    SetPlatform();
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("BNInfer");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1})
                      .NodeInputTd(0, xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), xDataType);
}

TEST_F(BNInferInferShapeTest, inferDataTypeFloat32) { CheckInferDataType(ge::DT_FLOAT); }

TEST_F(BNInferInferShapeTest, inferDataTypeFloat16) { CheckInferDataType(ge::DT_FLOAT16); }

TEST_F(BNInferInferShapeTest, inferDataTypeBfloat16) { CheckInferDataType(ge::DT_BF16); }
