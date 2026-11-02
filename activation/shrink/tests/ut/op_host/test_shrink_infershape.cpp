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
 * \file test_shrink_infershape.cpp
 * \brief Shrink InferShape UT：输出 shape 等于输入 shape 的透传契约，
 *        覆盖 fp16/fp32、标量（rank-0）、空 Tensor 与高秩传播。
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/shrink_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class ShrinkProtoTest : public testing::Test {
public:
    // public：SetUp/TearDown/SetPlatform 需被命名空间作用域的 RunInferShape/
    // AssertShapeEquals 自由函数调用；UT 构建的 -fno-access-control 会掩盖
    // protected 违例，此处按标准 C++ 显式开放访问。
    static void SetUpTestCase() { std::cout << "shrink Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "shrink Proto Test TearDown" << std::endl; }

    static void SetPlatform()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

// 输出 shape 须与输入逐维一致（透传契约）
static void AssertShapeEquals(const gert::Shape* actual, const gert::Shape& expected)
{
    ASSERT_NE(actual, nullptr);
    ASSERT_EQ(actual->GetDimNum(), expected.GetDimNum());
    for (size_t i = 0; i < expected.GetDimNum(); i++) {
        ASSERT_EQ(actual->GetDim(i), expected.GetDim(i));
    }
}

static void RunInferShape(gert::Shape& inputShape, gert::Shape& outputShape, ge::DataType dtype,
                          const gert::Shape& expectedShape)
{
    ShrinkProtoTest::SetPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Shrink")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&inputShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    AssertShapeEquals(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0), expectedShape);
}

TEST_F(ShrinkProtoTest, shrink_infershape_float16_test)
{
    gert::Shape inputShape = {4, 3, 4};
    gert::Shape outputShape = {};
    RunInferShape(inputShape, outputShape, ge::DT_FLOAT16, inputShape);
}

TEST_F(ShrinkProtoTest, shrink_infershape_float_test)
{
    gert::Shape inputShape = {1, 3, 4};
    gert::Shape outputShape = {};
    RunInferShape(inputShape, outputShape, ge::DT_FLOAT, inputShape);
}

// 标量输入（rank-0）：输出保持 rank-0
TEST_F(ShrinkProtoTest, shrink_infershape_scalar_test)
{
    gert::Shape inputShape = {};
    gert::Shape outputShape = {};
    RunInferShape(inputShape, outputShape, ge::DT_FLOAT, inputShape);
}

// 空 Tensor（0 元素）传播：输出保持同 shape 的空 Tensor
TEST_F(ShrinkProtoTest, shrink_infershape_empty_test)
{
    gert::Shape inputShape = {2, 0, 3};
    gert::Shape outputShape = {};
    RunInferShape(inputShape, outputShape, ge::DT_FLOAT16, inputShape);
}

// 高秩（8 维）ND 输入：输出逐维一致传播
TEST_F(ShrinkProtoTest, shrink_infershape_high_rank_test)
{
    gert::Shape inputShape = {2, 2, 2, 2, 2, 2, 2, 2};
    gert::Shape outputShape = {};
    RunInferShape(inputShape, outputShape, ge::DT_FLOAT, inputShape);
}
