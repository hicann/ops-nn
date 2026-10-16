/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_selu_infershape.cpp
 * \brief Selu infershape UT: output shape == input shape (y.shape==x.shape), 各 rank/dtype 代表
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/selu_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class SeluInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SeluInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SeluInfershape TearDown" << std::endl; }
};

static void RunSeluInfershape(gert::Shape& inputShape, ge::DataType dtype, const std::vector<int64_t>& expectShape)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Selu"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Selu")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&inputShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    // faker 将 shape 拷入内部 tensor, 输出 shape 须从 context 读回断言
    const gert::Shape* outShape = context->GetOutputShape(0);
    ASSERT_NE(outShape, nullptr);
    ASSERT_EQ(outShape->GetDimNum(), expectShape.size());
    for (size_t i = 0; i < expectShape.size(); i++) {
        EXPECT_EQ(outShape->GetDim(i), expectShape[i]);
    }
}

TEST_F(SeluInfershape, selu_infershape_test1_static_fp32)
{
    gert::Shape inputShape = {3, 4};
    RunSeluInfershape(inputShape, ge::DT_FLOAT, {3, 4});
}

TEST_F(SeluInfershape, selu_infershape_test2_static_fp16_3d)
{
    gert::Shape inputShape = {2, 3, 4};
    RunSeluInfershape(inputShape, ge::DT_FLOAT16, {2, 3, 4});
}

TEST_F(SeluInfershape, selu_infershape_test3_static_bf16)
{
    gert::Shape inputShape = {8};
    RunSeluInfershape(inputShape, ge::DT_BF16, {8});
}

TEST_F(SeluInfershape, selu_infershape_test4_static_int32_4d)
{
    gert::Shape inputShape = {2, 4, 8, 16};
    RunSeluInfershape(inputShape, ge::DT_INT32, {2, 4, 8, 16});
}

TEST_F(SeluInfershape, selu_infershape_test5_static_int8)
{
    gert::Shape inputShape = {5, 7};
    RunSeluInfershape(inputShape, ge::DT_INT8, {5, 7});
}

TEST_F(SeluInfershape, selu_infershape_test6_dynamic_shape)
{
    gert::Shape inputShape = {5, -1};
    RunSeluInfershape(inputShape, ge::DT_FLOAT, {5, -1});
}

TEST_F(SeluInfershape, selu_infershape_test7_empty_tensor)
{
    gert::Shape inputShape = {0};
    RunSeluInfershape(inputShape, ge::DT_FLOAT, {0});
}

TEST_F(SeluInfershape, selu_infershape_test8_scalar)
{
    gert::Shape inputShape = {};
    RunSeluInfershape(inputShape, ge::DT_FLOAT, {});
}
