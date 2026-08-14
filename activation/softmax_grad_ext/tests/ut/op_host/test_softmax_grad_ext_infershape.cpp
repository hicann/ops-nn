/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class SoftmaxGradExtInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
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

static bool ShapeEquals(const gert::Shape& shape, const std::vector<int64_t>& expected)
{
    if (shape.GetDimNum() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (shape.GetDim(i) != expected[i]) {
            return false;
        }
    }
    return true;
}

// y.shape == grad.shape (input0)
TEST_F(SoftmaxGradExtInferShapeTest, infershape_shape_eq_grad_fp16_3d)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxGradExt");
    ASSERT_NE(opImpl, nullptr);

    gert::Shape gradShape = {2, 32, 128};
    gert::Shape x1Shape = {2, 32, 128};
    gert::Shape x2Shape = {2, 32, 128};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&gradShape, &x1Shape, &x2Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(opImpl->infer_shape(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context->GetOutputShape(0), nullptr);
    EXPECT_TRUE(ShapeEquals(*context->GetOutputShape(0), {2, 32, 128}));
}

TEST_F(SoftmaxGradExtInferShapeTest, infershape_shape_eq_grad_fp32_4d)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxGradExt");
    ASSERT_NE(opImpl, nullptr);

    gert::Shape gradShape = {1, 64, 256, 512};
    gert::Shape x1Shape = {1, 64, 256, 512};
    gert::Shape x2Shape = {1, 64, 256, 512};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&gradShape, &x1Shape, &x2Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(opImpl->infer_shape(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context->GetOutputShape(0), nullptr);
    EXPECT_TRUE(ShapeEquals(*context->GetOutputShape(0), {1, 64, 256, 512}));
}

// y.dtype == grad.dtype (input0)
TEST_F(SoftmaxGradExtInferShapeTest, infershape_dtype_eq_grad)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxGradExt");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    ge::DataType gradDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&gradDtype, &gradDtype, &gradDtype})
                      .OutputDataTypes({&yDtype})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}
